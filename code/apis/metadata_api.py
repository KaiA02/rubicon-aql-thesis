from typing import Optional, Tuple
import sqlite3
import wikipedia
from tabulate import tabulate

from enums import DataSource
from rag.embedding_model import SentenceTransformerEmbeddingModel
from rag.rag import LowConfidenceMatchError
from apis.database_api import (
    get_table_names,
    get_table_columns,
    rag_best_match,
)

# =========================================================
#  Embedding model
# =========================================================

EMBEDDING_MODEL = SentenceTransformerEmbeddingModel(
    "sentence-transformers/all-MiniLM-L6-v2"
)

wikipedia.set_lang("en")



# =========================================================
#  DATASOURCE REGISTRY (SINGLE SOURCE OF TRUTH)
# =========================================================

DATASOURCE_CAPABILITIES = {
    DataSource.DB: {
        "aliases": ["db", "database"],
        "description": "Relational database (tables, columns, schema introspection)",
    },
    DataSource.WIKIPEDIA: {
        "aliases": ["wiki", "wikipedia"],
        "description": "Wikipedia articles (metadata & search)",
    },
    DataSource.EMAIL: {
        "aliases": ["email", "mail", "emails"],
        "description": "Gmail inbox metadata (headers, attachments, filters)",
    },
    DataSource.PILE: {
        "aliases": ["pile", "llm", "chatgpt"],
        "description": "LLM corpus",
    },
}

# =========================================================
#  PARSE: "<datasource>[.<query>] ?"  OR  "?"
# =========================================================

def parse_metadata_query(
    user_input: str,
) -> Optional[Tuple[Optional[DataSource], Optional[str]]]:

    if not user_input:
        return None

    s = user_input.strip()
    if not s.endswith("?"):
        return None

    s = s[:-1].strip()  # remove '?'

    # -------------------------------------------------
    # GLOBAL METADATA → "?"
    # -------------------------------------------------
    if not s:
        return (None, None)

    # -------------------------------------------------
    # BUILD ALIAS MAP FROM REGISTRY
    # -------------------------------------------------
    alias_map = {}
    for ds, meta in DATASOURCE_CAPABILITIES.items():
        for a in meta["aliases"]:
            alias_map[a] = ds

    # -------------------------------------------------
    # db ?
    # -------------------------------------------------
    if "." not in s:
        ds = alias_map.get(s.lower())
        return (ds, None) if ds else None

    # -------------------------------------------------
    # db.query ?
    # -------------------------------------------------
    parts = [p.strip() for p in s.split(".", 1)]
    if len(parts) != 2:
        return None

    ds_raw, query = parts
    ds = alias_map.get(ds_raw.lower())
    if not ds:
        return None

    return ds, query or None

# =========================================================
#  GLOBAL METADATA
# =========================================================

def _global_metadata() -> str:
    trace = [
        "META(GLOBAL): REQUEST RECEIVED",
        "MODE: DISCOVERY",
        "LLM: DISABLED",
        "RAG: DISABLED",
    ]

    rows = []
    for ds, meta in DATASOURCE_CAPABILITIES.items():
        rows.append([
            ds.value,
            ", ".join(meta["aliases"]),
            meta["description"],
        ])

    return (
        tabulate(
            rows,
            headers=["datasource", "aliases", "description"],
            tablefmt="psql",
        )
        + "\n\nTRACE:\n" + "\n".join(trace)
    )

# =========================================================
#  DB helpers
# =========================================================

def get_column_candidates(db_path: str):
    out = []
    for table in get_table_names(db_path):
        for col in get_table_columns(db_path, table):
            out.append((f"{table}.{col}", table, col))
    return out


def get_column_samples(db_path: str, table: str, column: str, limit: int = 5):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT DISTINCT {column}
            FROM {table}
            WHERE {column} IS NOT NULL
            LIMIT {limit}
            """
        )
        vals = [str(r[0]) for r in cur.fetchall()]
    except Exception:
        vals = []
    conn.close()
    return vals

# =========================================================
#  DB METADATA
# =========================================================

def _db_metadata(db_path: str, query: Optional[str]) -> str:
    trace = [
        "META(DB): REQUEST RECEIVED",
        "LLM: DISABLED (METADATA MODE)",
        "RAG: EMBEDDINGS = all-MiniLM-L6-v2",
    ]

    tables = get_table_names(db_path)

    if query is None:
        trace.append("META(DB): FULL SCHEMA OVERVIEW (NO RAG)")
        rows = [[t, ", ".join(get_table_columns(db_path, t))] for t in tables]
        return (
            tabulate(rows, headers=["table", "columns"], tablefmt="psql")
            + "\n\nTRACE:\n" + "\n".join(trace)
        )

    trace.append(f"META(DB): OBJECT QUERY '{query}'")

    table_labels = tables
    column_candidates = get_column_candidates(db_path)
    column_labels = [c[0] for c in column_candidates]

    all_labels = table_labels + column_labels
    all_types = (
        ["table"] * len(table_labels)
        + ["column"] * len(column_labels)
    )

    trace.append("RAG: RESOLVER START (TABLE + COLUMN)")

    try:
        match = rag_best_match(query, all_labels, EMBEDDING_MODEL)
    except LowConfidenceMatchError:
        match = None

    if not match:
        trace.append(f"RAG FAILED for query '{query}' (LOW CONFIDENCE)")
        return "No confident schema match found.\n\nTRACE:\n" + "\n".join(trace)

    idx = all_labels.index(match)
    kind = all_types[idx]

    trace.append(f"RAG RESOLVED TYPE: {kind.upper()}")
    trace.append(f"RAG RESOLVED OBJECT: {match}")

    if kind == "table":
        cols = get_table_columns(db_path, match)
        return (
            tabulate([[match, ", ".join(cols)]],
                     headers=["table", "columns"],
                     tablefmt="psql")
            + "\n\nTRACE:\n" + "\n".join(trace)
        )

    table, column = match.split(".", 1)
    samples = get_column_samples(db_path, table, column)

    rows = [
        ["table", table],
        ["column", column],
        ["sample values", ", ".join(samples) or "(none)"],
    ]

    return (
        tabulate(rows, headers=["property", "value"], tablefmt="psql")
        + "\n\nTRACE:\n" + "\n".join(trace)
    )



# =========================================================
#  WIKIPEDIA METADATA MODEL (INTROSPECTION ONLY)
# =========================================================
#MediaWiki Action API
WIKI_SCHEMA = {
    "page": {
        "description": "Wikipedia article page",
        "attributes": {
            "pageid": {
                "type": "int",
                "description": "Unique numeric Wikipedia page ID",
                "source": "query:info",
            },
            "title": {
                "type": "string",
                "description": "Canonical article title",
                "source": "query:info",
            },
            "url": {
                "type": "string",
                "description": "Canonical article URL (fullurl)",
                "source": "query:info+inprop=url",
            },
            "abstract": {
                "type": "string",
                "description": "Introductory summary (lead section)",
                "source": "query:extracts+exintro",
            },
            "text": {
                "type": "string",
                "description": "Full article text (plain)",
                "source": "query:extracts",
            },
            "images": {
                "type": "list[string]",
                "description": "Image file titles / URLs (best-effort)",
                "source": "query:images|pageimages",
            },
            "links": {
                "type": "list[string]",
                "description": "Internal Wikipedia links",
                "source": "query:links",
            },
            "categories": {
                "type": "list[string]",
                "description": "Page categories",
                "source": "query:categories",
            },
            "sources": {
                "type": "list[string]",
                "description": "Reference URLs (best-effort via HTML parsing)",
                "source": "parse:html+references",
            },
            "last_modified": {
                "type": "datetime",
                "description": "Last revision timestamp",
                "source": "query:revisions+timestamp",
            },
        },
    }
}

# =========================================================
#  WIKIPEDIA METADATA HELPERS
# =========================================================

def _wiki_global_metadata() -> str:
    attrs = WIKI_SCHEMA["page"]["attributes"]
    rows = [
        [a, info.get("type", ""), info.get("description", ""), info.get("source", "")]
        for a, info in attrs.items()
    ]
    return tabulate(
        rows,
        headers=["attribute", "type", "description", "source"],
        tablefmt="psql",
    )


def _wiki_attribute_metadata(attr: str) -> str:
    attrs = WIKI_SCHEMA["page"]["attributes"]
    if attr in attrs:
        info = attrs[attr]
        rows = [
            ["attribute", attr],
            ["type", info.get("type", "")],
            ["description", info.get("description", "")],
            ["source", info.get("source", "")],
        ]
        return tabulate(rows, headers=["property", "value"], tablefmt="psql")

    return (
        f"Unknown Wikipedia attribute '{attr}'.\n"
        f"Try: wiki ?\n"
        f"Known attributes: {', '.join(sorted(attrs.keys()))}"
    )


# =========================================================
#  WIKIPEDIA METADATA
# =========================================================

def _wikipedia_metadata(query: Optional[str]) -> str:
    trace = [
        "META(WIKI): REQUEST RECEIVED",
        "MODE: INTROSPECTION",
        "LLM: DISABLED (METADATA MODE)",
        "RAG: DISABLED",
    ]

    if not query:
        return _wiki_global_metadata() + "\n\nTRACE:\n" + "\n".join(trace)

    return _wiki_attribute_metadata(query.lower()) + "\n\nTRACE:\n" + "\n".join(trace)



# =========================================================
#  PILE METADATA (LLM ?)
# =========================================================

def _pile_metadata(query: Optional[str]) -> str:
    trace = [
        "META(LLM): REQUEST RECEIVED",
        "LLM: DISABLED (METADATA MODE)",
        "RAG: DISABLED",
    ]

    # -------------------------------------------------
    # GLOBAL: llm ?
    # -------------------------------------------------
    if query is None:
        rows = [
            ["text", "Short neutral summaries (2–4 sentences)"],
            ["table", "Structured JSON rendered dynamically"],
            ["filtering", "Contextual filtering via prompt"],
            ["synthesis", "Abstracted knowledge without raw citations"],
            ["limitations", "No browsing, no raw sources, no guaranteed schema"],
        ]

        return (
            tabulate(
                rows,
                headers=["capability", "description"],
                tablefmt="psql",
            )
            + "\n\nTRACE:\n"
            + "\n".join(trace)
        )

    q = query.lower().strip()

    # -------------------------------------------------
    # llm.text ?
    # -------------------------------------------------
    if q == "text":
        rows = [
            ["output type", "Plain text"],
            ["length", "2–4 sentences"],
            ["tone", "Neutral, factual"],
            ["formatting", "No bullet points, no greetings"],
            ["citations", "Not provided"],
        ]

        return (
            tabulate(
                rows,
                headers=["property", "value"],
                tablefmt="psql",
            )
            + "\n\nTRACE:\n"
            + "\n".join(trace)
        )

    # -------------------------------------------------
    # llm.table ?
    # -------------------------------------------------
    if q == "table":
        rows = [
            ["output structure", "JSON object rendered as blocks"],
            ["item count", "Determined dynamically by the model"],
            ["fields", "Model-defined (no fixed schema)"],
            ["format", "JSON internally, rendered dynamically"],
            ["sources", "No links or raw citations provided"],
        ]

        return (
                tabulate(
                    rows,
                    headers=["property", "value"],
                    tablefmt="psql",
                )
                + "\n\nTRACE:\n"
                + "\n".join(trace)
        )

    # -------------------------------------------------
    # llm.filtering ?
    # -------------------------------------------------
    if q == "filtering":
        rows = [
            ["mechanism", "Prompt-based semantic filtering"],
            ["determinism", "Model-dependent but temperature=0"],
            ["scope", "Applies to summaries and table generation"],
            ["guarantees", "No hard constraints, best-effort only"],
        ]

        return (
            tabulate(
                rows,
                headers=["property", "value"],
                tablefmt="psql",
            )
            + "\n\nTRACE:\n"
            + "\n".join(trace)
        )

    # -------------------------------------------------
    # llm.limitations ?
    # -------------------------------------------------
    if q == "limitations":
        rows = [
            ["schema guarantees", "Not enforced (model-defined structure)"],
            ["data browsing", "Not supported"],
            ["raw documents", "Not exposed"],
            ["ground truth verification", "Not guaranteed"],
        ]

        return (
            tabulate(
                rows,
                headers=["limitation", "description"],
                tablefmt="psql",
            )
            + "\n\nTRACE:\n"
            + "\n".join(trace)
        )

    # -------------------------------------------------
    # UNKNOWN SCOPE
    # -------------------------------------------------
    return (
        f"Unknown LLM metadata scope '{query}'.\n"
        f"Available scopes: text, table, filtering, limitations\n\n"
        "TRACE:\n"
        + "\n".join(trace)
    )


# =========================================================
#  EMAIL METADATA MODEL (INTROSPECTION ONLY)
# =========================================================

EMAIL_SCHEMA = {
    "message": {
        "description": "Single Gmail message (headers, labels, attachments)",
        "attributes": {

            "from": {
                "type": "string",
                "description": "Sender email address",
                "source": "Gmail header: From",
                "filterable": True,
            },

            "to": {
                "type": "string",
                "description": "Primary recipient(s)",
                "source": "Gmail header: To",
                "filterable": True,
            },

            "cc": {
                "type": "string",
                "description": "Carbon copy recipients",
                "source": "Gmail header: Cc",
                "filterable": True,
            },

            "bcc": {
                "type": "string",
                "description": "Blind carbon copy recipients (display-only)",
                "source": "Gmail header: Bcc",
                "filterable": False,
            },

            "reply_to": {
                "type": "string",
                "description": "Reply-To address if specified (display-only)",
                "source": "Gmail header: Reply-To",
                "filterable": False,
            },

            "subject": {
                "type": "string",
                "description": "Email subject line",
                "source": "Gmail header: Subject",
                "filterable": True,
            },

            "date": {
                "type": "datetime",
                "description": (
                    "Email timestamp used for temporal filtering. "
                ),
                "source": "Gmail API: internalDate (filtering), header Date (display)",
                "filterable": True,
                "operators": {
                    "after": {
                        "syntax": "after:YYYY/MM/DD",
                        "description": "Messages received after the given date (inclusive)",
                    },
                    "before": {
                        "syntax": "before:YYYY/MM/DD",
                        "description": "Messages received before the given date (exclusive)",
                    },
                    "newer_than": {
                        "syntax": "newer_than:Xd",
                        "description": "Messages newer than X days",
                    },
                    "older_than": {
                        "syntax": "older_than:Xd",
                        "description": "Messages older than X days",
                    },
                },
            },

            "snippet": {
                "type": "string",
                "description": "Short Gmail-provided message preview",
                "source": "Gmail API: snippet",
                "filterable": False,
            },

            "labels": {
                "type": "list[string]",
                "description": "Gmail labels applied to the message",
                "source": "Gmail API: labelIds",
                "filterable": True,
            },

            "hasAttachments": {
                "type": "boolean",
                "description": "Whether the message has attachments",
                "source": "Derived from MIME parts",
                "filterable": True,
            },

            "attachment_names": {
                "type": "list[string]",
                "description": "Names of attached files (display-only)",
                "source": "MIME payload traversal",
                "filterable": False,
            },

            "attachment_types": {
                "type": "list[string]",
                "description": "File extensions of attachments (display-only)",
                "source": "Derived from attachment filenames",
                "filterable": False,
            },

            "attachment_count": {
                "type": "int",
                "description": "Number of attachments (display-only)",
                "source": "Derived from attachment list length",
                "filterable": False,
            },

            "isForwarded": {
                "type": "boolean",
                "description": "Whether the email is a forwarded message (based on subject prefix Fwd:/FW:/WG:)",
                "source": "Derived from Subject header",
                "filterable": True,
            },
        },
    }
}


# =========================================================
#  EMAIL METADATA HELPERS
# =========================================================

def _email_global_metadata() -> str:
    attrs = EMAIL_SCHEMA["message"]["attributes"]
    rows = [
        [a, info.get("type", ""), info.get("description", ""), info.get("source", "")]
        for a, info in attrs.items()
    ]
    return tabulate(
        rows,
        headers=["attribute", "type", "description", "source"],
        tablefmt="psql",
    )


def _email_attribute_metadata(attr: str) -> str:
    attrs = EMAIL_SCHEMA["message"]["attributes"]

    # case-insensitive lookup
    attr_lookup = {k.lower(): k for k in attrs.keys()}
    canonical = attr_lookup.get(attr.lower())

    if canonical:
        info = attrs[canonical]
        rows = [
            ["attribute", canonical],
            ["type", info.get("type", "")],
            ["description", info.get("description", "")],
            ["source", info.get("source", "")],
        ]

        if "filterable" in info:
            rows.append(["filterable", str(info["filterable"])])

        operators = info.get("operators")
        if operators:
            for op, meta in operators.items():
                rows.append([
                    f"operator.{op}",
                    f"{meta.get('syntax', '')} — {meta.get('description', '')}"
                ])

        return tabulate(rows, headers=["property", "value"], tablefmt="psql")

    return (
        f"Unknown email attribute '{attr}'.\n"
        f"Try: email ?\n"
        f"Known attributes: {', '.join(sorted(attrs.keys()))}"
    )


# =========================================================
#  EMAIL METADATA
# =========================================================

def _email_metadata(query: Optional[str]) -> str:
    trace = [
        "META(EMAIL): REQUEST RECEIVED",
        "MODE: INTROSPECTION",
        "LLM: DISABLED (METADATA MODE)",
        "RAG: DISABLED",
    ]

    if not query:
        return _email_global_metadata() + "\n\nTRACE:\n" + "\n".join(trace)

    return _email_attribute_metadata(query.lower()) + "\n\nTRACE:\n" + "\n".join(trace)


# =========================================================
#  ENTRYPOINT
# =========================================================

def get_metadata_answer(
    db_path: str,
    ds: Optional[DataSource],
    query: Optional[str],
) -> str:

    if ds is None:
        return _global_metadata()

    if ds == DataSource.DB:
        return _db_metadata(db_path, query)
    if ds == DataSource.WIKIPEDIA:
        return _wikipedia_metadata(query)
    if ds == DataSource.PILE:
        return _pile_metadata(query)
    if ds == DataSource.EMAIL:
        return _email_metadata(query)

    return "Unknown metadata source."

