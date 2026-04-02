import re
from typing import List, Optional

from tabulate import tabulate
from openai import OpenAI

from enums import Datatype
from Configurations.config_loader import OPENAI_API_KEY
from Configurations.gmail_auth import get_gmail_service

from rag.rag import RAG, LowConfidenceMatchError
from rag.embedding_model import SentenceTransformerEmbeddingModel

from apis.metadata_api import EMAIL_SCHEMA
from apis.llm_quotes import extract_llm_chunks, contains_llm_marker


# =========================================================
#  LLM client
# =========================================================

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================================================
#  Gmail Fetch Limit
# =========================================================
GMAIL_FETCH_LIMIT = 50

# =========================================================
#  Embedding model
# =========================================================

EMBEDDING_MODEL = SentenceTransformerEmbeddingModel(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
#  LLM: translate WHERE → Gmail search query
# =========================================================

def llm_translate_where(chunk: str) -> str:
    prompt = f"""
Translate the following request into a valid Gmail search query.

Rules:
- Use Gmail search syntax
- Fields: from:, to:, cc:, bcc:, subject:, label:
- Relative time:
  - "last X days", "past X days", "recent", "in the last weeks" → newer_than:Xd
  - "older than X days", "before last week" → older_than:Xd
- Absolute time:
  - Use after:/before: ONLY with format YYYY/MM/DD
- Do NOT use YYYY-MM-DD
- Return ONLY the query string
- Do not include surrounding quotes
- Forwarded emails:
  - If user refers to forwarded emails, use:
    subject:(fwd OR fw OR wg)
  - Do NOT use is:forwarded


Request:
{chunk}
""".strip()

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    return (r.choices[0].message.content or "").strip()


# =========================================================
#  LLM: summarize emails (ALWAYS for return TEXT)
# =========================================================

def llm_summarize(messages: List[dict], instruction: Optional[str]) -> str:
    body = "\n\n".join(
        f"From: {m.get('from','')}\n"
        f"To: {m.get('to','')}\n"
        f"Cc: {m.get('cc','')}\n"
        f"Subject: {m.get('subject','')}\n"
        f"Date: {m.get('date','')}\n"
        f"Labels: {', '.join(m.get('labels', []) or [])}\n"
        f"Has attachments: {bool(m.get('hasAttachments', False))}\n"
        f"Attachments: {', '.join(m.get('attachment_names', []) or [])}\n"
        f"{m.get('snippet','')}"
        for m in messages
    )

    prompt = f"""
Summarize the following emails.

Instruction (if any):
{instruction or "Provide a concise summary."}

Emails:
{body}
""".strip()

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    return (r.choices[0].message.content or "").strip()


# =========================================================
#  Attribut Modus Helper
# =========================================================

def _split_email_where_operator(where_raw: str):
    """
    Splits WHERE into (op, lhs, rhs) where op is 'in' or '='.
    Operators are detected ONLY outside quotes.
    """
    def _find_outside_quotes(s: str, needle: str) -> int:
        in_single = False
        in_double = False
        n = len(needle)

        i = 0
        while i <= len(s) - n:
            ch = s[i]

            if ch == "'" and not in_double:
                in_single = not in_single
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                i += 1
                continue

            if not in_single and not in_double and s[i:i+n] == needle:
                return i

            i += 1
        return -1

    w = where_raw.strip()

    idx_in = _find_outside_quotes(w, " in ")
    idx_eq = _find_outside_quotes(w, "=")

    if idx_in != -1 and (idx_eq == -1 or idx_in < idx_eq):
        return "in", w[:idx_in].strip(), w[idx_in + 4:].strip()

    if idx_eq != -1:
        return "=", w[:idx_eq].strip(), w[idx_eq + 1:].strip()

    raise ValueError(
        "Invalid WHERE syntax.\n"
        "Use:\n"
        "  email.attr = \"value\"\n"
        "  \"value\" in email.attr"
    )


def is_email_attribute_where(where_raw: str) -> bool:
    if not where_raw or not where_raw.strip():
        return False

    try:
        op, lhs, rhs = _split_email_where_operator(where_raw)
    except ValueError:
        return False

    # email.attr = "value"
    if op == "=":
        return lhs.lower().startswith(("email.", "mail."))

    # "value" in email.attr
    if op == "in":
        return rhs.lower().startswith(("email.", "mail."))

    return False


def parse_email_attribute_where(where_raw: str, trace: Optional[List[str]] = None) -> list[str]:
    """
    Parses:

      email.from = "google"
      "google" in email.from
      mail.after = "2024/01/01"
      email.hasAttachments = true

    Rules:
    - '=' and 'in' are semantically identical
    - unknown attributes → ERROR
    - invalid syntax → ERROR
    - invalid boolean → ERROR

    Notes:
    - 'in' allowed only for string attributes
    - date filters use '=' syntax only
    """
    if trace is None:
        trace = []

    clauses = []

    # -------------------------------------------------
    # 1) Split operator FIRST (Wiki-style)
    # -------------------------------------------------
    op, lhs, rhs = _split_email_where_operator(where_raw)

    # canonical attribute lookup
    ATTR_LOOKUP = {
        k.lower(): k
        for k in EMAIL_SCHEMA["message"]["attributes"].keys()
    }

    FILTER_ONLY_ATTRS = {
        "after",
        "before",
        "newer_than",
        "older_than",
    }

    # -------------------------------------------------
    # 2) Normalize (value, attribute)
    # -------------------------------------------------
    if op == "in":
        # "value" in email.attr
        m = re.match(r"(['\"])(.*?)\1$", lhs)
        if not m:
            raise ValueError(
                "Invalid WHERE syntax.\n"
                "Missing quoted value before 'in'.\n"
                "Example:\n"
                "  \"google\" in email.from"
            )
        value = m.group(2)

        if not rhs.lower().startswith(("email.", "mail.")):
            raise ValueError(
                "Invalid WHERE target.\n"
                "Expected email.<attribute> or mail.<attribute>"
            )

        raw_attr = rhs.split(".", 1)[1].lower()

        if raw_attr in FILTER_ONLY_ATTRS:
            raise ValueError(
                "Invalid WHERE clause.\n"
                    f"Operator 'in' is not supported for filter-only attribute '{raw_attr}'.\n"
                    "Use one of:\n"
                    f"  email.{raw_attr} = \"YYYY/MM/DD\"   (absolute date)\n"
                    f"  email.{raw_attr} = \"30d\"         (relative, for newer_than/older_than style inputs)\n"
            )

    else:  # "="
        # email.attr = "value"
        if not lhs.lower().startswith(("email.", "mail.")):
            raise ValueError(
                "Invalid WHERE target.\n"
                "Expected email.<attribute> or mail.<attribute>"
            )

        raw_attr = lhs.split(".", 1)[1].lower()

        m = re.match(r"(['\"])(.*?)\1$", rhs)
        if not m:
            raise ValueError(
                "Invalid WHERE syntax.\n"
                "Missing quoted value after '='.\n"
                "Example:\n"
                "  email.from = \"google\""
            )

        value = m.group(2)

    # -------------------------------------------------
    # 3) Attribute validation
    # -------------------------------------------------
    attr = ATTR_LOOKUP.get(raw_attr)

    if attr is None and raw_attr not in FILTER_ONLY_ATTRS:
        raise ValueError(
            "Unknown email attribute.\n"
                f"Attribute: '{raw_attr}'\n"
                "Use: email ? for available attributes.\n"
                "Example:\n"
                "  where email.from = \"google\""
        )

    # -------------------------------------------------
    # 4) Mapping to Gmail query
    # -------------------------------------------------
    if attr in ("from", "to", "cc", "subject"):
        clauses.append(f"{attr}:{value}")

    elif attr in ("label", "labels"):
        clauses.append(f"label:{value}")

    elif raw_attr in FILTER_ONLY_ATTRS:
        clauses.append(f"{raw_attr}:{value}")

    elif attr == "hasAttachments":
        val = value.lower()
        if val in ("true", "1", "yes"):
            clauses.append("has:attachment")
        elif val in ("false", "0", "no"):
            clauses.append("-has:attachment")
        else:
            raise ValueError(
                "Invalid boolean value for email.hasAttachments.\n"
                "Use: true or false."
            )

    elif attr == "isForwarded":
        val = value.lower()
        if val in ("true", "1", "yes"):
            clauses.append("subject:(fwd OR fw OR wg)")
        elif val in ("false", "0", "no"):
            clauses.append("-subject:(fwd OR fw OR wg)")
        else:
            raise ValueError(
                "Invalid boolean value for email.isForwarded.\n"
                "Use: true or false."
            )

    return clauses

def is_email_attribute_find(find_raw: str) -> bool:
    if not find_raw:
        return False

    fl = find_raw.lower().strip()
    return fl.startswith("email.") or fl.startswith("mail.")


def parse_email_find_attributes(find_raw: str) -> List[str]:
    """
    Parses:
      email.subject
      email.subject, email.cc, mail.from

    Returns:
      ["subject", "cc", "from"]
    """
    if not find_raw:
        return []

    parts = [p.strip() for p in find_raw.split(",")]
    attrs = []

    MESSAGE_ATTRS = {
        k.lower()
        for k in EMAIL_SCHEMA["message"]["attributes"].keys()
    }

    FILTER_ONLY_ATTRS = {
        "after",
        "before",
        "newer_than",
        "older_than",
    }

    for p in parts:
        pl = p.lower()
        if not (pl.startswith("email.") or pl.startswith("mail.")):
            raise ValueError(
                "Invalid FIND attribute.\n"
                f"Expected email.<attribute>, got '{p}'."
            )

        attr = pl.split(".", 1)[1]

        # filter-only attributes are NOT allowed in FIND
        if attr in FILTER_ONLY_ATTRS:
            raise ValueError(
                "Invalid FIND attribute.\n\n"
                f"'{attr}' is a filter-only attribute and can only be used in WHERE clauses.\n\n"
                "Example:\n"
                f"  find email.subject from email where email.{attr} = \"30d\""
            )

        if attr not in MESSAGE_ATTRS:
            raise ValueError(
                "Unknown email attribute in FIND clause.\n"
                f"Attribute: '{attr}'\n"
                "Use: email ? for available attributes.\n"
                "Example:\n"
                "  find email.subject, email.cc from email"
            )

        attrs.append(attr)

    return attrs


# =========================================================
#  Gmail query helpers
# =========================================================

def normalize_gmail_query(q: Optional[str]) -> Optional[str]:
    if not q:
        return q

    q = q.replace('"', '')

    # Fix wrong date format (YYYY-MM-DD → YYYY/MM/DD)
    q = re.sub(
        r'(after|before):(\d{4})-(\d{2})-(\d{2})',
        r'\1:\2/\3/\4',
        q
    )

    return q


def has_semantic_tokens(q: Optional[str]) -> bool:
    """
    True if query contains any meaningful tokens (field filters or free text).
    """
    if not q:
        return False
    for p in q.split():
        if ":" in p:
            _, v = p.split(":", 1)
            if v.strip():
                return True
        else:
            if p.strip():
                return True
    return False


def relax_gmail_query(q: Optional[str]) -> Optional[str]:
    """
    LAST-resort fallback: keep only field filters.
    """
    if not q:
        return None
    parts = q.split()
    fields = [p for p in parts if ":" in p]
    if fields and len(fields) < len(parts):
        return " ".join(fields)
    return None


def list_message_ids(service, q: Optional[str], limit: int, trace: List[str]) -> List[str]:
    trace.append(f"GMAIL: LIST IDS (limit={limit}, q={q})")
    resp = service.users().messages().list(
        userId="me",
        maxResults=limit,
        q=q
    ).execute()

    ids = [m["id"] for m in (resp.get("messages") or [])]

    trace.append(f"GMAIL: LIST IDS returned {len(ids)} messages")

    return ids



# =========================================================
#  Gmail message parsing + attachments
# =========================================================

def _get_header(payload: dict, name: str) -> str:
    for h in payload.get("headers", []) or []:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def collect_attachments(payload: dict, files=None) -> List[str]:
    if files is None:
        files = []

    filename = payload.get("filename")
    body = payload.get("body", {}) or {}

    if filename and body.get("attachmentId"):
        files.append(filename)

    for part in payload.get("parts", []) or []:
        collect_attachments(part, files)

    return files


def get_message(service, msg_id: str) -> dict:
    msg = service.users().messages().get(
        userId="me",
        id=msg_id,
        format="full",
    ).execute()

    payload = msg.get("payload", {}) or {}

    attachment_names = collect_attachments(payload)
    attachment_types = [
        f.split(".")[-1].lower()
        for f in attachment_names if "." in f
    ]

    subject = _get_header(payload, "Subject")

    return {
        "id": msg_id,

        "from": _get_header(payload, "From"),
        "to": _get_header(payload, "To"),
        "cc": _get_header(payload, "Cc"),
        "bcc": _get_header(payload, "Bcc"),
        "reply_to": _get_header(payload, "Reply-To"),

        "subject": subject,
        "date": _get_header(payload, "Date"),

        "snippet": msg.get("snippet", "") or "",

        "hasAttachments": bool(attachment_names),
        "attachment_names": attachment_names,
        "attachment_types": attachment_types,
        "attachment_count": len(attachment_names),

        "labels": msg.get("labelIds", []) or [],
        "isForwarded": bool(re.search(r'\b(fwd|fw|wg):', subject.lower())),
    }


# =========================================================
#  RAG helpers (WHERE) - matching spaces
# =========================================================

def rag_best_match(
    query: str,
    candidates: List[str],
    candidate_embeddings_cache: dict,
    field: str
) -> Optional[str]:

    if not candidates:
        return None

    vecs = candidate_embeddings_cache.get(field)

    if vecs is None:
        vecs = [EMBEDDING_MODEL.embed(c) for c in candidates]
        candidate_embeddings_cache[field] = vecs

    try:
        idx = RAG(
            query=query,
            emb_column=vecs,
            k=1,
            emb_model=EMBEDDING_MODEL
        )[0]
    except LowConfidenceMatchError:
        return None

    return candidates[idx]


def collect_gmail_rag_candidates(messages: List[dict]) -> dict:
    """
    Candidates for repairing Gmail WHERE values.
    Includes an 'any' pool for repairing free tokens (e.g. compounds).
    """
    cand = {
        "from": set(),
        "to": set(),
        "cc": set(),
        "subject": set(),
        "label": set(),
        "any": set(),
    }

    for m in messages:
        for f in ("from", "to", "cc", "subject"):
            toks = re.findall(r"[a-zA-Z]{3,}", (m.get(f) or "").lower())
            cand[f].update(toks)
            cand["any"].update(toks)

        for lbl in m.get("labels", []):
            lbl_l = (lbl or "").lower()
            if lbl_l:
                cand["label"].add(lbl_l)
                cand["any"].add(lbl_l)

    return {k: list(v) for k, v in cand.items()}


def collect_header_texts(messages: List[dict]) -> dict:
    """
    For header-aware inference of where free tokens belong.
    """
    headers = {
        "subject": [],
        "from": [],
        "to": [],
        "cc": [],
        "label": [],
    }

    for m in messages:
        for f in ("subject", "from", "to", "cc"):
            if m.get(f):
                headers[f].append(m[f].lower())
        for lbl in m.get("labels", []):
            headers["label"].append(lbl.lower())

    return headers


def infer_field_for_free_tokens(phrase: str, header_texts: dict,
    header_embeddings: dict) -> Optional[str]:
    """
    Decide which field best matches free tokens by comparing to real inbox headers.
    Returns one of: subject/from/to/cc/label or None if nothing matches confidently.
    """
    best_field = None
    best_score = -1

    for field, texts in header_texts.items():
        if not texts:
            continue

        vecs = header_embeddings.get(field)

        if vecs is None:
            texts = header_texts.get(field, [])
            if not texts:
                continue
            vecs = [EMBEDDING_MODEL.embed(t) for t in texts]
            header_embeddings[field] = vecs  # cache it

        if not vecs:
            continue

        try:
            RAG(query=phrase, emb_column=vecs, k=1, emb_model=EMBEDDING_MODEL)
            score = 1
        except LowConfidenceMatchError:
            score = 0

        if score > best_score:
            best_score = score
            best_field = field

    return best_field if best_score > 0 else None


def group_field_phrases(parts: List[str]) -> List[str]:
    """
    Groups field:value + following free tokens into a single field phrase.

    Example:
      ["cc:john", "doe", "project"]
      -> ["cc:john doe project"]
    """
    grouped = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if ":" in p:
            f, v = p.split(":", 1)
            phrase = [v]
            j = i + 1
            while j < len(parts) and ":" not in parts[j]:
                phrase.append(parts[j])
                j += 1
            grouped.append(f"{f}:{' '.join(phrase)}")
            i = j
        else:
            grouped.append(p)
            i += 1
    return grouped


def _compact_free_phrase(phrase: str) -> str:
    return re.sub(r"\s+", "", (phrase or "").strip().lower())


def has_field_constraints(q: Optional[str]) -> bool:
    """
    True if query contains at least one field:value constraint with a non-empty value.
    """
    if not q:
        return False
    for p in q.split():
        if ":" in p:
            _, v = p.split(":", 1)
            if v.strip():
                return True
    return False

def cheap_string_repair(token: str, candidates: List[str]) -> Optional[str]:
    """
    Lightweight fallback for very short tokens (e.g. 'googl').
    Uses simple prefix matching.
    """
    t = token.lower()
    for c in candidates:
        if c.startswith(t) or t.startswith(c):
            return c
    return None


# =========================================================
#  NEW: RAG STRUCTURING BEFORE FIRST GMAIL SEARCH
# =========================================================

# PRE-RAG:
# Attempts to structure free-text WHERE clauses into Gmail fields
# only if a confident, inbox-aware semantic mapping is possible.
def rag_structure_where_pre_query(query: str, header_texts: dict,
    header_embeddings: dict, trace: List[str]) -> Optional[str]:
    """
    Convert free tokens into an inferred field BEFORE first Gmail search.

    Example:
      "from:google sicherheits warnung"
      -> "from:google subject:sicherheits warnung"

    If nothing inferred, returns None (no change).
    """
    q0 = (query or "").strip()
    trace.append(
        f"PRE-RAG: analyzing query='{q0}' for possible structuring"
    )
    if not q0:
        return None

    parts = group_field_phrases(q0.split())
    # keep already fielded tokens as-is
    field_parts = [p for p in parts if ":" in p]
    free_parts = [p for p in parts if ":" not in p]
    trace.append(
        f"PRE-RAG: detected field parts={field_parts}, free parts={free_parts}"
    )

    if not free_parts:
        return None

    phrase = " ".join(free_parts).strip()
    trace.append(
        f"PRE-RAG: attempting field inference for phrase='{phrase}'"
    )
    if not phrase:
        return None

    inferred = infer_field_for_free_tokens(phrase, header_texts, header_embeddings)

    if inferred:
        trace.append(
            f"PRE-RAG: inferred field='{inferred}' for phrase='{phrase}' (high confidence)"
        )
        new_q = " ".join(field_parts + [f"{inferred}:{phrase}"]).strip()
        return new_q if new_q != q0 else None
    else:
        trace.append(
            f"PRE-RAG: no confident field inference for phrase='{phrase}', "
            f"leaving query unstructured"
        )

    return None


# =========================================================
#  RAG: repair existing field values + free tokens (multi-iter)
# =========================================================

def rag_repair_gmail_query(query: str, candidate_spaces: dict,
    candidate_embeddings: dict,
    header_texts: dict,
    header_embeddings: dict, trace: List[str], max_iters: int = 4) -> Optional[str]:
    """
    Multi-step RAG repair (multi-iter):
      A) infer field for free tokens (header-aware) if confident
      B) if (A) can't infer, still try to COMPACT/REPAIR free tokens (compound fix)
      C) repair existing field values (from:googl -> from:google etc.)
      repeats until stable or max_iters

    Returns repaired query string or None if nothing changed.
    """
    q0 = (query or "").strip()
    if not q0:
        return None

    cand = candidate_spaces

    current = q0
    changed_any = False

    for _ in range(max_iters):
        parts = group_field_phrases(current.split())

        free_tokens = [p for p in parts if ":" not in p]
        field_parts = [p for p in parts if ":" in p]

        iter_changed = False

        # --- Step A: infer field for free tokens (header-aware) ---
        if free_tokens:
            phrase = " ".join(free_tokens).strip()
            if phrase:
                inferred = infer_field_for_free_tokens(phrase, header_texts, header_embeddings)

                if inferred:
                    trace.append(
                        f"RAG(WHERE): inferred field '{inferred}' for free tokens '{phrase}'"
                    )
                    field_parts.append(f"{inferred}:{phrase}")
                    parts = field_parts
                    iter_changed = True
                else:
                    # --- Step B: no field inference; repair free tokens as FREE TEXT (compound/typo) ---
                    compact = _compact_free_phrase(phrase)
                    if cand.get("any"):
                        best_free = rag_best_match(
                            compact,
                            cand["any"],
                            candidate_embeddings,
                            "any"
                        )
                    else:
                        best_free = None
                    if best_free and best_free != compact:
                        trace.append(f"RAG(WHERE): free tokens '{phrase}' -> '{best_free}' (free text)")
                        parts = field_parts + [best_free]
                        iter_changed = True
                    else:
                        repaired_free: List[str] = []
                        any_changed = False
                        for t in phrase.split():
                            if cand.get("any"):
                                bt = rag_best_match(
                                    t.lower(),
                                    cand["any"],
                                    candidate_embeddings,
                                    "any"
                                )

                            else:
                                bt = None
                            if bt and bt != t.lower():
                                trace.append(f"RAG(WHERE): token '{t}' -> '{bt}' (free text)")
                                repaired_free.append(bt)
                                any_changed = True
                            else:
                                repaired_free.append(t)
                        if any_changed:
                            parts = field_parts + repaired_free
                            iter_changed = True

        # --- Step C: repair existing field values ---
        repaired_parts: List[str] = []
        for part in parts:
            if ":" not in part:
                repaired_parts.append(part)
                continue

            field, value = part.split(":", 1)
            field_l = field.lower()
            value_l = value.lower().strip()

            space = cand.get(field_l)
            if not space or not value_l:
                repaired_parts.append(part)
                continue

            best = rag_best_match(
                value_l,
                space,
                candidate_embeddings,
                field_l
            )

            # Fallback for very short tokens (e.g. 'googl')
            if not best and len(value_l) <= 5:
                cheap = cheap_string_repair(value_l, space)
                if cheap:
                    trace.append(
                        f"RAG(WHERE): short-token repair '{value_l}' -> '{cheap}' (prefix heuristic)"
                    )
                    best = cheap

            if best and best != value_l:
                trace.append(
                    f"RAG(WHERE): field '{field_l}' value '{value_l}' "
                    f"→ '{best}' (high-confidence semantic repair)"
                )
                repaired_parts.append(f"{field}:{best}")
                iter_changed = True
            else:
                repaired_parts.append(part)

        # dedupe while preserving order
        new_q = " ".join(dict.fromkeys(repaired_parts)).strip()

        if new_q == current or not iter_changed:
            break

        current = new_q
        changed_any = True

    return current if changed_any and current != q0 else None


# =========================================================
#  RAG helpers (TEXT)
# =========================================================

def filter_with_embeddings(messages, embeddings, keep_messages):
    keep_ids = {m["id"] for m in keep_messages}

    new_msgs = []
    new_embs = [] if embeddings is not None else None

    if embeddings is None:
        # No embeddings yet → just filter messages
        return [m for m in messages if m["id"] in keep_ids], None

    for m, e in zip(messages, embeddings):
        if m["id"] in keep_ids:
            new_msgs.append(m)
            new_embs.append(e)

    return new_msgs, new_embs


def ensure_message_embeddings(messages, message_embeddings):
    """
    Lazy embedding creation.
    Only create embeddings if not already created.
    """
    if message_embeddings is not None:
        return message_embeddings

    message_docs = [
        (
            f"From: {m.get('from', '')}\n"
            f"To: {m.get('to', '')}\n"
            f"Cc: {m.get('cc', '')}\n"
            f"Bcc: {m.get('bcc', '')}\n"
            f"Subject: {m.get('subject', '')}\n"
            f"Labels: {', '.join(m.get('labels', []) or [])}\n"
            f"Has attachments: {bool(m.get('hasAttachments', False))}\n"
            f"Attachments: {', '.join(m.get('attachment_names', []) or [])}\n"
            f"{m.get('snippet', '')}"
        )
        for m in messages
    ]

    return [EMBEDDING_MODEL.embed(d) for d in message_docs]


def rag_filter_emails(messages: List[dict], message_embeddings: List[List[float]], query: str, trace: List[str], k: int = 1) -> List[dict]:

    vectors = message_embeddings

    try:
        idxs = RAG(
            query=query,
            emb_column=vectors,
            k=min(k, len(vectors)),
            emb_model=EMBEDDING_MODEL,
        )
    except LowConfidenceMatchError:
        trace.append("RAG(TEXT): no confident match")
        return []

    trace.append(f"RAG(TEXT): selected indices {idxs}")
    return [messages[i] for i in idxs]


# =========================================================
#  FIND helpers
# =========================================================

def keyword_filter(messages: List[dict], query: str) -> List[dict]:
    q = query.lower()
    out = []
    for m in messages:
        text = " ".join([
            m.get("from", ""),
            m.get("to", ""),
            m.get("cc", ""),
            m.get("bcc", ""),
            m.get("subject", ""),
            m.get("snippet", ""),
            " ".join(m.get("labels", []) or []),
            " ".join(m.get("attachment_names", []) or []),
            " ".join(m.get("attachment_types", []) or []),
        ]).lower()
        if q in text:
            out.append(m)
    return out


def normalize_compound(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def clean_find_query(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    # Remove ALL leading/trailing quotes (` ' ")
    text = re.sub(r'^[\'"`]+', '', text)
    text = re.sub(r'[\'"`]+$', '', text)

    return text.strip()


# =========================================================
#  Operators (AND, OR, NOT) helpers
# =========================================================

def negate_gmail_query(q: str) -> str:
    """
    Gmail has no grouping. To negate a conjunction, negate each token.
    Example: "from:google subject:warnung" -> "-from:google -subject:warnung"
    """
    parts = group_field_phrases((q or "").split())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return ""
    neg = []
    for p in parts:
        if p.startswith("-"):
            neg.append(p[1:])  # double negation: -(-x) -> x
        else:
            neg.append("-" + p)
    return " ".join(neg)


def _tokenize_where(s: str):
    tokens = []
    buf = []
    in_single = False
    in_double = False
    i = 0

    def flush():
        part = "".join(buf).strip()
        if part:
            tokens.append(part)
        buf.clear()

    while i < len(s):
        ch = s[i]

        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch); i += 1; continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch); i += 1; continue

        if not in_single and not in_double:
            if ch == "(":
                flush(); tokens.append("("); i += 1; continue
            if ch == ")":
                flush(); tokens.append(")"); i += 1; continue

            for op in ("not", "and", "or"):
                if (
                    s[i:i+len(op)].lower() == op
                    and (i == 0 or s[i-1].isspace())
                    and (i+len(op) == len(s) or s[i+len(op)].isspace())
                ):
                    flush()
                    tokens.append(op.upper())
                    i += len(op)
                    break
            else:
                buf.append(ch); i += 1
            continue

        buf.append(ch); i += 1

    flush()
    return tokens

class _WhereParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _eat(self, expected=None):
        tok = self._peek()
        if expected and tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}'")
        self.pos += 1
        return tok

    def parse_expr(self):
        node = self.parse_and()
        while self._peek() == "OR":
            self._eat("OR")
            node = ("OR", node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_not()
        while self._peek() == "AND":
            self._eat("AND")
            node = ("AND", node, self.parse_not())
        return node

    def parse_not(self):
        if self._peek() == "NOT":
            self._eat("NOT")
            return ("NOT", self.parse_not())
        return self.parse_atom()

    def parse_atom(self):
        tok = self._peek()
        if tok == "(":
            self._eat("(")
            node = self.parse_expr()
            if self._peek() != ")":
                raise ValueError("Unclosed '(' in WHERE clause")
            self._eat(")")
            return node

        if tok in ("AND", "OR", "NOT", ")"):
            raise ValueError(f"Unexpected token '{tok}'")

        self._eat()
        return ("COND", tok)

def is_raw_gmail_filter(cond: str) -> bool:
    """
    True if cond is already valid Gmail syntax and must NOT create a post-filter.
    """
    return bool(re.match(
        r'^(from|to|cc|bcc|subject|label|has|after|before|newer_than|older_than):',
        cond.strip().lower()
    ))

def is_pure_gmail_node(node):
    return (
        node.get("gmail") is not None
        and node.get("post") is None
        and not node.get("semantic", False)
    )

def is_pure_gmail_query(q: Optional[str]) -> bool:
    if not q:
        return False
    return all(
        ":" in part and part.split(":", 1)[1].strip()
        for part in q.replace("(", "").replace(")", "").split()
        if part.upper() not in ("AND", "OR", "NOT")
    )


def eval_email_condition(cond: str, trace: list[str]) -> dict:
    # -------------------------------------------------
    # DSL SYNTAX VALIDATION (before anything else)
    # -------------------------------------------------

    if cond.lower().strip().startswith(("email.", "mail.")):
        try:
            _split_email_where_operator(cond)  # validiert: nur '=' oder ' in ' außerhalb quotes
        except ValueError:
            raise ValueError(
                "Invalid operator in WHERE clause.\n"
                "Use '=' or 'in' for email attributes.\n\n"
                "Example:\n"
                '  email.from = "google"\n'
                '  "google" in email.from'
            )

    if cond.lower().strip().startswith(("email.", "mail.")):
        op, lhs, rhs = _split_email_where_operator(cond)

        # User wrote: email.attr in "value"  (wrong direction)
        if op == "in" and lhs.lower().startswith(("email.", "mail.")):
            raise ValueError(
                "Invalid WHERE syntax.\n"
                "For 'in', the VALUE must be on the left side.\n\n"
                "Use:\n"
                '  "google" in email.from\n'
                "or:\n"
                '  email.from = "google"'
            )

    # ATTR
    if is_email_attribute_where(cond):
        parts = parse_email_attribute_where(cond, trace)
        return {"gmail": " ".join(parts), "post": None, "semantic": False}

    # LLM
    if contains_llm_marker(cond):
        trace.append(f"LLM(WHERE): translating semantic clause '{cond}'")
        chunk = extract_llm_chunks(cond)[0]
        q = normalize_gmail_query(llm_translate_where(chunk))
        trace.append(f"LLM(WHERE): → {q}")
        return {"gmail": q, "post": None, "semantic": True}

    # RAW Gmail filter (has:, after:, before:, from:, ...)
    if is_raw_gmail_filter(cond):
        return {"gmail": cond, "post": None, "semantic": False}

    # RAW / freier Text
    return {
        "gmail": cond,
        "post": lambda m: cond.lower() in (
            (m.get("subject", "") + " " + m.get("snippet", "")).lower()
        ),
        "semantic": False
    }

# =========================================================
#  LOGIC NORMALIZATION (NNF)
# =========================================================

def push_not_down(node):
    kind = node[0]

    if kind == "NOT":
        inner = node[1]

        if inner[0] == "NOT":
            # NOT NOT A -> A
            return push_not_down(inner[1])

        if inner[0] == "AND":
            # NOT (A AND B) -> (NOT A) OR (NOT B)
            return (
                "OR",
                push_not_down(("NOT", inner[1])),
                push_not_down(("NOT", inner[2])),
            )

        if inner[0] == "OR":
            # NOT (A OR B) -> (NOT A) AND (NOT B)
            return (
                "AND",
                push_not_down(("NOT", inner[1])),
                push_not_down(("NOT", inner[2])),
            )

        return ("NOT", push_not_down(inner))

    if kind in ("AND", "OR"):
        return (
            kind,
            push_not_down(node[1]),
            push_not_down(node[2]),
        )

    return node

# =========================================================
#  DNF TRANSFORMATION
# =========================================================

def to_dnf(node):
    kind = node[0]

    if kind == "AND":
        left = to_dnf(node[1])
        right = to_dnf(node[2])

        # Distribute AND over OR
        if left[0] == "OR":
            return (
                "OR",
                to_dnf(("AND", left[1], right)),
                to_dnf(("AND", left[2], right)),
            )

        if right[0] == "OR":
            return (
                "OR",
                to_dnf(("AND", left, right[1])),
                to_dnf(("AND", left, right[2])),
            )

        return ("AND", left, right)

    if kind == "OR":
        return ("OR", to_dnf(node[1]), to_dnf(node[2]))

    if kind == "NOT":
        return node  # already pushed down

    return node

def extract_dnf_clauses(node):
    if node[0] == "OR":
        return extract_dnf_clauses(node[1]) + extract_dnf_clauses(node[2])

    return [node]


def build_email_where(tree, trace):
    kind = tree[0]

    if kind == "COND":
        result = eval_email_condition(tree[1], trace)
        result["contains_not"] = False
        # semantic ist schon drin
        return result

    if kind == "NOT":
        inner = build_email_where(tree[1], trace)

        # Wenn inner rein Gmail → direkt delegieren
        if is_pure_gmail_node(inner):
            return {
                "gmail": negate_gmail_query(inner["gmail"]),
                "post": None,
                "contains_not": True,
                "semantic": inner.get("semantic", False)
            }

        # Sonst KEIN Gmail-Filter
        return {
            "gmail": None,
            "post": (
                lambda m, f=inner["post"]:
                not f(m) if f else False
            ),
            "contains_not": True,
            "semantic": inner.get("semantic", False)
        }

    if kind == "AND":
        a = build_email_where(tree[1], trace)
        b = build_email_where(tree[2], trace)

        if is_pure_gmail_node(a) and is_pure_gmail_node(b):
            return {
                "gmail": f"{a['gmail']} {b['gmail']}",
                "post": None,
                "contains_not": a["contains_not"] or b["contains_not"],
                "semantic": a.get("semantic", False) or b.get("semantic", False)
            }

        return {
            "gmail": " ".join(filter(None, [a["gmail"], b["gmail"]])),
            "post": lambda m: (
                    (a["post"](m) if a["post"] else True)
                    and (b["post"](m) if b["post"] else True)
            ),
            "contains_not": a["contains_not"] or b["contains_not"],
            "semantic": a.get("semantic", False) or b.get("semantic", False)
        }

    if kind == "OR":
        raise RuntimeError(
            "OR should not appear after DNF transformation. "
            "Check DNF logic."
        )

    raise RuntimeError("Invalid WHERE AST")


# =========================================================
#  MAIN ENTRYPOINT
# =========================================================

def get_email_answer(parsed: dict):
    trace: List[str] = ["EMAIL: REQUEST RECEIVED"]

    rtype = parsed.get("rtype") or Datatype.TABLE
    where_raw = (parsed.get("where_raw") or "").strip()
    find_raw = (parsed.get("find_raw") or "").strip()

    service = get_gmail_service(trace)

    query: Optional[str] = None
    where_chunk: Optional[str] = None

    # =========================================================
    #  WHERE (Attr/ LLM / RAW)
    # =========================================================

    built = None
    post_filter = None
    or_mode = False
    ids = []
    message_embeddings = None

    header_texts = {}
    candidate_spaces = {}
    header_embeddings = {}
    candidate_embeddings = {}

    if where_raw:
        try:
            tokens = _tokenize_where(where_raw)
            parser = _WhereParser(tokens)
            tree = parser.parse_expr()

            tree_raw = tree

            # 1) push NOT down
            tree = push_not_down(tree)

            # 2) transform to DNF
            tree = to_dnf(tree)

            # 3) extract OR-clauses
            dnf_clauses = extract_dnf_clauses(tree)

            MAX_DNF_BRANCHES = 8

            if len(dnf_clauses) > MAX_DNF_BRANCHES:
                raise ValueError(
                    "Query too complex.\n"
                    "Too many OR combinations after normalization.\n"
                    f"Maximum allowed: {MAX_DNF_BRANCHES}."
                )

            if parser.pos != len(tokens):
                raise ValueError(
                    "Syntax error in WHERE clause.\n"
                    "Missing logical operator (AND/OR) between conditions."
                )

            # -------------------------------------------------
            # STRICT DSL VALIDATION (after parsing)
            # -------------------------------------------------

            if tree_raw[0] == "COND":
                cond_text = tree_raw[1].lower()

                # count DSL attributes
                dsl_count = len(re.findall(r'\b(email|mail)\.', cond_text))

                # detect gmail fields
                gmail_fields = re.findall(
                    r'\b(from|to|cc|bcc|subject|label|has|after|before|newer_than|older_than):',
                    cond_text
                )

                gmail_count = len(gmail_fields)

                # Case 1: multiple DSL attributes without AND/OR → error
                if dsl_count > 1:
                    raise ValueError(
                        "Syntax error in WHERE clause.\n"
                        "Missing logical operator (AND/OR) between DSL conditions.\n\n"
                        "Example:\n"
                        '  where email.from = "google" AND email.subject = "security"'
                    )

                # Case 2: DSL + Gmail mixed without AND/OR → error
                if dsl_count > 0 and gmail_count > 0:
                    raise ValueError(
                        "Syntax error in WHERE clause.\n"
                        "Missing logical operator (AND/OR) between conditions.\n\n"
                        "Example:\n"
                        '  where from:google AND email.subject = "security"'
                    )

            # =========================================================
            #  INBOX CONTEXT (for Pre-RAG / Repair)
            # =========================================================
            trace.append(
                "CONTEXT: fetching recent inbox messages for semantic inference (Pre-RAG / Repair)"
            )
            ctx_ids = list_message_ids(service, None, GMAIL_FETCH_LIMIT, trace)
            ctx_msgs = [get_message(service, i) for i in ctx_ids]

            # =========================================================
            #  EMBEDDING CACHE (per request)
            # =========================================================

            header_texts = collect_header_texts(ctx_msgs)
            candidate_spaces = collect_gmail_rag_candidates(ctx_msgs)

            header_embeddings = {}

            candidate_embeddings = {}


            # =========================================================
            #  Execute DNF clauses (OR-split only if needed)
            # =========================================================

            if len(dnf_clauses) == 1:
                # Single clause: behave like normal query (NO pre-execution here)
                built_single = build_email_where(dnf_clauses[0], trace)
                query = built_single.get("gmail")
                post_filter = built_single.get("post")
                built = built_single  # IMPORTANT: for contains_not later
                or_mode = False
                trace.append(f"EMAIL: WHERE (single) → gmail={query}")

            else:
                # Multi clause: OR split execution (DNF)
                trace.append("EMAIL: OR handled via DNF split strategy")

                all_ids = set()
                contains_not_any = False

                for clause in dnf_clauses:
                    built_clause = build_email_where(clause, trace)
                    contains_not_any = contains_not_any or bool(
                        built_clause.get("contains_not")
                    )
                    q_branch = built_clause.get("gmail")
                    post_branch = built_clause.get("post")

                    trace.append(f"GMAIL: DNF branch search -> {q_branch}")
                    ids_branch = list_message_ids(service, q_branch, GMAIL_FETCH_LIMIT, trace)

                    # -------------------------------
                    # SAFE BRANCH REPAIR (recommended)
                    # -------------------------------
                    if (not ids_branch) and q_branch and has_field_constraints(q_branch):
                        trace.append("RAG(WHERE/BRANCH): 0 results, attempting safe semantic repair")

                        repaired_branch = rag_repair_gmail_query(
                            q_branch,
                            candidate_spaces,
                            candidate_embeddings,
                            header_texts,
                            header_embeddings,
                            trace,
                            max_iters=2
                        )

                        # Only accept if repair yields results (hard validation)
                        if repaired_branch and repaired_branch != q_branch:
                            trace.append(f"GMAIL: BRANCH RETRY (repaired) -> {repaired_branch}")
                            ids_retry = list_message_ids(service, repaired_branch, GMAIL_FETCH_LIMIT, trace)

                            if ids_retry:
                                trace.append("RAG(WHERE/BRANCH): accepted repair (validated by results)")
                                q_branch = repaired_branch
                                ids_branch = ids_retry
                            else:
                                trace.append("RAG(WHERE/BRANCH): rejected repair (still 0 results)")

                    msgs_branch = [get_message(service, mid) for mid in ids_branch]

                    if post_branch:
                        msgs_branch = [m for m in msgs_branch if post_branch(m)]

                    all_ids.update([m["id"] for m in msgs_branch])

                ids = list(all_ids)

                # In OR mode we've already executed; don't run strict search again
                query = None
                post_filter = None
                built = {"contains_not": contains_not_any}  # minimal info for later checks
                or_mode = True


        except ValueError as e:
            return str(e) + "\n\nTRACE:\n" + "\n".join(trace)


    # =========================================================
    #  PRE-RAG: structure free tokens BEFORE first search
    # =========================================================
    if not or_mode and query and has_semantic_tokens(query):
        trace.append(
            f"PRE-RAG: evaluating semantic structuring for query='{query}'"
        )
        structured = rag_structure_where_pre_query(
            query or "", header_texts, header_embeddings, trace
        )
        if structured:
            trace.append(f"GMAIL: STRUCTURED WHERE -> {structured}")
            query = structured

    # =========================================================
    #  STRICT GMAIL SEARCH
    # =========================================================

    if not or_mode:
        trace.append(f"SEARCH: executing user query against Gmail (q={query})")
        ids = list_message_ids(service, query, GMAIL_FETCH_LIMIT, trace)

    # =========================================================
    #  RAG REPAIR (ONLY if 0 results AND field constraints exist)
    # =========================================================
    if not or_mode and not ids and has_field_constraints(query):
        trace.append(
            "RAG(WHERE): no results for structured query, attempting semantic repair"
        )

        repaired = rag_repair_gmail_query(
            query or "",
            candidate_spaces,
            candidate_embeddings,
            header_texts,
            header_embeddings,
            trace,
            max_iters=4
        )

        if repaired:
            trace.append(f"GMAIL: RETRY (RAG repaired WHERE) -> {repaired}")
            query = repaired
            ids = list_message_ids(service, query, GMAIL_FETCH_LIMIT, trace)
        else:
            trace.append("RAG(WHERE): no confident repair applicable")

    # =========================================================
    #  RELAX = LAST FALLBACK (on CURRENT query state)
    # =========================================================
    if not or_mode and not ids:
        trace.append(
            f"RELAX: no results after repair for q='{query}', attempting relaxation"
        )
        relaxed = relax_gmail_query(query)
        if relaxed:
            trace.append(f"GMAIL: RELAX WHERE -> {relaxed}")
            query = relaxed
            ids = list_message_ids(service, query, GMAIL_FETCH_LIMIT, trace)
        else:
            trace.append("RELAX: not applicable (no removable free tokens)")

    # =========================================================
    #  DROP POST-FILTER IF GMAIL QUERY IS NOW PURE
    # =========================================================
    if (
            post_filter
            and query
            and is_pure_gmail_query(query)
            and not (built and built.get("contains_not", False))
    ):
        trace.append(
            "EMAIL: post-filter discarded (pure Gmail query without NOT)"
        )
        post_filter = None

    # =========================================================
    #  FETCH MESSAGES
    # =========================================================
    messages = []
    for i, mid in enumerate(ids, 1):
        trace.append(f"GMAIL: GET MESSAGE {i}/{len(ids)}")
        messages.append(get_message(service, mid))

    trace.append(f"EMAIL: FETCHED {len(messages)} MESSAGES")

    if not messages:
        return "No mails found.\n\nTRACE:\n" + "\n".join(trace)

    if post_filter:
        before = len(messages)
        messages = [m for m in messages if post_filter(m)]
        after = len(messages)

        if after < before:
            trace.append(
                f"EMAIL: applied logical post-filter ({before} → {after})"
            )


    # =========================================================
    #  FIND: EMAIL ATTRIBUTE OUTPUT (MULTI)
    # =========================================================
    try:
        is_attr_find = bool(find_raw) and is_email_attribute_find(find_raw)
    except ValueError as e:
        return str(e) + "\n\nTRACE:\n" + "\n".join(trace)

    if is_attr_find:
        try:
            attrs = parse_email_find_attributes(find_raw)
        except ValueError as e:
            return str(e) + "\n\nTRACE:\n" + "\n".join(trace)

        # --- mapping FIND attr -> message dict key ---
        ATTR_TO_MESSAGE_KEY = {
            "hasattachments": "hasAttachments",
            "isforwarded": "isForwarded",
        }

        # --- boolean attributes (must be rendered explicitly) ---
        BOOLEAN_ATTRS = {
            "hasattachments",
            "isforwarded",
        }

        rows = []
        for m in messages:
            row = []
            for a in attrs:
                key = ATTR_TO_MESSAGE_KEY.get(a, a)
                val = m.get(key)

                if a in BOOLEAN_ATTRS:
                    row.append("true" if val else "false")
                else:
                    row.append(val)

            rows.append(row)

        trace.append(f"FIND(ATTR): returning attributes {attrs}")

        return (
                tabulate(
                    rows,
                    headers=attrs,
                    tablefmt="psql"
                )
                + "\n\nTRACE:\n"
                + "\n".join(trace)
        )

    # =========================================================
    #  FIND (keyword → normalize → RAG)
    # =========================================================

    GENERIC_FIND_TERMS = {
        "*",
        "mail",
        "mails",
        "email",
        "emails",
        "all",
        "everything",
    }

    instruction: Optional[str] = None

    if find_raw:
        cleaned = clean_find_query(find_raw).lower().strip()

        if cleaned in GENERIC_FIND_TERMS:
            trace.append(f"FIND: ignored generic term '{cleaned}'")
            instruction = None
        else:
            instruction = cleaned
            trace.append(f"FIND: '{instruction}'")

    if instruction:
        messages_kw = keyword_filter(messages, instruction)
        if messages_kw:
            trace.append("FIND: keyword match")
            messages, message_embeddings = filter_with_embeddings(
                messages,
                message_embeddings,
                messages_kw
            )
        else:
            normalized = normalize_compound(instruction)
            trace.append(f"FIND: normalized '{normalized}'")
            messages_norm = keyword_filter(messages, normalized)

            if messages_norm:
                messages, message_embeddings = filter_with_embeddings(
                    messages,
                    message_embeddings,
                    messages_norm
                )

            else:
                trace.append("FIND: fallback to RAG")
                message_embeddings = ensure_message_embeddings(
                    messages,
                    message_embeddings
                )
                messages_rag = rag_filter_emails(messages, message_embeddings, instruction, trace, k=999)

                messages, message_embeddings = filter_with_embeddings(
                    messages,
                    message_embeddings,
                    messages_rag
                )

        if not messages:
            return "No relevant mails found.\n\nTRACE:\n" + "\n".join(trace)

    # =========================================================
    #  RETURN TYPES
    # =========================================================
    if rtype == Datatype.TABLE:
        rows = [
            (
                m.get("date", ""),
                m.get("from", ""),
                m.get("to", ""),
                m.get("cc", ""),
                m.get("subject", ""),
                m.get("hasAttachments", False),
            )
            for m in messages
        ]
        return (
            tabulate(
                rows,
                headers=["date", "from", "to", "cc", "subject", "hasAttachments"],
                tablefmt="psql",
            )
            + "\n\nTRACE:\n"
            + "\n".join(trace)
        )

    if rtype == Datatype.TEXT:
        if len(messages) > 1:
            message_embeddings = ensure_message_embeddings(
                messages,
                message_embeddings
            )
            trace.append("RAG(TEXT): selecting single representative mail")
            text_query = instruction or "most relevant email"
            messages_text = rag_filter_emails(messages, message_embeddings, text_query, trace, k=1)

            if messages_text:
                messages, message_embeddings = filter_with_embeddings(
                    messages,
                    message_embeddings,
                    messages_text
                )

            else:
                trace.append(
                    "RAG(TEXT): no confident representative found, "
                    "falling back to first message"
                )
                messages = [messages[0]]

        trace.append("LLM(TEXT): SUMMARY")
        summary = llm_summarize(messages, instruction)
        return summary + "\n\nTRACE:\n" + "\n".join(trace)

    return "Invalid return type for email."


