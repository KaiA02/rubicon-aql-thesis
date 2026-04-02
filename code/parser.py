import re
from enums import DataSource, Datatype

"""
Universal but structured query parser.

Syntax:
    find <something> from <something> [where <something>] [return <type>]

IMPORTANT:
- Single quotes '...' are PRESERVED (LLM marker)
- Parser does NOT decide about LLM / RAG / SQL
- where and return are optional
"""

# =========================================================
#  Helper
# =========================================================

def _mask_single_quoted(text: str) -> str:
    """
    Mask ONLY the content inside single quotes, but keep the quotes themselves.
    This ensures:
      - DSL keywords inside quotes are ignored
      - Regex spans still align with the original quoted segments
    """
    def repl(m):
        inner = m.group(1)
        return "'" + (" " * len(inner)) + "'"
    return re.sub(r"'([^']*)'", repl, text)

def _extract_from_original(original: str, span: tuple[int, int]) -> str:
    """
    Extract substring from original text using a span obtained
    from matching on a masked version of the text.
    """
    start, end = span
    return original[start:end]

# =========================================================
#  QUOTE NORMALIZATION
# =========================================================

def _normalize_smart_quotes(s: str) -> str:
    if not s:
        return s
    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "’": "'",
        "‘": "'",
        "‚": "'",
        "‛": "'",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s


# =========================================================
#  QUERY REGEX (WHERE + RETURN OPTIONAL)
# =========================================================

QUERY_PATTERN = re.compile(
    r"""
    find\s+(?P<find>.+?)                 
    \s+from\s+(?P<from>.+?)             
    (?:\s+where\s+(?P<where>.*?))?      
    (?:\s+return\s+(?P<return>\w+))?    
    \s*$                                 
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


# =========================================================
#  MAIN PARSER
# =========================================================

def parse_prompt(prompt: str):
    if not prompt or not prompt.strip():
        return {
            "error": "Empty query. Use: find <something> from <something> [where <filter>] [return <type>]"
        }

    text = _normalize_smart_quotes(prompt).strip()

    masked_text = _mask_single_quoted(text)
    match = QUERY_PATTERN.match(masked_text)

    if not match:
        return {
            "error": (
                "Invalid query structure. Use: "
                "find <something> from <something> [where <filter>] [return <type>]"
            )
        }

    # Extract RAW parts from ORIGINAL text using spans
    find_raw = _extract_from_original(text, match.span("find")).strip()
    from_raw = _extract_from_original(text, match.span("from")).strip()
    where_raw = (
        _extract_from_original(text, match.span("where")).strip()
        if match.group("where") is not None
        else ""
    )
    return_raw = (
        _extract_from_original(text, match.span("return")).strip().lower()
        if match.group("return") is not None
        else ""
    )

    # -----------------------------------------------------
    # SOURCE + SUBSOURCE (ONLY FROM from_raw)
    # -----------------------------------------------------

    source_alias_map = {
        "wikipedia": DataSource.WIKIPEDIA,
        "wiki": DataSource.WIKIPEDIA,
        "pile": DataSource.PILE,
        "chatgpt": DataSource.PILE,
        "llm": DataSource.PILE,
        "db": DataSource.DB,
        "database": DataSource.DB,
        "email": DataSource.EMAIL,
        "emails": DataSource.EMAIL,
        "mail": DataSource.EMAIL,
    }

    datasource = None
    subsource = None

    tokens = from_raw.split()
    first_token = tokens[0] if tokens else ""

    # Case 1: alias.subsource (db.ZIP_USA)
    if "." in first_token:
        alias, rest = first_token.split(".", 1)
        ds = source_alias_map.get(alias.lower())
        if ds:
            datasource = ds
            subsource = rest.strip()

            remainder = from_raw[len(first_token):].strip()
            if remainder:
                subsource = f"{subsource} {remainder}".strip()

    # Case 2: alias only (db ...)
    if datasource is None:
        ds = source_alias_map.get(first_token.lower())
        if ds:
            datasource = ds
            remainder = from_raw[len(first_token):].strip()
            if remainder:
                subsource = remainder

    # Case 3: fallback → DB
    if datasource is None:
        datasource = DataSource.DB
        subsource = from_raw

    # -----------------------------------------------------
    # RETURN TYPE
    # -----------------------------------------------------

    if return_raw:
        dtype_map = {
            "text": Datatype.TEXT,
            "table": Datatype.TABLE,
            "interface": Datatype.INTERFACE,
            "picture": Datatype.PICTURE if hasattr(Datatype, "PICTURE") else Datatype.TEXT,
        }
        rtype = dtype_map.get(return_raw)
        if rtype is None:
            return {"error": f"Unknown return type '{return_raw}'"}
    else:
        rtype = None

    # -----------------------------------------------------
    # BUILD RESULT
    # -----------------------------------------------------

    return {
        # RAW (für LLM / SQL / RAG Decision)
        "find_raw": find_raw,
        "from_raw": from_raw,
        "where_raw": where_raw,
        "return_raw": return_raw,

        # STRUCTURE
        "source": datasource,
        "subsource": subsource,
        "rtype": rtype,
    }
