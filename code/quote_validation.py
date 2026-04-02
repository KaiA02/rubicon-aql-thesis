import re

SMART_SINGLE_QUOTES = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
}

SMART_DOUBLE_QUOTES = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
}

def normalize_quotes(s: str) -> str:
    if not s:
        return s
    for bad, good in {**SMART_SINGLE_QUOTES, **SMART_DOUBLE_QUOTES}.items():
        s = s.replace(bad, good)
    return s

def validate_single_quotes(prompt: str) -> tuple[bool, bool, str]:
    """
    Returns:
      (is_valid, has_llm_marker, error_message)

    - Valid means: all single quotes are balanced (supports SQL escape inside '...': '')
    - has_llm_marker means: at least one complete '...'(non-empty) exists anywhere
    """
    s = normalize_quotes(prompt)

    in_single = False
    i = 0
    while i < len(s):
        if s[i] == "'":
            # Escaped quote inside a SQL string: ''
            if in_single and i + 1 < len(s) and s[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
        i += 1

    if in_single:
        return (
            False,
            False,
            "Syntax error: A single quotation mark (') was opened but not closed. "
            "Did you perhaps mean to write the relevant part completely in '...' so that I can use the LLM?"
        )

    has_marker = bool(re.search(r"'[^']+'", s))
    return (True, has_marker, "")