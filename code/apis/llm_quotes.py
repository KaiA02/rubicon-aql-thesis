from typing import List


def extract_string_literals(text: str) -> List[str]:
    """
    Extracts ALL single-quoted string literals using true quote counting.
    Example:
      "'P' AND 'city is boston'"
      -> ["P", "city is boston"]
    """
    literals = []
    buf = []
    in_string = False

    for ch in text or "":
        if ch == "'":
            if in_string:
                literals.append("".join(buf))
                buf = []
                in_string = False
            else:
                in_string = True
        else:
            if in_string:
                buf.append(ch)

    return literals


def extract_llm_chunks(text: str) -> List[str]:
    """
    LLM chunks are quoted literals with semantic intent
    (heuristic: contain at least one whitespace).
    """
    return [lit for lit in extract_string_literals(text) if " " in lit]


def contains_llm_marker(text: str) -> bool:
    return bool(extract_llm_chunks(text))
