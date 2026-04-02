import wikipedia
import re
import requests
from functools import lru_cache
from datetime import datetime, timezone
from apis.metadata_api import WIKI_SCHEMA
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
from apis.wiki_api import build_wiki_user_agent
from apis.wiki_api import clean_wiki_title
import time


# =========================================================
#  CACHING (PROCESS-LOCAL)
# =========================================================

@lru_cache(maxsize=128)
def _cached_summary(title: str) -> str:
    try:
        return wikipedia.summary(title, auto_suggest=False, redirect=True)
    except Exception:
        return ""


@lru_cache(maxsize=128)
def _cached_page(title: str):
    # IMPORTANT: always explicit title-based loading
    return wikipedia.page(title=title, auto_suggest=False, redirect=True)


# =========================================================
#  MEDIAWIKI HELPERS
# =========================================================

def _fetch_last_modified(title: str, trace: list[str]) -> str | None:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "timestamp",
        "rvlimit": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }

    try:
        trace.append("WIKI: FETCH last_modified (MediaWiki API)")
        headers = {
            "User-Agent": build_wiki_user_agent()
        }

        r = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        trace.append(f"WIKI: last_modified FETCH FAILED ({e})")
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    revisions = page.get("revisions")
    if not revisions:
        return None

    ts = revisions[0].get("timestamp")
    if not ts:
        return None

    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    iso = dt.astimezone(timezone.utc).isoformat()
    trace.append(f"WIKI: last_modified = {iso}")
    return iso


def _fetch_mediawiki_html(title: str, trace: list[str]) -> str | None:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "redirects": 1,
    }

    try:
        trace.append("WIKI: FETCH HTML (MediaWiki API parse)")
        headers = {
            "User-Agent": build_wiki_user_agent()
        }

        r = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        trace.append(f"WIKI: HTML FETCH ERROR ({e})")
        return None

    if "error" in data:
        return None

    return data.get("parse", {}).get("text", {}).get("*")


def _fetch_sources(title: str, trace: list[str]) -> list[str]:
    html = _fetch_mediawiki_html(title, trace)
    if not html:
        trace.append("WIKI: SOURCES HTML NOT AVAILABLE")
        return []

    soup = BeautifulSoup(html, "html.parser")
    urls: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.startswith("http"):
            continue

        parsed = urlparse(href)
        if not parsed.netloc:
            continue

        if "wikipedia.org" in parsed.netloc:
            continue
        if "wikimedia.org" in parsed.netloc:
            continue

        urls.add(href)

    trace.append(f"WIKI: SOURCES FOUND = {len(urls)}")
    return sorted(urls)


# =========================================================
#  ATTRIBUTE RESOLVER (GENERIC, SCHEMA-DRIVEN)
# =========================================================

def resolve_wiki_attribute(page, attr: str, trace: list[str]):
    attr = attr.lower()
    trace.append(f"WIKI: RESOLVE ATTRIBUTE '{attr}'")

    if attr == "pageid":
        return page.pageid

    if attr == "title":
        return page.title

    if attr == "url":
        return page.url

    if attr == "abstract":
        return _cached_summary(page.title)

    if attr == "text":
        return page.content

    if attr == "images":
        return page.images or []

    if attr == "links":
        return page.links or []

    if attr == "categories":
        return page.categories or []

    if attr == "sources":
        return _fetch_sources(page.title, trace)

    if attr == "last_modified":
        return _fetch_last_modified(page.title, trace)

    # should never happen due to schema validation
    raise KeyError(attr)

# =========================================================
#  HELPER
# =========================================================

def _title_from_wikipedia_url(url: str) -> str | None:
    try:
        p = urlparse(url.strip())
    except Exception:
        return None

    if p.scheme not in ("http", "https"):
        return None

    host = (p.netloc or "").lower()
    if "wikipedia.org" not in host:
        return None

    # expected: /wiki/<Title>
    path = p.path or ""
    if not path.startswith("/wiki/"):
        return None

    title_part = path[len("/wiki/"):]
    if not title_part:
        return None

    # decode + normalize
    title_part = unquote(title_part)
    title_part = title_part.replace("_", " ")
    return title_part.strip() or None


def _find_outside_quotes(s: str, needle: str) -> int:
    """
    Returns index of needle in s, but only if it occurs outside '...' or "..."
    """
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

def _tokenize_where(s: str):
    """
    Tokenizes WHERE into:
      - '(', ')'
      - operators: AND, OR, NOT
      - condition strings
    Outside of quotes only.
    """
    tokens = []
    buf = []
    in_single = False
    in_double = False
    i = 0

    def flush_buf():
        part = "".join(buf).strip()
        if part:
            tokens.append(part)
        buf.clear()

    while i < len(s):
        ch = s[i]

        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue

        if not in_single and not in_double:
            if ch == "(":
                flush_buf()
                tokens.append("(")
                i += 1
                continue
            if ch == ")":
                flush_buf()
                tokens.append(")")
                i += 1
                continue

            for op in ("not", "and", "or"):
                if (
                        s[i:i + len(op)].lower() == op
                        and (i == 0 or s[i - 1].isspace())
                        and (i + len(op) == len(s) or s[i + len(op)].isspace())
                ):
                    flush_buf()
                    tokens.append(op.strip().upper())
                    i += len(op)
                    break
            else:
                buf.append(ch)
                i += 1
            continue

        buf.append(ch)
        i += 1

    flush_buf()
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

    # OR (lowest)
    def parse_expr(self):
        node = self.parse_and()
        while self._peek() == "OR":
            self._eat("OR")
            node = ("OR", node, self.parse_and())
        return node

    # AND
    def parse_and(self):
        node = self.parse_not()
        while self._peek() == "AND":
            self._eat("AND")
            node = ("AND", node, self.parse_not())
        return node

    # NOT
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
            raise ValueError(f"Unexpected token '{tok}' in WHERE clause")

        self._eat()
        if not tok.strip():
            raise ValueError("Empty condition in WHERE clause")
        return ("COND", tok)

def apply_wiki_where_logical(page, where_raw: str, trace: list[str]) -> bool:
    if not where_raw or not where_raw.strip():
        trace.append("WIKI: WHERE EMPTY → PASS")
        return True

    try:
        tokens = _tokenize_where(where_raw)
        parser = _WhereParser(tokens)
        tree = parser.parse_expr()

        if parser._peek() is not None:
            raise ValueError("Unexpected token at end of WHERE clause")

    except ValueError as e:
        raise ValueError(
            "Invalid WHERE logic.\n"
            f"{e}\n\n"
            "Examples:\n"
            "  where 'MIT' in wiki.abstract and not 'banana' in wiki.abstract\n"
            "  where ('A' in wiki.abstract or 'B' in wiki.abstract) and not 'C' in wiki.title"
        )

    def eval_node(node):
        kind = node[0]

        if kind == "COND":
            return apply_wiki_where(page, node[1], trace)

        if kind == "NOT":
            trace.append("WIKI: LOGICAL NOT")
            return not eval_node(node[1])

        if kind == "AND":
            trace.append("WIKI: LOGICAL AND")
            return eval_node(node[1]) and eval_node(node[2])

        if kind == "OR":
            trace.append("WIKI: LOGICAL OR")
            return eval_node(node[1]) or eval_node(node[2])

        raise RuntimeError("Invalid WHERE AST")

    return eval_node(tree)


def _split_where_operator(where_raw: str):
    """
    Splits WHERE into (op, lhs, rhs) where op is 'in' or '='.
    Operators are detected ONLY outside quotes.
    """
    w = where_raw.strip()

    idx_in = _find_outside_quotes(w, " in ")
    idx_eq = _find_outside_quotes(w, "=")

    if idx_in != -1 and (idx_eq == -1 or idx_in < idx_eq):
        lhs = w[:idx_in].strip()
        rhs = w[idx_in + len(" in "):].strip()
        return "in", lhs, rhs

    if idx_eq != -1:
        lhs = w[:idx_eq].strip()
        rhs = w[idx_eq + 1:].strip()
        return "=", lhs, rhs

    raise ValueError(
        "Invalid WHERE syntax.\n"
        "Use:\n"
        "  where 'value' in wiki.<attribute>\n"
        "  or: where wiki.<attribute> = 'value'"
    )



# =========================================================
#  SEARCH VALUE EXTRACTION (FOR ATTRIBUTE QUERIES)
# =========================================================

def extract_search_value_from_where(where_raw: str) -> str:
    """
    Extracts the semantic search value from WHERE:
      wiki.title = "Germany"
      wiki.title = 'Germany'
      'MIKE' in wiki.abstract
    """
    w = where_raw.strip()

    # wiki.attr = "X"
    m = re.search(r"=\s*(['\"])(.*?)\1", w)
    if m:
        return m.group(2)

    # 'X' in wiki.attr
    m = re.search(r"(['\"])(.*?)\1", w)
    if m:
        return m.group(2)

    raise ValueError(
        "Could not determine search value from WHERE clause.\n"
        "Use:\n"
        "  where wiki.<attribute> = \"value\"\n"
        "  or: where 'value' in wiki.<attribute>"
    )


# =========================================================
#  WHERE PARSER + APPLY (DETERMINISTIC, NO LLM)
# =========================================================

def apply_wiki_where(page, where_raw: str, trace: list[str]) -> bool:
    trace.append(f"WIKI: APPLY WHERE '{where_raw}'")

    if not where_raw or not where_raw.strip():
        trace.append("WIKI: WHERE EMPTY → PASS")
        return True

    w = where_raw.strip()

    # -------------------------------------------------
    # operator detection
    # -------------------------------------------------
    op, lhs, rhs = _split_where_operator(w)

    # -------------------------------------------------
    # extract quoted value (supports '...' and \"...\")
    # -------------------------------------------------
    side = lhs if op == "in" else rhs
    m = re.search(r"(['\"])(.*?)\1", side)
    if not m:
        raise ValueError(
            "Invalid WHERE syntax.\n"
            "Missing quoted value.\n"
            "Examples:\n"
            "  where 'MIKE' in wiki.abstract\n"
            "  where \"Germany\" in wiki.title"
        )

    needle = m.group(2).lower()

    # -------------------------------------------------
    # extract attribute target
    # -------------------------------------------------
    attr_expr = rhs if op == "in" else lhs
    attr_expr = attr_expr.strip().lower()

    if attr_expr.startswith("wiki."):
        attr = attr_expr.split(".", 1)[1]
    elif attr_expr.startswith("wikipedia."):
        attr = attr_expr.split(".", 1)[1]
    else:
        raise ValueError(
            "Invalid WHERE target.\n"
            "Expected: wiki.<attribute> or wikipedia.<attribute>"
        )

    # -------------------------------------------------
    # validate attribute via schema
    # -------------------------------------------------
    if attr not in WIKI_SCHEMA["page"]["attributes"]:
        raise ValueError(
            f"Unknown Wikipedia attribute '{attr}'.\n"
            f"Use: wiki ?  or  wiki.page ?"
        )

    value = resolve_wiki_attribute(page, attr, trace)

    if value is None:
        trace.append(f"WIKI: ATTRIBUTE '{attr}' IS NULL")
        return False

    # -------------------------------------------------
    # apply operator
    # -------------------------------------------------
    if isinstance(value, list):
        values = [str(v).lower() for v in value]
    else:
        values = [str(value).lower()]

    if op == "in":
        hit = any(needle in v for v in values)
    else:  # "="
        hit = any(needle == v for v in values)

    trace.append(f"WIKI: WHERE RESULT → {'MATCH' if hit else 'NO MATCH'}")
    return hit


# =========================================================
#  SEARCH + FILTER PIPELINE
# =========================================================

def wikipedia_search_and_filter(
    query: str,
    where_raw: str,
    trace: list[str],
    limit: int = 13,
):
    trace.append("WIKI: SEARC<H MODE (NO LLM)")
    trace.append(f"WIKI: SEARCH QUERY = '{query}'")

    # -------------------------------------------------
    # DSL NORMALIZATION (remove surrounding quotes)
    # -------------------------------------------------
    query = clean_wiki_title(query)
    trace.append(f"WIKI: NORMALIZED SEARCH QUERY = '{query}'")

    pages = []

    # -------------------------------------------------
    # DIRECT URL → TITLE SHORT-CIRCUIT
    # -------------------------------------------------
    direct_title = _title_from_wikipedia_url(query)
    if direct_title:
        trace.append(f"WIKI: DETECTED URL → TITLE '{direct_title}'")
        try:
            trace.append(f"WIKI: LOAD PAGE (DIRECT) '{direct_title}'")
            page = _cached_page(direct_title)
        except Exception as e:
            trace.append(f"WIKI: DIRECT PAGE LOAD FAILED '{direct_title}' ({e})")
        else:
            if apply_wiki_where_logical(page, where_raw, trace):
                pages.append(page)

        trace.append(f"WIKI: FILTERED RESULTS = {len(pages)}")
        return pages

    # -------------------------------------------------
    # NORMAL SEARCH PATH
    # -------------------------------------------------
    titles = wikipedia.search(query, results=limit)
    trace.append(f"WIKI: SEARCH RESULTS = {len(titles)}")

    for title in titles:
        time.sleep(0.05)
        try:
            trace.append(f"WIKI: LOAD PAGE '{title}'")
            page = _cached_page(title)
        except Exception as e:
            trace.append(f"WIKI: PAGE LOAD FAILED '{title}' ({e})")
            continue

        if not apply_wiki_where_logical(page, where_raw, trace):
            continue

        pages.append(page)

    trace.append(f"WIKI: FILTERED RESULTS = {len(pages)}")
    return pages

