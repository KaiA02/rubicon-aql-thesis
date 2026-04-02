import re
import wikipedia
import requests
from tabulate import tabulate
from io import StringIO
from openai import OpenAI
from enums import Datatype
from Configurations.config_loader import OPENAI_API_KEY
from Configurations.config_loader import (
    WIKI_USER_AGENT_NAME,
    WIKI_CONTACT_EMAIL,
)
from apis.llm_quotes import extract_llm_chunks, contains_llm_marker

# For table extraction
import pandas as pd
from bs4 import BeautifulSoup

wikipedia.set_lang("en")
client = OpenAI(api_key=OPENAI_API_KEY)

def _build_user_agent() -> str:
    if WIKI_CONTACT_EMAIL:
        return (
            f"{WIKI_USER_AGENT_NAME} "
            f"(Academic bachelor thesis project; "
            f"contact: {WIKI_CONTACT_EMAIL})"
        )
    return WIKI_USER_AGENT_NAME

def build_wiki_user_agent() -> str:
    return _build_user_agent()


# =========================================================
#  LLM: Wikipedia TITLE translator
# =========================================================

def llm_pick_wikipedia_title(user_query: str) -> str:
    prompt = f"""
You are translating a user request into a Wikipedia page title.

Rules:
- Return EXACTLY ONE Wikipedia page title
- Use the most canonical and common page
- Do NOT add explanations
- Do NOT invent new concepts

User request:
{user_query}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


# =========================================================
#  LLM: Summary filter (semantic gatekeeper)
# =========================================================

def llm_filter_summary(title: str, summary: str, flt: str) -> str:
    prompt = (
        f"Title:\n{title}\n\n"
        f"Summary:\n{summary}\n\n"
        f"User filter:\n{flt}\n\n"
        "Decide whether the summary contains relevant information for the filter.\n"
        "If not, answer exactly:\n"
        "\"No relevant information for this filter in the Wikipedia summary.\""
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a careful Wikipedia filtering assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# =========================================================
#  Disambiguation helper
# =========================================================

def _choose_from_disambiguation(options, flt: str | None) -> str:
    if flt:
        fl = flt.lower()
        for opt in options:
            if fl in opt.lower():
                return opt
    return options[0]


# =========================================================
#  MediaWiki HTML Fetch
# =========================================================

def _fetch_mediawiki_html(title: str, trace: list[str]) -> str | None:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "redirects": 1,
    }

    headers = {
        "User-Agent": _build_user_agent()
    }

    try:
        trace.append("WIKI: FETCH HTML (MediaWiki API parse)")
        r = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        trace.append(f"WIKI: HTML FETCH ERROR: {e}")
        return None

    if "error" in data:
        trace.append(f"WIKI: HTML PARSE ERROR: {data['error'].get('info', 'unknown')}")
        return None

    return data.get("parse", {}).get("text", {}).get("*")


# =========================================================
#  SUMMARY MODE
# =========================================================

def clean_wiki_title(title: str) -> str:
    """
    Removes surrounding double quotes ONLY if they wrap real content.
    Keeps single quotes intact (LLM marker).
    Does NOT strip a single standalone quote like '"'.
    """
    if not title:
        return ""

    title = title.strip()

    if (
        len(title) >= 2
        and title.startswith('"')
        and title.endswith('"')
        and title[1:-1].strip()
    ):
        title = title[1:-1]

    return title.strip()


def wikipedia_summary(title: str, flt: str, rtype: Datatype, trace: list[str]) -> str:
    title = clean_wiki_title(title)
    trace.append(f"WIKI: INPUT TITLE = {title}")

    # LLM TITLE TRANSLATION (ONLY IF '...')
    if contains_llm_marker(title):
        chunk = extract_llm_chunks(title)[0]
        new_title = llm_pick_wikipedia_title(chunk)
        trace.append(f"LLM(WIKI_TITLE): '{chunk}' -> '{new_title}'")
        title = new_title

    # FETCH SUMMARY
    try:
        summary = wikipedia.summary(title, auto_suggest=True, redirect=True)


    except wikipedia.DisambiguationError as e:
        choice = _choose_from_disambiguation(e.options[:5], flt)

        if choice != title:
            trace.append(f"WIKI: DISAMBIGUATION → '{choice}'")
        else:
            trace.append("WIKI: DISAMBIGUATION (NO CHANGE)")

        summary = wikipedia.summary(choice, auto_suggest=False, redirect=True)
        title = choice


    except wikipedia.PageError:
        trace.append(f"WIKI: PAGE LOAD FAILED – '{title}'")

        # -------------------------------------------------
        # TECHNICAL SEARCH FALLBACK (SAME INTENT)
        # -------------------------------------------------
        trace.append("WIKI: TECHNICAL SEARCH FALLBACK")
        hits = wikipedia.search(title, results=5)
        if not hits:
            trace.append("WIKI: SEARCH FALLBACK → NO RESULTS")
            return f"Could not load '{title}' (no page found)."
        choice = _choose_from_disambiguation(hits, flt)
        trace.append(f"WIKI: SEARCH FALLBACK → '{choice}'")

        try:
            page = wikipedia.page(choice, auto_suggest=False, redirect=True)
            summary = page.summary
            title = choice
        except wikipedia.PageError:
            trace.append("WIKI: SEARCH FALLBACK PAGE LOAD FAILED")
            return f"Could not load '{title}' (no page found)."

    # NO FILTER
    if not flt or not flt.strip():
        trace.append("WIKI: NO FILTER")
        return summary if rtype == Datatype.TEXT else _meta_table(title, trace)

    # FILTER LOGIC
    # CASE 1: UNQUOTED FILTER → deterministic token match
    if not contains_llm_marker(flt):
        trace.append("WIKI: APPLY DETERMINISTIC FILTER")
        trace.append("WHERE: no semantic interpretation requested")

        flt_lower = flt.lower()
        summary_lower = summary.lower()

        if flt_lower in summary_lower:
            trace.append("WIKI: DIRECT MATCH")
            return summary if rtype == Datatype.TEXT else _meta_table(title, trace)

        tokens = [t for t in re.split(r"\W+", flt_lower) if len(t) > 3]
        if tokens and any(tok in summary_lower for tok in tokens):
            trace.append("WIKI: TOKEN MATCH")
            return summary if rtype == Datatype.TEXT else _meta_table(title, trace)

        trace.append("WIKI: NO MATCH (DETERMINISTIC)")
        return (
            "No relevant information found using deterministic filtering.\n"
            "If you intended semantic interpretation, put the filter in '...'."
        )

    # CASE 2: QUOTED FILTER → ALWAYS LLM
    trace.append("WIKI: SKIP TOKEN MATCH (LLM FILTER REQUESTED)")
    trace.append("LLM(WIKI_SUMMARY_FILTER)")

    decision = llm_filter_summary(title, summary, flt)
    if decision.startswith("No relevant information"):
        return decision

    return summary if rtype == Datatype.TEXT else _meta_table(title, trace, note=decision)


def _meta_table(title: str, trace: list[str], note: str | None = None):
    trace.append("WIKI: META TABLE")

    try:
        page = wikipedia.page(title, auto_suggest=False, redirect=True)
    except wikipedia.PageError:
        return f"Could not load '{title}' (no page found)."

    if note:
        return tabulate(
            [[page.pageid, page.title, page.url, note]],
            headers=["pageid", "title", "url", "note"],
            tablefmt="psql",
        )

    return tabulate(
        [[page.pageid, page.title, page.url]],
        headers=["pageid", "title", "url"],
        tablefmt="psql",
    )




# =========================================================
#  TABLE MODE
# =========================================================

def wikipedia_tables(title: str, flt: str, rtype: Datatype, trace: list[str]):
    title = clean_wiki_title(title)
    trace.append("WIKI: TABLE MODE")

    # LLM TITLE TRANSLATION (ONLY IF '...')
    if contains_llm_marker(title):
        chunk = extract_llm_chunks(title)[0]
        new_title = llm_pick_wikipedia_title(chunk)
        trace.append(f"LLM(WIKI_TITLE): '{chunk}' -> '{new_title}'")
        title = new_title

    # Fetch robust HTML from MediaWiki
    html = _fetch_mediawiki_html(title, trace)

    # Fallback: try wikipedia library HTML
    if not html:
        trace.append("WIKI: HTML FALLBACK (wikipedia.page().html())")
        try:
            page = wikipedia.page(title, auto_suggest=True, redirect=True)

        except wikipedia.DisambiguationError as e:
            choice = _choose_from_disambiguation(e.options[:5], flt)

            if choice != title:
                trace.append(f"WIKI: DISAMBIGUATION → '{choice}'")
                title = choice
            else:
                trace.append("WIKI: DISAMBIGUATION (NO CHANGE)")

            page = wikipedia.page(title, auto_suggest=False, redirect=True)

        except wikipedia.PageError:
            trace.append(f"WIKI: PAGE ERROR – '{title}' NOT FOUND")
            return "No tables found."

        html = page.html()

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    if not tables:
        trace.append("WIKI: NO TABLE ELEMENTS FOUND")
        return "No tables found."

    try:
        dfs = pd.read_html(StringIO(html))
    except ValueError:
        trace.append("WIKI: TABLE PARSE FAILED")
        return "No tables found."

    if flt:
        trace.append("WIKI: TABLE FILTER (SUBSTRING)")
        f = flt.lower()
        dfs = [
                  df for df in dfs
                  if df.astype(str).apply(
                lambda c: c.str.lower().str.contains(f).any()
            ).any()
              ] or dfs

    if rtype == Datatype.TEXT:
        return f"Found {len(dfs)} tables."

    if rtype == Datatype.TABLE:
        clean_tables = []

        for df in dfs:
            # 1) MultiIndex-Spalten flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    " – ".join(
                        str(c) for c in col
                        if c and str(c).lower() != "nan"
                    )
                    for col in df.columns
                ]

            # 2) NaN-Werte entfernen
            df = df.fillna("")

            # 3) Mehrzeilige Zellen einzeilig machen
            df = df.apply(
                lambda col: col.map(
                    lambda x: " ".join(str(x).splitlines())
                    if isinstance(x, str) else x
                )
            )

            clean_tables.append(
                tabulate(
                    df,
                    headers="keys",
                    tablefmt="psql",
                    showindex=False
                )
            )

        trace.append(
            "WIKI: NOTE – TABLE DISPLAY IS BEST-EFFORT (ASCII WIDTH & MULTILINE LIMITATIONS)")
        return "\n\n".join(clean_tables)

    return "Unsupported return type."


# =========================================================
#  PICTURE MODE
# =========================================================

_BAD_IMAGE_SUBSTRINGS = [
    "commons-logo",
    "wikimedia-logo",
    "question_book",
    "magnify-clip",
]

def _is_bad_image(url: str) -> bool:
    return any(bad in url.lower() for bad in _BAD_IMAGE_SUBSTRINGS)


def _pick_best_image(urls: list[str]) -> str:
    clean = [u for u in urls if not _is_bad_image(u)] or urls
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        for u in clean:
            if u.lower().endswith(ext):
                return u
    return clean[0]


def wikipedia_pictures(title: str, flt: str, rtype: Datatype, trace: list[str]):
    title = clean_wiki_title(title)
    trace.append("WIKI: PICTURE MODE")

    if contains_llm_marker(title):
        chunk = extract_llm_chunks(title)[0]
        new_title = llm_pick_wikipedia_title(chunk)
        trace.append(f"LLM(WIKI_TITLE): '{chunk}' -> '{new_title}'")
        title = new_title

    try:
        page = wikipedia.page(title, auto_suggest=True, redirect=True)

    except wikipedia.DisambiguationError as e:
        choice = _choose_from_disambiguation(e.options[:5], flt)

        if choice != title:
            trace.append(f"WIKI: DISAMBIGUATION → '{choice}'")
            title = choice
        else:
            trace.append("WIKI: DISAMBIGUATION (NO CHANGE)")

        page = wikipedia.page(title, auto_suggest=False, redirect=True)

    except wikipedia.PageError:
        trace.append(f"WIKI: PAGE ERROR – '{title}' NOT FOUND")
        return "No images found."

    images = page.images or []

    if flt:
        trace.append("WIKI: IMAGE FILTER (SUBSTRING)")
        images = [i for i in images if flt.lower() in i.lower()] or images

    if not images:
        return "No images found."

    if rtype == Datatype.PICTURE:
        return _pick_best_image(images)

    if rtype == Datatype.TEXT:
        return f"Found {len(images)} images."

    if rtype == Datatype.TABLE:
        return tabulate(
            list(enumerate(images)),
            headers=["index", "image_url"],
            tablefmt="psql",
        )

    return "Unsupported return type."


# =========================================================
#  MAIN ENTRYPOINT
# =========================================================

def get_wikipedia_answer(title, columns, subsource, flt, rtype):
    trace: list[str] = ["WIKI: REQUEST RECEIVED"]

    if rtype is None:
        rtype = Datatype.TEXT
        trace.append("WIKI: DEFAULT RETURN TYPE = TEXT")

    col_lower = [c.lower() for c in (columns or [])]

    if subsource is None:
        if "table" in col_lower:
            subsource = "tables"
        elif "picture" in col_lower or "image" in col_lower:
            subsource = "pictures"

    if subsource == "tables":
        result = wikipedia_tables(title, flt, rtype, trace)
    elif subsource == "pictures":
        result = wikipedia_pictures(title, flt, rtype, trace)
    else:
        result = wikipedia_summary(title, flt, rtype, trace)

    return result + "\n\nTRACE:\n" + "\n".join(trace)
