import sqlite3
import re
from typing import List, Optional
from dataclasses import dataclass

from tabulate import tabulate
from openai import OpenAI

from enums import Datatype
from Configurations.config_loader import OPENAI_API_KEY
from rag.rag import RAG, EmbeddingModel, LowConfidenceMatchError
from rag.embedding_model import SentenceTransformerEmbeddingModel
from apis.llm_quotes import extract_llm_chunks, contains_llm_marker


# =========================================================
#  LLM client (ONLY for '...' translation, NOT embeddings)
# =========================================================

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================================================
#  Embedding Model
# =========================================================

EMBEDDING_MODEL = SentenceTransformerEmbeddingModel(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
#  SQLite metadata helpers
# =========================================================

def get_table_names(db_path: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = [row[0] for row in cur.fetchall()]
    conn.close()
    return names


def get_table_columns(db_path: str, table: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(f"PRAGMA table_info({table})")
        rows = cur.fetchall()
        cols = [r[1] for r in rows]
    except Exception:
        cols = []
    conn.close()
    return cols


def exec_sqlite(db_path: str, sql: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        conn.close()
        return None, str(e)

# =========================================================
#  SQLite error classification
# =========================================================

@dataclass
class SqliteIssue:
    kind: str       # NO_TABLE | NO_COLUMN | OTHER
    token: str
    message: str


def classify_sqlite_error(err: str) -> SqliteIssue:
    msg = err or ""

    m = re.search(r"no such table:\s*([A-Za-z0-9_]+)", msg, re.IGNORECASE)
    if m:
        return SqliteIssue("NO_TABLE", m.group(1), msg)

    m = re.search(r"no such column:\s*([A-Za-z0-9_]+)", msg, re.IGNORECASE)
    if m:
        return SqliteIssue("NO_COLUMN", m.group(1), msg)

    return SqliteIssue("OTHER", "", msg)

# =========================================================
#  String literal normalization
#  "Boston" -> 'Boston'
# =========================================================

def normalize_sql_string_literals(sql: str) -> str:
    def repl(m):
        return "= '" + m.group(1) + "'"
    return re.sub(r'=\s*"([^"]*)"', repl, sql)

# =========================================================
#  LLM output sanitizer
# =========================================================

def sanitize_llm_where(expr: str) -> str:
    s = (expr or "").strip()

    # remove markdown code fences
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s)

    # remove leading WHERE
    if re.match(r"^\s*WHERE\b", s, re.IGNORECASE):
        s = re.sub(r"^\s*WHERE\b\s*", "", s, flags=re.IGNORECASE)

    s = s.rstrip(";").strip()

    if len(s) >= 2 and (
        (s[0] == s[-1] == "'") or
        (s[0] == s[-1] == '"') or
        (s[0] == s[-1] == "`")
    ):
        s = s[1:-1].strip()

    return s

# =========================================================
#  RAG helpers
# =========================================================

def rag_best_match(query: str, candidates: List[str], emb: EmbeddingModel) -> Optional[str]:
    if not candidates:
        return None
    vecs = [emb.embed(c) for c in candidates]
    try:
        idx = RAG(query, vecs, k=1, emb_model=emb)[0]
    except LowConfidenceMatchError:
        return None
    return candidates[idx]


def replace_identifier(sql: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, sql)


def extract_from_table(sql: str) -> Optional[str]:
    m = re.search(r"\bFROM\s+([A-Za-z0-9_]+)\b", sql, re.IGNORECASE)
    return m.group(1) if m else None

# =========================================================
#  LLM translators (ONLY for '...')
# =========================================================

def llm_pick_table(db_path: str, chunk: str) -> str:
    tables = get_table_names(db_path)
    prompt = f"""
Pick EXACTLY ONE table name from this list.
Return only the name.

Tables: {tables}
User description: {chunk}
""".strip()

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content.strip()


def llm_pick_columns(db_path: str, table: str, chunk: str) -> str:
    cols = get_table_columns(db_path, table)
    prompt = f"""
Map the request to column names.

Rules:
- Return comma-separated names OR *
- Use ONLY these columns: {cols}

Request: {chunk}
""".strip()

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content.strip()


def llm_pick_where(db_path: str, table: str, chunk: str) -> str:
    cols = get_table_columns(db_path, table)
    prompt = f"""
Convert to SQLite WHERE condition.
Return ONLY the condition (no WHERE).

Allowed columns: {cols}
User filter: {chunk}
""".strip()

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content.strip()

# =========================================================
#  Replace quoted LLM chunks inside a larger select string
#  Example: "city, 'zip type'" -> "city, ZIP_TYPE"
# =========================================================

def translate_llm_chunks_in_select(db_path: str, table: str, select_expr: str, trace: List[str]) -> str:
    out = select_expr
    for chunk in extract_llm_chunks(select_expr):
        mapped = llm_pick_columns(db_path, table, chunk)
        trace.append(f"LLM(FIND): '{chunk}' -> {mapped}")
        out = re.sub(rf"'{re.escape(chunk)}'", mapped, out, count=1)
    return out


def translate_llm_chunks_in_where(db_path: str, table: str, where_expr: str, trace: List[str]) -> str:
    out = where_expr
    for chunk in extract_llm_chunks(where_expr):
        raw = llm_pick_where(db_path, table, chunk)
        sanitized = sanitize_llm_where(raw)
        trace.append(f"LLM(WHERE): '{chunk}' -> {raw}")
        if raw != sanitized:
            trace.append("WARN: stripped wrapping quotes/backticks from LLM WHERE")
        out = re.sub(rf"'{re.escape(chunk)}'", sanitized, out, count=1)
    return sanitize_llm_where(out)

# =========================================================
#  SQL builder (SQL-first, LLM only if '...')
# =========================================================

def build_sql(db_path: str, parsed: dict, trace: List[str], limit: int = 50) -> str:
    chunk = None

    find_raw = parsed.get("find_raw", "").strip()
    from_raw = parsed.get("subsource", "").strip()
    where_raw = parsed.get("where_raw", "").strip()

    if '"' in find_raw:
        trace.append('WARN: FIND contains double quotes (") → SQL literal, not LLM')

    table = from_raw
    if contains_llm_marker(from_raw):
        chunk = extract_llm_chunks(from_raw)[0]
        table = llm_pick_table(db_path, chunk)
        trace.append(f"LLM(FROM): '{chunk}' -> {table}")

    select = find_raw if find_raw else "*"
    if contains_llm_marker(find_raw):
        select = translate_llm_chunks_in_select(db_path, table, find_raw, trace)

    where = where_raw
    if where_raw and contains_llm_marker(where_raw):
        where = translate_llm_chunks_in_where(db_path, table, where_raw, trace)
    if where_raw and not contains_llm_marker(where_raw):
        trace.append("WHERE: no semantic interpretation requested")

    sql = f"SELECT {select} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    sql += f" LIMIT {limit}"

    sql = normalize_sql_string_literals(sql)
    trace.append(f"BUILT SQL:\n{sql}")
    return sql

# =========================================================
#  RAG repair loop
# =========================================================

def rag_repair_loop(db_path: str, sql: str, emb: EmbeddingModel, trace: List[str]):
    current = sql

    for i in range(5):
        rows, err = exec_sqlite(db_path, current)
        if err is None:
            trace.append("EXEC OK")
            return rows, None, current

        issue = classify_sqlite_error(err)
        trace.append(f"SQL ERROR: {issue.message}")

        if issue.kind == "NO_TABLE":
            best = rag_best_match(issue.token, get_table_names(db_path), emb)
            if not best:
                trace.append(f"RAG FAILED for table '{issue.token}'")
                return None, err, current
            current = replace_identifier(current, issue.token, best)
            trace.append(f"RAG TABLE FIX: {issue.token} -> {best}")
            continue

        if issue.kind == "NO_COLUMN":
            table = extract_from_table(current)
            if not table:
                return None, err, current
            best = rag_best_match(issue.token, get_table_columns(db_path, table), emb)
            if not best:
                trace.append(f"RAG FAILED for column '{issue.token}'")
                return None, err, current
            current = replace_identifier(current, issue.token, best)
            trace.append(f"RAG COLUMN FIX: {issue.token} -> {best}")
            continue

        return None, err, current

    return None, "RAG loop exceeded max iterations", current

# =========================================================
#  Validation error
# =========================================================

class DSLValidationError(Exception):
    pass


def validate_parsed_dsl(parsed: dict):
    where_raw = parsed.get("where_raw", "").strip().lower()
    if where_raw.startswith("return"):
        raise DSLValidationError(
            "`where` must contain a condition. "
            "Remove `where` or add a filter."
        )

# =========================================================
#  MAIN ENTRYPOINT
# =========================================================

def get_database_answer(db_path: str, parsed: dict):
    trace: List[str] = []

    try:
        validate_parsed_dsl(parsed)
    except DSLValidationError as e:
        return f"DSL error: {e}"

    rtype = parsed.get("rtype") or Datatype.TABLE

    emb = EMBEDDING_MODEL
    trace.append("EMBEDDINGS: all-MiniLM-L6-v2")

    try:
        sql = build_sql(db_path, parsed, trace)
    except Exception as e:
        return f"Build error: {e}\n\nTRACE:\n" + "\n".join(trace)

    rows, err, final_sql = rag_repair_loop(db_path, sql, emb, trace)

    if err:
        return f"Database error: {err}\n\nFINAL SQL:\n{final_sql}\n\nTRACE:\n" + "\n".join(trace)

    if not rows:
        return "No rows found.\n\nTRACE:\n" + "\n".join(trace)

    headers = rows[0].keys()
    data = [tuple(r[h] for h in headers) for r in rows]

    if rtype == Datatype.TEXT:
        return f"{len(rows)} rows returned.\n\nTRACE:\n" + "\n".join(trace)

    return tabulate(data, headers=headers, tablefmt="psql") + "\n\nTRACE:\n" + "\n".join(trace)
