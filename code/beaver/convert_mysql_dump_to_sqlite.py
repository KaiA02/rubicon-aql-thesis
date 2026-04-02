import os
import re
import sqlite3
from pathlib import Path

# Paths for input dump and output SQLite database
BASE = Path(__file__).resolve().parent
MYSQL_DUMP = BASE / "new_dw_indexed.sql"
SQLITE_DB = BASE / "beaver.sqlite3"


def clean_line(line: str) -> str:
    """
    Remove or convert MySQL-specific syntax so SQLite can process it.
    """
    stripped = line.strip()

    # Skip MySQL comments and dump metadata
    if stripped.startswith("--"):
        return ""
    if stripped.startswith("/*") and stripped.endswith("*/;"):
        return ""
    if stripped.startswith("/*!") and stripped.endswith("*/;"):
        return ""

    # Skip MySQL session settings and commands SQLite does not support
    if stripped.upper().startswith("SET "):
        return ""
    if stripped.upper().startswith("CREATE DATABASE"):
        return ""
    if stripped.upper().startswith("USE "):
        return ""

    # Skip LOCK/UNLOCK and key-disable commands
    skip_tokens = [
        "LOCK TABLES", "UNLOCK TABLES",
        "DISABLE KEYS", "ENABLE KEYS"
    ]
    for tok in skip_tokens:
        if tok in stripped.upper():
            return ""

    # Remove MySQL backticks
    line = line.replace("`", "")

    # Remove MySQL table options (engine, charset, collations)
    line = re.sub(r"ENGINE=\w+\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"DEFAULT CHARSET=\w+\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"CHARSET=\w+\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"COLLATE=\w+\s*", "", line, flags=re.IGNORECASE)

    # Convert MySQL datatypes to SQLite-compatible ones
    line = re.sub(r"varchar\(\d+\)", "TEXT", line, flags=re.IGNORECASE)
    line = re.sub(r"char\(\d+\)", "TEXT", line, flags=re.IGNORECASE)
    line = re.sub(r"int\(\d+\)", "INTEGER", line, flags=re.IGNORECASE)
    line = re.sub(r"datetime\(\d+\)", "DATETIME", line, flags=re.IGNORECASE)

    # ENUM -> TEXT for SQLite
    line = re.sub(r"ENUM\s*\(.*?\)", "TEXT", line, flags=re.IGNORECASE)

    # Remove MySQL index definitions in CREATE TABLE
    if stripped.upper().startswith("KEY "):
        return ""

    return line


def iter_statements(file):
    """
    Yield complete SQL statements ending with ';'.
    Keeps track of string quotes so semicolons inside strings don't break.
    """
    buf = []
    in_string = False
    str_char = None

    while True:
        line = file.readline()

        # End of file
        if not line:
            if buf:
                yield "".join(buf)
            break

        line = clean_line(line)
        if not line.strip():
            continue

        i = 0
        while i < len(line):
            ch = line[i]

            # Enter or exit string mode
            if ch in ('"', "'"):
                if not in_string:
                    in_string = True
                    str_char = ch
                elif str_char == ch:
                    in_string = False

            # If semicolon outside string -> statement ends
            if ch == ";" and not in_string:
                buf.append(ch)
                yield "".join(buf)
                buf = []
            else:
                buf.append(ch)

            i += 1


def convert():
    """
    Main conversion routine: read dump -> convert -> execute SQL in SQLite.
    """
    if not MYSQL_DUMP.exists():
        raise FileNotFoundError(f"Dump not found: {MYSQL_DUMP}")

    # Remove old SQLite file
    if SQLITE_DB.exists():
        SQLITE_DB.unlink()

    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()

    # Uncomment to debug:
    # print("Starting conversion...")
    # print("Reading:", MYSQL_DUMP)

    with MYSQL_DUMP.open("r", encoding="utf-8", errors="ignore") as f:
        for stmt in iter_statements(f):
            sql = stmt.strip()
            if not sql or sql == ";":
                continue

            # Uncomment to debug each SQL statement:
            # print("Executing SQL:", sql)

            try:
                cur.executescript(sql)
            except Exception as e:
                print("SQL Error:", e)
                print("Failed statement:", sql)
                pass

    conn.commit()
    conn.close()

    # Uncomment to debug:
    # print("Conversion finished.")
    # print("SQLite DB created at:", SQLITE_DB)


if __name__ == "__main__":
    convert()
