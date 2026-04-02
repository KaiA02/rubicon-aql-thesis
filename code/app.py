from flask import Flask, render_template, request

from apis.wiki_api import get_wikipedia_answer
from apis.thePile_api import get_pile_answer
from apis.database_api import get_database_answer
from apis.email_api import get_email_answer

from enums import DataSource, Datatype
from parser import parse_prompt
from quote_validation import validate_single_quotes, normalize_quotes
from apis.metadata_api import parse_metadata_query, get_metadata_answer, WIKI_SCHEMA
from apis.wiki_search_layer import (wikipedia_search_and_filter, resolve_wiki_attribute,
                                    extract_search_value_from_where)
from tabulate import tabulate

from authentification.auth import init_auth, AUTH_DB_PATH, User, bcrypt
from flask_login import login_user, login_required, logout_user, current_user
import sqlite3
from flask import redirect

import time




# =========================================================
#  Telemetry / Logging (system-level, source-agnostic)
# =========================================================

import json
import os
from datetime import datetime, timezone

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "general.log.jsonl")

def write_global_log(entry):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_user_log(username, entry):
    user_log_dir = os.path.join(LOG_DIR, "users")
    os.makedirs(user_log_dir, exist_ok=True)

    user_log_path = os.path.join(
        user_log_dir,
        f"{username}.jsonl"
    )

    with open(user_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")



os.makedirs(LOG_DIR, exist_ok=True)


def log_app_start():
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "app_start",
                "started_at": datetime.now(timezone.utc).isoformat()
            }) + "\n")
    except Exception:
        pass



app = Flask(__name__)
init_auth(app)
log_app_start()

DB_PATH = "./beaver/dw.sqlite"


@app.route("/", methods=["GET", "POST"])
@login_required
def home():

    user_input = ""
    response = None
    engine = None

    if request.method == "POST":

        # -------------------------------------------------
        # 1) Read + normalize input
        # -------------------------------------------------
        user_input = request.form.get("user_input", "").strip()
        user_input = normalize_quotes(user_input)

        # -------------------------------------------------
        # 2) Quote validation (syntax-level)
        # -------------------------------------------------
        is_valid, has_llm_marker, err = validate_single_quotes(user_input)
        if not is_valid:
            return render_template(
                "index.html",
                response=err,
                user_input=user_input,
                engine="parser",
            )

        # -------------------------------------------------
        # METADATA MODE: "<datasource>[.<query>] ?"
        # -------------------------------------------------
        meta = parse_metadata_query(user_input)
        if meta is not None:
            ds, query = meta
            response = get_metadata_answer(DB_PATH, ds, query)
            engine = "metadata"
            return render_template(
                "index.html",
                title="Bachelorproject",
                response=response,
                user_input=user_input,
                engine=engine,
            )

        # -------------------------------------------------
        # 3) Parse DSL
        # -------------------------------------------------
        parsed = parse_prompt(user_input)

        if "error" in parsed:
            # ---------------------------------------------
            # Metadata-aware error override for '?'
            # ---------------------------------------------
            if user_input.strip().endswith("?"):
                return render_template(
                    "index.html",
                    response=(
                        "Invalid metadata query.\n\n"
                        "Use one of:\n"
                        "  ?\n"
                        "  db ?\n"
                        "  db.<object> ?\n"
                        "  wiki ?\n"
                        "  wiki.<object> ?\n"
                        "  pile ?\n"
                        "  pile.<object> ?\n"
                        "  email ?"
                        "  email.<object> ?\n"
                    ),
                    user_input=user_input,
                    engine="metadata",
                )

            # ---------------------------------------------
            # Normal DSL parser error
            # ---------------------------------------------
            return render_template(
                "index.html",
                response=parsed["error"],
                user_input=user_input,
                engine="parser",
            )

        # -------------------------------------------------
        # 4) Default return type (if omitted)
        # -------------------------------------------------
        if parsed.get("rtype") is None:
            if parsed["source"] == DataSource.DB:
                parsed["rtype"] = Datatype.TABLE
            elif parsed["source"] == DataSource.EMAIL:
                parsed["rtype"] = Datatype.TABLE
            elif parsed["source"] == DataSource.WIKIPEDIA:
                pass  # Wikipedia decides later (legacy vs attribute search)
            else:
                parsed["rtype"] = Datatype.TEXT

        # -------------------------------------------------
        # 5) Routing
        # -------------------------------------------------
        try:
            src = parsed["source"]

            if src == DataSource.DB:
                response = get_database_answer(DB_PATH, parsed)
                engine = "Database"



            elif src == DataSource.WIKIPEDIA:
                trace = ["WIKI: REQUEST RECEIVED"]
                where_raw = (parsed.get("where_raw") or "").strip()
                wl = where_raw.lower()
                # -------------------------------------------------
                # Wikipedia search + attribute filter
                # ONLY if wiki.<attr> or wikipedia.<attr> is used
                # -------------------------------------------------
                uses_wiki_attr = ("wiki." in wl or "wikipedia." in wl)

                has_logic = any(op in wl for op in (" and ", " or ", " not ", "(", ")"))

                if has_logic and not uses_wiki_attr:
                    response = (
                            "Logical operators (AND / OR / NOT) are only supported for "
                            "Wikipedia attribute filters (wiki.<attribute>).\n\n"
                            "Examples:\n"
                            "  where 'MIT' in wiki.abstract and 'nuno' in wiki.title\n"
                            "  where ('countries' in wiki.categories or 'states' in wiki.categories)\n\n"
                            "If you intended a plain Wikipedia summary filter, use a single condition:\n"
                            "  where economy\n\n"
                            "TRACE:\n"
                            + "\n".join(trace)
                    )
                    engine = "Wikipedia (Error)"
                    return render_template(
                        "index.html",
                        response=response,
                        user_input=user_input,
                        engine=engine,
                    )

                # WIKI-ATTR-MODE ist IMMER aktiv, sobald wiki.<attr> vorkommt
                is_wiki_attr_where = uses_wiki_attr

                if uses_wiki_attr and not has_logic:
                    # NUR atomare WHEREs vorvalidieren
                    if not (" in " in wl or "=" in wl):
                        response = (
                                "Invalid WHERE syntax.\n"
                                "Unsupported operator.\n\n"
                                "Use one of:\n"
                                "  where 'value' in wiki.<attribute>\n"
                                "  where wiki.<attribute> = 'value'\n\n"
                                "TRACE:\n"
                                + "\n".join(trace)
                        )
                        engine = "Wikipedia (Error)"
                        return render_template(
                            "index.html",
                            response=response,
                            user_input=user_input,
                            engine=engine,
                        )

                if is_wiki_attr_where:
                    # -------------------------------------------------
                    # RETURN TYPE VALIDATION (SEARCH MODE ONLY)
                    # -------------------------------------------------
                    # Default return for Wikipedia attribute search
                    if is_wiki_attr_where and parsed.get("rtype") is None:
                        parsed["rtype"] = Datatype.TABLE
                    if parsed.get("rtype") == Datatype.TEXT:
                        response = (
                                "Return type 'text' is not supported for Wikipedia attribute search queries.\n"
                                "Use: return table (default).\n\n"
                                "If you intended a single-page Wikipedia summary, use:\n"
                                "find <title> from wikipedia where <plain filter>\n\n"
                                "TRACE:\n"
                                + "\n".join(trace)
                        )
                        engine = "Wikipedia (Error)"
                        return render_template(
                            "index.html",
                            response=response,
                            user_input=user_input,
                            engine=engine,
                        )
                    try:
                        # -------------------------------------------------
                        # DETERMINE SEARCH QUERY (NOT OUTPUT TARGET!)
                        # -------------------------------------------------
                        raw_target = (parsed.get("find_raw") or "").strip()
                        target_lc = raw_target.lower()

                        # -------------------------------------------------
                        # PARSE FIND ATTRIBUTES (comma-separated)
                        # -------------------------------------------------
                        attrs = [
                            t.strip().lower().split(".", 1)[1]
                            for t in raw_target.split(",")
                            if t.strip().lower().startswith(("wiki.", "wikipedia."))
                        ]

                        # keep only valid schema attributes
                        valid_attrs = [
                            a for a in attrs
                            if a in WIKI_SCHEMA["page"]["attributes"]
                        ]

                        # -------------------------------------------------
                        # DETERMINE SEARCH QUERY
                        # -------------------------------------------------
                        raw_target = (parsed.get("find_raw") or "").strip()

                        find_parts = [p.strip() for p in raw_target.split(",") if p.strip()]

                        find_has_only_wiki_attrs = (
                                find_parts and all(
                            p.lower().startswith(("wiki.", "wikipedia."))
                            for p in find_parts
                            )
                        )

                        if find_has_only_wiki_attrs:
                            # FIND ist reine Projektion → WHERE bestimmt die Suche
                            search_query = extract_search_value_from_where(where_raw)
                            trace.append("WIKI: SEARCH QUERY SOURCE = WHERE")
                        else:
                            # FIND enthält semantischen Suchtext → FIND bestimmt die Suche
                            search_query = raw_target
                            trace.append("WIKI: SEARCH QUERY SOURCE = FIND")

                        # -------------------------------------------------
                        # EXECUTE SEARCH + FILTER
                        # -------------------------------------------------
                        pages = wikipedia_search_and_filter(
                            query=search_query,
                            where_raw=where_raw,
                            trace=trace,
                        )

                        # -------------------------------------------------
                        # OUTPUT PROJECTION
                        # -------------------------------------------------
                        if valid_attrs:
                            headers = valid_attrs
                            rows = []

                            LIST_DISPLAY_LIMIT = 20
                            for p in pages:
                                row = []
                                for a in valid_attrs:
                                    val = resolve_wiki_attribute(p, a, trace)
                                    if isinstance(val, list):
                                        total = len(val)
                                        shown = val[:LIST_DISPLAY_LIMIT]
                                        formatted = "\n".join(map(str, shown))
                                        if total > LIST_DISPLAY_LIMIT:
                                            formatted += f"\n... ({total - LIST_DISPLAY_LIMIT} more)"
                                        val = formatted
                                    row.append(val)
                                rows.append(row)

                        else:
                            # default fallback projection
                            headers = ["title", "pageid", "url"]
                            rows = [
                                [p.title, p.pageid, p.url]
                                for p in pages
                            ]

                        response = (
                                tabulate(rows, headers=headers, tablefmt="psql")
                                + "\n\nTRACE:\n"
                                + "\n".join(trace)
                        )
                        engine = "Wikipedia (Search+Filter)"

                    except ValueError as e:
                        response = (
                                str(e)
                                + "\n\nTRACE:\n"
                                + "\n".join(trace)
                        )
                        engine = "Wikipedia (Error)"
                else:
                    # -------------------------------------------------
                    # LEGACY MODE (unchanged behavior)
                    # -------------------------------------------------
                    response = get_wikipedia_answer(
                        title=parsed.get("find_raw"),
                        columns=None,
                        subsource=parsed.get("subsource"),
                        flt=where_raw,
                        rtype=parsed.get("rtype"),
                    )
                    engine = "Wikipedia"

            elif src == DataSource.PILE:
                response = get_pile_answer(
                    parsed.get("find_raw"),
                    parsed.get("where_raw"),
                    parsed.get("rtype"),
                )
                engine = "The Pile"

            elif src == DataSource.EMAIL:
                response = get_email_answer(parsed)
                engine = "Email"

            else:
                response = "Unknown data source."
                engine = "system"

        except Exception as e:
            response = f"Internal error: {e}"
            engine = "error"


    return render_template(
        "index.html",
        title="Bachelorproject",
        response=response,
        user_input=user_input,
        engine=engine,
    )

# -------------------------------------------------
# Debug helpers
# -------------------------------------------------

@app.route("/debug/tables")
def debug_tables():
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    conn.close()

    return "<pre>" + "\n".join(tables) + "</pre>"


@app.route("/debug/columns/<table>")
def debug_columns(table):
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(f"PRAGMA table_info({table})")
        columns = [r[1] for r in cur.fetchall()]
    except Exception as e:
        columns = [f"Error: {e}"]
    conn.close()

    return "<pre>" + "\n".join(columns) + "</pre>"

# =========================================================
#  Login route & Register route & logout
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(AUTH_DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        )
        user = cur.fetchone()
        conn.close()

        if user and bcrypt.check_password_hash(user[2], password):
            login_user(User(user[0], user[1]))
            login_event = {
                "event": "login",
                "user_id": user[0],
                "username": user[1],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ip": request.remote_addr,
                "user_agent": request.headers.get("User-Agent")
            }

            write_global_log(login_event)
            return redirect("/")

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        if not username or not password:
            return render_template("register.html", error="Username and password required")

        pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")

        conn = sqlite3.connect(AUTH_DB_PATH)
        cur = conn.cursor()

        try:
            # prüfen ob user existiert
            cur.execute("SELECT id FROM users WHERE username = ?", (username,))
            existing = cur.fetchone()

            if existing:
                conn.close()
                return render_template("register.html", error="Username already exists")

            # neuen user einfügen
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, pw_hash),
            )
            conn.commit()

        finally:
            conn.close()

        return redirect("/login")

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_event = {
        "event": "logout",
        "user_id": current_user.id,
        "username": current_user.username,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ip": request.remote_addr
    }

    write_global_log(logout_event)

    logout_user()
    return redirect("/login")


# =========================================================
#  Telemetry endpoint (frontend typing analytics -> backend)
# =========================================================

@app.route("/search-log", methods=["POST"])
@login_required
def search_log():
    data = request.get_json(silent=True)

    if not data and request.data:
        try:
            data = json.loads(request.data.decode("utf-8"))
        except Exception:
            return "", 204

    if not data:
        return "", 204

    user_id = None
    username = None

    if current_user.is_authenticated:
        user_id = current_user.id
        username = current_user.username

    pauses = data.get("typing", {}).get("pauses", [])
    pause_count = len(pauses)

    avg_pause_ms = 0
    if pause_count > 0:
        total_pause = sum(p["duration_ms"] for p in pauses)
        avg_pause_ms = total_pause / pause_count

    clean_entry = {
        "event": "search",
        "user_id": user_id,
        "username": username,

        "session": data.get("session"),

        "query_final": data.get("query_final"),

        "typing": {
            "started_at": data["typing"]["started_at"],
            "ended_at": data["typing"]["ended_at"],
            "active_duration_ms": data["typing"]["active_duration_ms"],
            "total_duration_ms": data["typing"]["total_duration_ms"],
            "pause_count": pause_count,
            "avg_pause_ms": avg_pause_ms
        },

        "submit": data.get("submit"),

        "server_received_at": datetime.now(timezone.utc).isoformat()
    }

    write_global_log(clean_entry)

    if username:
        write_user_log(username, clean_entry)

    return "", 204


if __name__ == "__main__":
    app.run(debug=True)
