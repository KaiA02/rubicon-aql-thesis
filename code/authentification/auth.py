import sqlite3
from flask_login import LoginManager, UserMixin
from flask_bcrypt import Bcrypt
import os
import secrets
from Configurations.config_loader import FLASK_SECRET_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_DB_PATH = os.path.join(BASE_DIR, "auth.sqlite")

login_manager = LoginManager()
bcrypt = Bcrypt()


def init_auth(app):
    app.secret_key = FLASK_SECRET_KEY or secrets.token_hex(32)

    login_manager.init_app(app)
    login_manager.login_view = "login"
    bcrypt.init_app(app)

    init_auth_db()


def init_auth_db():
    conn = sqlite3.connect(AUTH_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username


@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(AUTH_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()

    if row:
        return User(row[0], row[1])
    return None
