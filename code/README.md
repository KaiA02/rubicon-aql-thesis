# RUBICON – Implementation
 
Flask web application implementing the AQL query interface.
 
## Requirements
 
- Python 3.10+
- OpenAI API key
- Internet access (the embedding model `all-MiniLM-L6-v2` is downloaded on first start)
## Setup
 
1. Install dependencies (from inside `code/`):
```
   pip install -r requirements.txt
```
 
2. Create `Configurations/config.ini`:
```ini
   [OPENAI]
   API_KEY = sk-...
 
   [FLASK]
   SECRET_KEY = your-random-secret
 
   [WIKIPEDIA]
   USER_AGENT_NAME = RubiconAQL/1.0
   CONTACT_EMAIL = you@example.com
```
 
   Generate a secret key with:
 
```
   python -c "import secrets; print(secrets.token_hex(32))"
```
 
   The `[FLASK]` and `[WIKIPEDIA]` sections are optional. Without a secret key, a random one is generated on each start, which logs out all users after a restart.
 
3. *Optional – email queries:* Place a Google OAuth client secret (type "Desktop app", Gmail read-only scope) at `Configurations/client_secret.json`. On the first email query a browser window opens for sign-in, and a `token.json` is stored.
4. *Optional – database queries:* Place the BEAVER data warehouse MySQL dump at `beaver/new_dw_indexed.sql` and convert it:
```
   python beaver/convert_mysql_dump_to_sqlite.py
```
 
   This creates `beaver/dw.sqlite`.
 
## Run
 
From inside `code/`:
 
```
python app.py
```
 
Open http://127.0.0.1:5000/register, create an account and log in.
 
API keys, credentials and the BEAVER dataset are not included in this repository.
