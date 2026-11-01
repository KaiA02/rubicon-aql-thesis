import configparser

config = configparser.ConfigParser()
config.read('Configurations/config.ini')

OPENAI_API_KEY = config["OPENAI"]["API_KEY"]

WIKI_USER_AGENT_NAME = config.get(
    "WIKIPEDIA",
    "USER_AGENT_NAME",
    fallback="KaiBachelorProject/1.0"
)

WIKI_CONTACT_EMAIL = config.get(
    "WIKIPEDIA",
    "CONTACT_EMAIL",
    fallback=""
)

FLASK_SECRET_KEY = config.get(
    "FLASK",
    "SECRET_KEY",
    fallback=None
)
