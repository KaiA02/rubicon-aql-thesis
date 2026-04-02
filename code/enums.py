from enum import Enum, auto

class DataSource(Enum):
    DB = "db"
    WIKIPEDIA = "wikipedia"
    EMAIL = "email"
    PILE = "pile"

class Datatype(Enum):
    TEXT = "text"
    TABLE = "table"
    INTERFACE = "interface"
    PICTURE = "picture"

