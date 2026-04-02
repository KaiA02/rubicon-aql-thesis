from openai import OpenAI
from enums import Datatype
from Configurations.config_loader import OPENAI_API_KEY
import textwrap
import json
import re

# =========================================================
# OPENAI CLIENT
# =========================================================

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================================================
# PROMPTS
# =========================================================

BASE_PROMPT = """You are a retrieval assistant.

Search term: "{term}"
Filter/context: "{flt}"
Output type: {out_type}

Rules:
- Keep the answer neutral and factual.
- Do NOT invent links.
"""

TEXT_PROMPT_SUFFIX = """
Return 2–4 sentences max.
No bullet points.
No greetings.
"""

TABLE_PROMPT_SUFFIX = """
Return valid JSON only.

- You may decide the best structure.
- Be consistent in the structure.
- Do not invent links.
- Do not add explanations outside JSON.
"""

# =========================================================
# HELPERS
# =========================================================

def clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("|", "/")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def render_block(row: dict, width: int = 80) -> str:
    """
    Dynamically render any JSON object as terminal block.
    """
    lines = []

    for key, value in row.items():
        value = clean(str(value))
        wrapped = textwrap.wrap(value, width=width)

        if wrapped:
            lines.append(f"| {key:<10} | {wrapped[0]}")
            for line in wrapped[1:]:
                lines.append(f"| {'':<10} | {line}")
        else:
            lines.append(f"| {key:<10} |")

    return "\n".join(lines)

# =========================================================
# MAIN ENTRYPOINT
# =========================================================

def get_pile_answer(term: str, flt: str, rtype: Datatype | None):
    trace = ["PILE: REQUEST RECEIVED"]

    if rtype is None:
        rtype = Datatype.TEXT
        trace.append("PILE: DEFAULT RETURN TYPE = TEXT")

    flt = (flt or "").strip()

    # -----------------------------------------------------
    # PROMPT SELECTION
    # -----------------------------------------------------

    if rtype == Datatype.TABLE:
        out_type = "TABLE"
        system_prompt = (
            BASE_PROMPT.format(term=term, flt=flt, out_type="TABLE")
            + TABLE_PROMPT_SUFFIX
        )
    else:
        out_type = "TEXT"
        system_prompt = (
            BASE_PROMPT.format(term=term, flt=flt, out_type="TEXT")
            + TEXT_PROMPT_SUFFIX
        )

    trace.append(f"PILE: MODE = {out_type}")

    user_prompt = (
        f'Give me the {out_type.lower()} for "{term}".'
        if not flt
        else f'Give me the {out_type.lower()} for "{term}" filtered by "{flt}".'
    )

    # -----------------------------------------------------
    # LLM CALL
    # -----------------------------------------------------

    try:
        if rtype == Datatype.TABLE:

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = resp.choices[0].message.content.strip()
            data = json.loads(content)

            if isinstance(data, dict):

                if len(data) == 1 and isinstance(list(data.values())[0], list):
                    data = list(data.values())[0]
                else:
                    data = [data]

            blocks = []
            for row in data:
                if isinstance(row, dict):
                    blocks.append(render_block(row))
                else:
                    blocks.append(render_block({"value": row}))

            separator = "\n" + "-" * 60 + "\n"
            result = separator.join(blocks)

            return result + "\n\nTRACE:\n" + "\n".join(trace)

        # -------------------------------------------------
        # TEXT OUTPUT
        # -------------------------------------------------

        else:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = resp.choices[0].message.content.strip()
            formatted = textwrap.fill(content, width=100)

            return formatted + "\n\nTRACE:\n" + "\n".join(trace)

    except Exception as e:
        return (
            f"ERROR: {repr(e)}\n\nTRACE:\n"
            + "\n".join(trace)
        )
