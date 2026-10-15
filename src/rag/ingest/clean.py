import re

PICTURE_MARKER = re.compile(r"<!-- (Start|End) of picture text -->")
LINE_BREAK = re.compile(r"<br>")
UNDERLINE = re.compile(r"</?u>")
BOLD = re.compile(r"\*\*(.+?)\*\*")
ITALIC = re.compile(r"(?<!\w)_(.+?)_(?!\w)")
CYRILLIC_WORD = re.compile(r"\w*[\u0400-\u04FF]\w*")
SPACES = re.compile(r"[ \u00a0]+")
TRAILING_SPACES = re.compile(r" +$", re.MULTILINE)
BLANK_LINES = re.compile(r"\n{3,}")
LATIN_TO_CYRILLIC = str.maketrans("ABCEHKMOPTXaceopxy", "АВСЕНКМОРТХасеорху")


def clean_markdown(text: str) -> str:
    text = PICTURE_MARKER.sub("", text)
    text = LINE_BREAK.sub(" ", text)
    text = UNDERLINE.sub("", text)
    text = BOLD.sub(r"\1", text)
    text = ITALIC.sub(r"\1", text)
    text = CYRILLIC_WORD.sub(
        lambda word: word.group().translate(LATIN_TO_CYRILLIC), text
    )
    text = SPACES.sub(" ", text)
    text = TRAILING_SPACES.sub("", text)
    text = BLANK_LINES.sub("\n\n", text)
    return text.strip()


def match_key(text: str) -> str:
    raise NotImplementedError
