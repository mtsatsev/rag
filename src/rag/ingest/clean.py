import re

PICTURE_MARKER = re.compile(r"<!-- (Start|End) of picture text -->")
LINE_BREAK = re.compile(r"<br>")
UNDERLINE = re.compile(r"</?u>")
BOLD = re.compile(r"\*\*(.+?)\*\*")
ITALIC = re.compile(r"(?<!\w)_(.+?)_(?!\w)")


def clean_markdown(text: str) -> str:
    text = PICTURE_MARKER.sub("", text)
    text = LINE_BREAK.sub(" ", text)
    text = UNDERLINE.sub("", text)
    text = BOLD.sub(r"\1", text)
    return ITALIC.sub(r"\1", text)
