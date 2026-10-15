import hashlib
import re
from collections.abc import Iterable
from itertools import takewhile

from rag.ingest.clean import clean_markdown
from rag.models import Chunk, ChunkMetadata, DocumentInfo, ParsedDocument

PARAGRAPH_SEPARATOR = "\n\n"
HEADING = re.compile(r"^(#{1,6}) (.+)$")
TOP_LEVEL_HEADING = re.compile(r"^#{1,2} ")
SECTION_SEPARATOR = " > "
LINE_SEPARATOR = "\n"
TABLE_SEPARATOR_ROW = re.compile(r"^\|(\s*:?-+:?\s*\|)+$")

Paragraph = tuple[int, tuple[str, ...], str]


def chunk_document(
    doc: ParsedDocument, info: DocumentInfo, max_chars: int = 1500
) -> list[Chunk]:
    paragraphs: list[Paragraph] = [
        (page_number, section, piece)
        for page_number, section, text in with_sections(
            (page.page_number, paragraph)
            for page in doc.pages
            for paragraph in clean_markdown(page.markdown).split(PARAGRAPH_SEPARATOR)
            if paragraph
        )
        for piece in split_oversized(text, max_chars)
    ]

    groups: list[list[Paragraph]] = []
    for paragraph in paragraphs:
        text = paragraph[2]
        if (
            groups
            and group_length(groups[-1]) + len(PARAGRAPH_SEPARATOR) + len(text)
            <= max_chars
            and not starts_new_section(groups[-1], text)
        ):
            groups[-1].append(paragraph)
        else:
            carried = pop_trailing_headings(groups[-1]) if groups else []
            if groups and not groups[-1]:
                groups.pop()
            groups.append([*carried, paragraph])

    return [to_chunk(doc, info, index, group) for index, group in enumerate(groups)]


def with_sections(paragraphs: Iterable[tuple[int, str]]) -> list[Paragraph]:
    headings: list[tuple[int, str]] = []
    result = []
    for page_number, paragraph in paragraphs:
        if match := HEADING.match(paragraph):
            level = len(match.group(1))
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, match.group(2)))
        section = tuple(title for _, title in headings)
        result.append((page_number, section, paragraph))
    return result


def split_oversized(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = text.split(LINE_SEPARATOR)
    header = lines[:2] if is_table(lines) else []
    pieces: list[list[str]] = [[]]
    for line in lines[len(header) :]:
        if (
            pieces[-1]
            and len(LINE_SEPARATOR.join([*header, *pieces[-1], line])) > max_chars
        ):
            pieces.append([])
        pieces[-1].append(line)
    return [LINE_SEPARATOR.join([*header, *piece]) for piece in pieces]


def is_table(lines: list[str]) -> bool:
    return (
        len(lines) >= 2
        and all(line.startswith("|") for line in lines)
        and TABLE_SEPARATOR_ROW.match(lines[1]) is not None
    )


def is_heading(text: str) -> bool:
    return HEADING.match(text) is not None


def starts_new_section(group: list[Paragraph], text: str) -> bool:
    only_headings_so_far = all(is_heading(paragraph[2]) for paragraph in group)
    return TOP_LEVEL_HEADING.match(text) is not None and not only_headings_so_far


def pop_trailing_headings(group: list[Paragraph]) -> list[Paragraph]:
    carried: list[Paragraph] = []
    while group and is_heading(group[-1][2]):
        carried.insert(0, group.pop())
    return carried


def common_section(group: list[Paragraph]) -> str:
    sections = [section for _, section, text in group if not is_heading(text)]
    sections = sections or [group[-1][1]]
    shared = takewhile(lambda titles: len(set(titles)) == 1, zip(*sections))
    return SECTION_SEPARATOR.join(titles[0] for titles in shared)


def group_length(group: list[Paragraph]) -> int:
    return len(PARAGRAPH_SEPARATOR.join(paragraph[2] for paragraph in group))


def to_chunk(
    doc: ParsedDocument,
    info: DocumentInfo,
    index: int,
    group: list[Paragraph],
) -> Chunk:
    return Chunk(
        chunk_id=hashlib.sha256(f"{doc.content_hash}:{index}".encode()).hexdigest(),
        text=PARAGRAPH_SEPARATOR.join(paragraph[2] for paragraph in group),
        metadata=ChunkMetadata(
            doc_id=doc.doc_id,
            file_name=doc.file_name,
            content_hash=doc.content_hash,
            doc_type=info.doc_type,
            fund=info.fund,
            umbrella_fund=info.umbrella_fund,
            section=common_section(group),
            page_start=group[0][0],
            page_end=group[-1][0],
            chunk_index=index,
        ),
    )
