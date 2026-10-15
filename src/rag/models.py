from typing import Literal, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class ParsedPage(BaseModel):
    page_number: int = Field(
        ge=1, description="1-based page number in the original PDF"
    )
    markdown: str = Field(description="Markdown content of the page")

    model_config = ConfigDict(frozen=True, validate_assignment=True)


class ParsedDocument(BaseModel):
    doc_id: str = Field(description="File stem, e.g. '11074'")
    file_name: str = Field(description="Original file name")
    content_hash: str = Field(description="sha256 of the file bytes")
    pages: list[ParsedPage] = Field(description="List of parsed pages in the document")

    model_config = ConfigDict(
        frozen=True, validate_assignment=True, arbitrary_types_allowed=True
    )


DocType = Literal["prospectus", "kid", "sfdr", "factsheet", "investor_rights"]


class DocumentInfo(BaseModel):
    doc_type: DocType = Field(description="Document type")
    fund: str | None = Field(description="Fund name, if applicable")
    umbrella_fund: str | None = Field(description="Umbrella fund name, if applicable")

    model_config = ConfigDict(frozen=True, validate_assignment=True)


class Umbrella(BaseModel):
    name: str = Field(description="Canonical umbrella name, stored on chunks")
    aliases: list[str] = Field(default=[], description="Other ways users write it")

    model_config = ConfigDict(frozen=True, validate_assignment=True)


class Fund(BaseModel):
    name: str = Field(description="Canonical sub-fund name, stored on chunks")
    umbrella: str = Field(description="Canonical name of the umbrella it belongs to")
    isin: str | None = Field(default=None, description="ISIN code, if known")
    aliases: list[str] = Field(default=[], description="Other ways users write it")

    model_config = ConfigDict(frozen=True, validate_assignment=True)


class DocumentEntry(BaseModel):
    doc_type: DocType = Field(description="Document type")
    fund: str | None = Field(
        default=None, description="Canonical fund name; None if not fund-specific"
    )
    umbrella: str | None = Field(
        default=None,
        description="Umbrella for umbrella-level documents; derived from fund otherwise",
    )
    multi_fund: bool = Field(
        default=False, description="True if the fund changes inside the document"
    )

    model_config = ConfigDict(frozen=True, validate_assignment=True)


class FundRegistry(BaseModel):
    umbrellas: list[Umbrella] = Field(description="Known umbrella funds")
    funds: list[Fund] = Field(description="Known sub-funds")
    documents: dict[str, DocumentEntry] = Field(description="File name -> entry")

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        umbrellas = {umbrella.name for umbrella in self.umbrellas}
        funds = {fund.name for fund in self.funds}
        for fund in self.funds:
            if fund.umbrella not in umbrellas:
                raise ValueError(
                    f"Fund {fund.name!r} has unknown umbrella {fund.umbrella!r}"
                )
        for file_name, entry in self.documents.items():
            if entry.fund is not None and entry.fund not in funds:
                raise ValueError(
                    f"Document {file_name!r} has unknown fund {entry.fund!r}"
                )
            if entry.umbrella is not None and entry.umbrella not in umbrellas:
                raise ValueError(
                    f"Document {file_name!r} has unknown umbrella {entry.umbrella!r}"
                )
        return self


class ChunkMetadata(BaseModel):
    doc_id: str = Field(description="File stem, e.g. '11074'")
    file_name: str = Field(description="Original file name")
    content_hash: str = Field(
        description="sha256 of the source file; used to detect replaced documents"
    )
    doc_type: DocType = Field(description="Document type")
    fund: str | None = Field(description="Fund name, if applicable")
    umbrella_fund: str | None = Field(description="Umbrella fund name, if applicable")
    section: str = Field(description="Section name")
    page_start: int = Field(ge=1, description="Starting page number")
    page_end: int = Field(ge=1, description="Ending page number")
    chunk_index: int = Field(ge=0, description="Index of the chunk within the document")

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError(
                f"page_start ({self.page_start}) cannot be greater than page_end ({self.page_end})"
            )
        return self


class Chunk(BaseModel):
    chunk_id: str = Field(
        description="Hash of content_hash and chunk_index, stable when the same file is re-ingested"
    )
    text: str = Field(description="Cleaned text, shown to the LLM and cited")
    metadata: ChunkMetadata = Field(description="Metadata associated with the chunk")

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    @computed_field
    @property
    def embed_text(self) -> str:
        header = " | ".join(
            part for part in (self.metadata.fund, self.metadata.section) if part
        )
        return f"{header}\n\n{self.text}" if header else self.text


class Paragraph(NamedTuple):
    page_number: int
    section: tuple[str, ...]
    text: str
    fund: str | None = None
