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
