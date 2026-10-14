from pydantic import BaseModel, ConfigDict, Field


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
