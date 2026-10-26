from pydantic import BaseModel, Field

from app.models.document import Document


class ChatRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Question to answer using the knowledge base.",
    )

    model: str = Field(
        default="azure/gpt-5-mini",
        description="LLM model used to generate the response.",
    )

    limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of retrieved chunks used as context.",
    )


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Question used to retrieve relevant documents.",
    )

    limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of retrieved chunks.",
    )


class SearchResponse(BaseModel):
    results: list[Document]


class ModelsResponse(BaseModel):
    models: list[str]
    default: str
