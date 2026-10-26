from typing import Any

from pydantic import BaseModel

##This model will represent the chunks retrieved from Qdrant, which are then passed as context to the LLM.
class Document(BaseModel):
    page_content: str
    metadata: dict[str, Any] | None = None
