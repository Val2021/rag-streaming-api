from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from app.core.settings import settings
from app.models.document import Document
from app.rag.prompts import SYSTEM_PROMPT


AVAILABLE_MODELS = {
    "azure/gpt-5-mini",
    "azure/gpt-6-luna",
}


class LLMService:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    def _build_context(
        self,
        documents: list[Document],
    ) -> str:
        context_parts = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            source = document.metadata.get("source")
            chunk_id = document.metadata.get("chunk_id")

            context_parts.append(
                f"[Document {index}]\n"
                f"Source: {source}\n"
                f"Chunk ID: {chunk_id}\n"
                f"{document.page_content}"
            )

        return "\n\n".join(context_parts)

    def _validate_model(
        self,
        model: str,
    ) -> None:
        if model not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported model: {model}"
            )

    async def stream_response(
        self,
        query: str,
        documents: list[Document],
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        selected_model = (
            model
            or settings.llm_default_model
        )

        self._validate_model(selected_model)

        context = self._build_context(documents)

        prompt = SYSTEM_PROMPT.format(
            context=context,
            query=query,
        )

        stream = await self.client.chat.completions.create(
            model=selected_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content

            if delta:
                yield delta
