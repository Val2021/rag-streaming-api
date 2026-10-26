from openai import AsyncOpenAI

from app.core.settings import settings


class Embedder:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.embedding_model

    async def embed_text(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding

    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [
            item.embedding
            for item in response.data
        ]
