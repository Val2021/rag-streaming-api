from qdrant_client import QdrantClient

from app.core.settings import settings
from app.models.document import Document
from app.rag.embedder import Embedder


class QdrantRetriever:
    def __init__(self) -> None:
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        self.embedder = Embedder()
        self.collection_name = settings.collection_name

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[Document]:
        top_k = limit or settings.retrieval_top_k

        query_embedding = await self.embedder.embed_text(query)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        documents = []

        for point in response.points:
            payload = point.payload or {}

            documents.append(
                Document(
                    page_content=payload.get("text", ""),
                    metadata={
                        "score": point.score,
                        "source": payload.get("source"),
                        "chunk_id": payload.get("chunk_id"),
                        "headings": payload.get("headings"),
                        "page_numbers": payload.get("page_numbers"),
                    },
                )
            )

        return documents
