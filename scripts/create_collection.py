from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.settings import settings


def main() -> None:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    if client.collection_exists(
        collection_name=settings.collection_name
    ):
        print(
            f"Collection '{settings.collection_name}' already exists. Deleting..."
        )

        client.delete_collection(
            collection_name=settings.collection_name
        )

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(
            size=settings.embedding_dimension,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Collection '{settings.collection_name}' created successfully."
    )


if __name__ == "__main__":
    main()
