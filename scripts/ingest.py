import argparse
import asyncio
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.core.settings import settings
from app.rag.embedder import Embedder
from app.rag.ingestion import DocumentIngestionService


BATCH_SIZE = 32


async def ingest(dry_run: bool = False) -> None:
    data_dir = Path("data")

    ingestion_service = DocumentIngestionService()
    embedder = Embedder()

    chunks = ingestion_service.process_directory(data_dir)

    print(f"Total chunks generated: {len(chunks)}")

    if not chunks:
        print("No PDF files or chunks found.")
        return

    if dry_run:
        print("\nDry run enabled. No embeddings or Qdrant uploads will be performed.")

        for chunk in chunks[:5]:
            print("\n---")
            print(f"Source: {chunk['metadata'].get('source')}")
            print(f"Chunk ID: {chunk['metadata'].get('chunk_id')}")
            print(chunk["text"][:500])

        return

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        embeddings = await embedder.embed_texts(texts)

        points = []

        for chunk, embedding in zip(
            batch,
            embeddings,
            strict=True,
        ):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        **chunk["metadata"],
                    },
                )
            )

        client.upsert(
            collection_name=settings.collection_name,
            points=points,
        )

        print(
            f"Uploaded {min(start + BATCH_SIZE, len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    print("\nIngestion completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process PDFs without generating embeddings or uploading to Qdrant.",
    )

    args = parser.parse_args()

    asyncio.run(
        ingest(
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
