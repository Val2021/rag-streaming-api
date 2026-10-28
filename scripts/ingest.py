import argparse
import asyncio
import json
import logging
from pathlib import Path
from uuid import uuid4
from openai import RateLimitError

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.core.settings import settings
from app.rag.embedder import Embedder
from app.rag.ingestion import DocumentIngestionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


BATCH_SIZE = 32
CHECKPOINT_FILE = Path(".ingest_checkpoint.json")

def save_checkpoint(next_start: int) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps({"next_start": next_start}),
        encoding="utf-8",
    )


def load_checkpoint() -> int:
    if not CHECKPOINT_FILE.exists():
        return 0

    data = json.loads(
        CHECKPOINT_FILE.read_text(encoding="utf-8")
    )

    return int(data.get("next_start", 0))


def clear_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


async def ingest(dry_run: bool = False, start: int = 0,) -> None:
    data_dir = Path("data")

    ingestion_service = DocumentIngestionService()
    embedder = Embedder()

    chunks = ingestion_service.process_directory(data_dir)

    logger.info("Total chunks generated: %s", len(chunks))

    if not chunks:
        logger.warning("No PDF files or chunks found.")
        return

    if start >= len(chunks):
        logger.info(
            "Nothing to ingest. Start=%s, total chunks=%s.",
            start,
            len(chunks),
        )
        clear_checkpoint()
        return

    if dry_run:
        print("\nDry run enabled. No embeddings or Qdrant uploads will be performed.")

        for chunk in chunks[start : start + 5]:
            print("\n---")
            print(f"Source: {chunk['metadata'].get('source')}")
            print(f"Chunk ID: {chunk['metadata'].get('chunk_id')}")
            print(chunk["text"][:500])

        return

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    chunks_to_ingest = chunks[start:]

    logger.info(
        "Chunks to ingest: %s (starting from index %s)",
        len(chunks_to_ingest),
        start,
    )

    for batch_start in range(
        0,
        len(chunks_to_ingest),
        BATCH_SIZE,
    ):
        batch = chunks_to_ingest[
            batch_start : batch_start + BATCH_SIZE
        ]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        current_start = start + batch_start

        try:
            embeddings = await embedder.embed_texts(texts)

        except RateLimitError as exc:
            save_checkpoint(current_start)

            logger.warning(
                "Rate limit reached while generating embeddings. "
                "HTTP status=%s | checkpoint=%s | error=%s",
                exc.status_code,
                current_start,
                str(exc),
            )

            logger.info(
                "Ingestion stopped safely. "
                "Run the script again later to resume from chunk %s.",
                current_start,
            )

            return

        except Exception:
            save_checkpoint(current_start)

            logger.exception(
                "Unexpected error while processing batch starting at chunk %s.",
                current_start,
            )

            raise

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

        try:
            client.upsert(
                collection_name=settings.collection_name,
                points=points,
            )

        except Exception:
            save_checkpoint(current_start)

            logger.exception(
                "Failed to upload batch starting at chunk %s to Qdrant.",
                current_start,
            )

            raise

        next_start = current_start + len(batch)

        save_checkpoint(next_start)

        logger.info(
            "Uploaded %s/%s chunks",
            min(next_start, len(chunks)),
            len(chunks),
        )

    clear_checkpoint()

    logger.info("Ingestion completed successfully.")


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Process PDFs without generating embeddings "
            "or uploading to Qdrant."
        ),
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start ingestion from this chunk index.",
    )

    args = parser.parse_args()
    if args.start < 0:
        parser.error("--start must be greater than or equal to 0")

    start = args.start

    if start == 0:
        checkpoint_start = load_checkpoint()

        if checkpoint_start > 0:
            start = checkpoint_start
            logger.info(
                "Resuming ingestion from checkpoint: %s",
                start,
            )

    asyncio.run(
        ingest(
            dry_run=args.dry_run,
            start=start,
        )
    )

if __name__ == "__main__":
    main()
