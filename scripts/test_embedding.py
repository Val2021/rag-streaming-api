import asyncio

from app.rag.embedder import Embedder


async def main() -> None:
    embedder = Embedder()

    text = "A saúde mental faz parte do cuidado integral na atenção primária."

    embedding = await embedder.embed_text(text)

    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")


if __name__ == "__main__":
    asyncio.run(main())
