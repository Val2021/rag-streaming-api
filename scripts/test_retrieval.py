import asyncio

from app.rag.retriever import QdrantRetriever


async def main() -> None:
    retriever = QdrantRetriever()

    query = (
        "Quais habilidades de comunicação são importantes "
        "no cuidado em saúde mental?"
    )

    documents = await retriever.retrieve(
        query=query,
        limit=5,
    )

    print(f"\nQuery: {query}")
    print(f"Retrieved documents: {len(documents)}")

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print("\n" + "=" * 80)
        print(f"RESULT {index}")

        print(
            f"Score: {document.metadata.get('score')}"
        )

        print(
            f"Source: {document.metadata.get('source')}"
        )

        print(
            f"Chunk ID: {document.metadata.get('chunk_id')}"
        )

        print(
            f"Headings: {document.metadata.get('headings')}"
        )

        print("\nText:")
        print(document.page_content[:1000])


if __name__ == "__main__":
    asyncio.run(main())
