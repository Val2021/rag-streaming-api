from fastapi import APIRouter

from app.models.api import SearchRequest, SearchResponse
from app.rag.retriever import QdrantRetriever


router = APIRouter(
    prefix="/search",
    tags=["search"],
)

retriever = QdrantRetriever()


@router.post(
    "",
    response_model=SearchResponse,
)
async def search(
    request: SearchRequest,
) -> SearchResponse:
    documents = await retriever.retrieve(
        query=request.query,
        limit=request.limit,
    )

    return SearchResponse(
        results=documents,
    )
