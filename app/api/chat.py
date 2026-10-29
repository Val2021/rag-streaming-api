import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.api import ChatRequest, ModelsResponse
from app.rag.retriever import QdrantRetriever
from app.services.llm_service import LLMService, AVAILABLE_MODELS
from app.core.settings import settings



router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

retriever = QdrantRetriever()
llm_service = LLMService()

@router.get(
    "/models",
    response_model=ModelsResponse,
)
async def models() -> ModelsResponse:
    return ModelsResponse(
        models=sorted(AVAILABLE_MODELS),
        default=settings.llm_default_model,
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
) -> StreamingResponse:
    try:
        documents = await retriever.retrieve(
            query=request.query,
            limit=request.limit,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve relevant documents.",
        ) from exc

    async def event_generator():
        try:
            sources = [
                {
                    "source": document.metadata.get("source"),
                    "chunk_id": document.metadata.get("chunk_id"),
                    "score": document.metadata.get("score"),
                    "headings": document.metadata.get("headings"),
                }
                for document in documents
            ]

            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "sources",
                        "sources": sources,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

            async for delta in llm_service.stream_response(
                query=request.query,
                documents=documents,
                model=request.model,
            ):
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "text_delta",
                            "delta": delta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                    }
                )
                + "\n\n"
            )

        except ValueError as exc:
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "message": str(exc),
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        except Exception:
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "message": "Failed to generate the response.",
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
