from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.search import router as search_router
from app.core.settings import settings


app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
)

app.include_router(search_router)
app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
    }
