from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    api_title: str = "RAG Streaming API"
    api_description: str = (
        "FastAPI chatbot with RAG and streaming responses"
    )
    api_version: str = "0.1.0"

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str | None = None
    collection_name: str = "mental_health_knowledge_base"

    # Embeddings
    embedding_model: str = "cohere/embed-v4.0"

    # Retrieval
    retrieval_top_k: int = 5

    # LLM provider
    llm_base_url: str
    llm_api_key: str

    # Available models
    llm_default_model: str = "azure/gpt-5-mini"

    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }


settings = Settings()
