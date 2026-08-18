from langchain_huggingface import HuggingFaceEmbeddings

from app.config.settings import settings


embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
)