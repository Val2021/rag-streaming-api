from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from transformers import AutoTokenizer


class DocumentIngestionService:
    def __init__(
        self,
        tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 500,
    ) -> None:
        self.converter = DocumentConverter()

        huggingface_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model
        )

        tokenizer = HuggingFaceTokenizer(
            tokenizer=huggingface_tokenizer,
            max_tokens=max_tokens,
        )

        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
        )

    def load_document(self, file_path: Path):
        result = self.converter.convert(
            source=file_path
        )

        return result.document

    def create_chunks(
        self,
        document,
        source_name: str,
    ) -> list[dict[str, Any]]:
        chunks = []

        for index, chunk in enumerate(
            self.chunker.chunk(dl_doc=document)
        ):
            metadata = {
                "chunk_id": index,
                "source": source_name,
            }

            if hasattr(chunk, "meta"):
                if hasattr(chunk.meta, "headings"):
                    metadata["headings"] = (
                        chunk.meta.headings
                    )

                if hasattr(chunk.meta, "page_numbers"):
                    metadata["page_numbers"] = (
                        chunk.meta.page_numbers
                    )

            chunks.append(
                {
                    "text": chunk.text,
                    "metadata": metadata,
                }
            )

        return chunks

    def process_file(
        self,
        file_path: Path,
    ) -> list[dict[str, Any]]:
        document = self.load_document(
            file_path
        )

        return self.create_chunks(
            document=document,
            source_name=file_path.name,
        )

    def process_directory(
        self,
        directory: Path,
    ) -> list[dict[str, Any]]:
        all_chunks = []

        pdf_files = sorted(
            directory.glob("*.pdf")
        )

        for pdf_file in pdf_files:
            chunks = self.process_file(
                pdf_file
            )

            all_chunks.extend(chunks)

        return all_chunks
