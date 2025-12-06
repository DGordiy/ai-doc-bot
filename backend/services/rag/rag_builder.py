from backend.services.utils import truncate_text


def build_rag_context(results: list[dict], max_total_chars: int = 3000) -> str:
    context_parts = []
    total_chars = 0

    for r in results:
        text = truncate_text(r["text"], 600)
        meta = r["metadata"]
        chunk_info = (
            f"File: {meta['file']} | chunk {meta['chunk_index'] + 1}/{meta['total_chunks']}"
        )

        block = f"{chunk_info}\n{text}"
        block_len = len(block)

        if total_chars + block_len > max_total_chars:
            break

        context_parts.append(block)
        total_chars += block_len

    return "\n\n---\n\n".join(context_parts)
