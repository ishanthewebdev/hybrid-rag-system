from sentence_transformers import util
from config import EMBED_MODEL


def compress_context(query, chunks, embed_model, query_embedding,
                     max_chars=1200, top_sentences=2):

    final_sentences = []

    for chunk in chunks:
        sentences = [s.strip() for s in chunk.split(". ") if len(s.strip()) > 20]
        if not sentences:
            continue

        sent_embs = embed_model.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        scores = util.cos_sim(query_embedding, sent_embs)[0]
        top_ids = scores.argsort(descending=True)[:top_sentences]

        for idx in top_ids:
            final_sentences.append(sentences[int(idx)])

        if sum(len(s) for s in final_sentences) > max_chars:
            break

    return ". ".join(final_sentences)
