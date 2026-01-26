from config import MAX_CONTEXT_CHARS

# def build_context(chunks):
#     context, total = [], 0

#     for ch in chunks:
#         if total + len(ch) > MAX_CONTEXT_CHARS:
#             break
#         context.append(ch.strip())
#         total += len(ch)

#     return "\n\n---\n\n".join(context)

# def build_context(chunks):
#     ctx, total = [], 0
#     for c in chunks:
#         if total + len(c) > MAX_CONTEXT_CHARS:
#             break
#         ctx.append(c)
#         total += len(c)
#     return "\n\n---\n\n".join(ctx)

def build_context(chunks: list[dict]) -> str:
    blocks = []

    for c in chunks:
        block = f"""
SOURCE: {c.get("source", "unknown")}
PAGE: {c.get("page", "N/A")}
CONTENT:
{c["text"]}
"""
        blocks.append(block.strip())

    return "\n\n---\n\n".join(blocks)
