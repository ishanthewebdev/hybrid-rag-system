def answer_with_rag(query, context, model):
    if not context or len(context) < 50:
        return "I don't know based on the provided document."

    prompt = f"""
Use ONLY the context.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
    resp = model.invoke(prompt)
    return resp.content if hasattr(resp, "content") else resp
