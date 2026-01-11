# # def answer_with_rag(query, context, model):
# #     if not context or len(context) < 50:
# #         return "I don't know based on the provided document."

# #     prompt = f"""
# # Use ONLY the context.

# # CONTEXT:
# # {context}

# # QUESTION:
# # {query}

# # ANSWER:
# # """
# #     resp = model.invoke(prompt)
# #     return resp.content if hasattr(resp, "content") else resp
# def answer_with_context(query: str, context: str, model):
#     if not context or len(context.strip()) < 50:
#         return "I don't know based on the provided document."

#     prompt = f"""
# Use ONLY the information in the CONTEXT to answer the QUESTION.
# If the answer is not in the context, say:
# "I don't know based on the provided document."

# CONTEXT:
# {context}

# QUESTION:
# {query}

# ANSWER:
# """
#     response = model.invoke(prompt)
#     return response
from llm.model import get_llm

_llm = get_llm()   # loaded ONCE

def answer_with_context(query: str, context: str):
    if not context or len(context.strip()) < 50:
        return "I don't know based on the provided document."

    prompt = f"""
Use ONLY the information in the CONTEXT to answer the QUESTION.
If the answer is not in the context, say:
"I don't know based on the provided document."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = _llm.invoke(prompt)
    return response.content.strip()
