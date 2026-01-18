import json
from config import EMBED_MODEL

# def validate_answer(query, context, answer):
#     prompt = f"""
# Check if answer is supported by context.
# Return JSON with:
# supported, relevance (0-1), reason
# """
#     resp = EMBED_MODEL.(prompt)

#     try:
#         text = resp.content if hasattr(resp, "content") else resp
#         return json.loads(text)
#     except:
#         return {"supported": False, "relevance": 0.0, "reason": "Parse failed"}
def validate_answer(query, context, answer, llm):
    prompt = f"""
You are a strict RAG answer judge.

USER QUESTION:
{query}

CONTEXT:
{context}

MODEL ANSWER:
{answer}

Check:
- Is the answer supported by the context?
- Does it answer the question?

Return ONLY valid JSON:
{{
  "supported": true or false,
  "relevance": number between 0 and 1,
  "reason": "short reason"
}}
"""
    try:
        resp = llm.invoke(prompt)
        text = resp.content.strip()

        data = json.loads(text)

        return {
            "supported": bool(data.get("supported", False)),
            "relevance": float(data.get("relevance", 0.0))
        }

    except Exception:
        # 🔒 HARD FAIL SAFE
        return {
            "supported": False,
            "relevance": 0.0
        }
    # resp = llm.invoke(prompt)

    # try:
    #     return json.loads(resp.content)
    # except Exception:
    #     return {
    #         "supported": False,
    #         "relevance": 0.0,
    #         "reason": "Validator failed to parse response"
    #     }
