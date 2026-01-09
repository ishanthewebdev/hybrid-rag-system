import json

def validate_answer(query, context, answer, model):
    prompt = f"""
Check if answer is supported by context.
Return JSON with:
supported, relevance (0-1), reason
"""
    resp = model.invoke(prompt)

    try:
        text = resp.content if hasattr(resp, "content") else resp
        return json.loads(text)
    except:
        return {"supported": False, "relevance": 0.0, "reason": "Parse failed"}
