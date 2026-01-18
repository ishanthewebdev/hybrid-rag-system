import json

def rewrite_query(query, model):
    prompt = f"""
Rewrite the query clearly for search.
- More descriptive
- Expand abbreviations
- Include ML terminology
Return ONLY JSON:
{{"rewritten": "..."}} 

 original Query: {query}
"""
    resp = model.invoke(prompt)

    try:
        text = resp.content if hasattr(resp, "content") else resp
        data = json.loads(text[text.find("{"):text.rfind("}")+1])
        return data.get("rewritten", query)
    except:
        return query
