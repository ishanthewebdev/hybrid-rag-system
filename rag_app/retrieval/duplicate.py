
def duplicate_chunks(chunks:list[dict],max_similar=0.9):
    seen=set()
    unique=[]
    for chunk in chunks:
        text=chunk["text"].strip()
        key=text[:100]

        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique        