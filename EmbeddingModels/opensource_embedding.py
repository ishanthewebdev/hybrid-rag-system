from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# text="delhi is the capital of india"
# vector=embedding.embed_query(text)
# print(str(vector))
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents=[
    "ravindra Jadeja - He is an Indian cricketer who plays for the national team as an all-rounder. He has played over 200 ODIs and 50 Tests for India and is a left-hand batsman and left-arm orthodox spin bowler.",
    "Mohammed Shami - He is an Indian fast bowler who made his debut in 2013 against the West Indies and has since then established himself firmly in the Indian Test side. He is known for his pace and swing bowling and has taken over 100 wickets in 35 Tests.",
    "chris Gayle - He is a West Indian batsman who played for the national team from 2000 to 2015, famous for his six-hitting abilities and batting average of over 40.",
    "Tim Southee - He is a New Zealand fast bowler known for extracting swing from any surfaces around the world. He has taken over 200 wickets in Tests and T20 cricket and has played an instrumental role in winning the 2015 World Cup for his country."
]

query="tell me about a cricketer"

doc_embedding=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

scores=cosine_similarity([query_embedding],doc_embedding)[0]
# print(list(enumerate(scores)))
print(scores)
print(sorted(list(enumerate(scores)),key=lambda x:x[1]))




#now implemention embedding of mulitiples lines and finding the most accurate from a user query


