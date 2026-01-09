from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # temperature=0.1
    )
model = ChatHuggingFace(llm=llm)

loader=TextLoader('cricket.txt',encoding='utf-8') #every loader give ouput in the form of list
docs=loader.load()
print(docs[0])
 