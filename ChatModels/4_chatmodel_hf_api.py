from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
# load_dotenv()
load_dotenv(dotenv_path=r"C:\Users\ishan\Desktop\langchain campusx\.env")

print("Token loaded:", os.getenv("HUGGINGFACEHUB_API_TOKEN")[:10] + "...")
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # temperature=0.1
    )

model = ChatHuggingFace(llm=llm)

result=model.invoke("write down info about five random cricketers")
print(result.content)
