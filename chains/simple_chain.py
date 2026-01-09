from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# load_dotenv()
load_dotenv(dotenv_path=r"C:\Users\ishan\Desktop\langchain campusx\.env")

print("Token loaded:", os.getenv("HUGGINGFACEHUB_API_TOKEN")[:10] + "...")
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # temperature=0.1
    )

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)



parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()