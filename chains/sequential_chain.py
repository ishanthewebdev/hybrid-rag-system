from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # temperature=0.1
    )

model = ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Give me a detailed report on this{topic}",
    input_variables=['topic']

)
prompt2=PromptTemplate(
    template="Give me a 5 pointer short summary of the following \n{text}",
    input_variables=['text']
)

parser=StrOutputParser()

chain=prompt1|model|parser|prompt2|model|parser

result=chain.invoke({'topic':'unemployment in India'})

print(result)