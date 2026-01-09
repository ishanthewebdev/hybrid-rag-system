from langchain_huggingface import HuggingFaceEmbeddings ,ChatHuggingFace,HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv
#
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
load_dotenv()

# st.header("research tool")
# model=
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # temperature=0.1
    )
model = ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage(content="you are a helpful assitant which gived answer in less than 50 words"),
    # HumanMessage(content=user_input),
]
while True:
    user_input=input("you: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    result= model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)

#now we have messages jiske thorough jab hum chat history maintain karte hai so wo define kar deta hai ki konsa messg user ne bhja and konsa messg AI ne reply kia
