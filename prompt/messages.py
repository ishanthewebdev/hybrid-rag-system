from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
#
load_dotenv()

# st.header("research tool")
# model=
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # temperature=0.1
    )
model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="you are a helpful assitant"),
    HumanMessage(content="tell me about langchain")
]

result= model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)