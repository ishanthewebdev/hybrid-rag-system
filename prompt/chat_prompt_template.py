# from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
# from langchain_core.messages import SystemMessage, AIMessage,HumanMessage
# from langchain_huggingface import HuggingFaceEmbeddings ,ChatHuggingFace,HuggingFaceEndpoint
# import streamlit as st
# from dotenv import load_dotenv
# #
# from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
# load_dotenv()
# llm=HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     # temperature=0.1
#     )
# model = ChatHuggingFace(llm=llm)

# chat_template=ChatPromptTemplate([
#     ('system',"Your are a helpful expert in {domain} which defines the given topic in 50 words"),
#     MessagesPlaceholder(variable_name='chat_history')
#     ('human',"Explain in simple terms, what is {topic}") #this menthod is espcially for chatprompttemplate
#     # SystemMessage(content="Your are a helpful expert in {domain} which defines the given topic in 50 words"),
#     # HumanMessage(content="Explain in simple terms, what is {topic}")

# ])
# chat_history=[
#     # SystemMessage(content="you are a helpful assitant which gived answer in less than 50 words"),
#     # HumanMessage(content=user_input),
# ]

# user_input_domain=input("tell us the domain  ")
# user_input_topic=input(" tell us the topic  ")
# prompt= chat_template.invoke({'domain':user_input_domain,'topic':user_input_topic})
# while True:
#     # user_input=input("you: ")
#     prompt= chat_template.invoke({'domain':user_input_domain,'topic':user_input_topic})

#     chat_history.append(HumanMessage(content=user_input_domain))
#     chat_history.append(HumanMessage(content=user_input_topic))

#     if user_input_domain=='exit':
#         break
#     result= model.invoke(chat_template)
#     chat_history.append(AIMessage(content=result.content))
#     print("AI: ",result.content)

# print(chat_history)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from fpdf import FPDF


# Load .env
load_dotenv()

# Initialize model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature=0.3,
)

model = ChatHuggingFace(llm=llm)

# Prompt template (System + chat history + human input)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful expert in {domain} who explains the given topic in simple terms under 50 words."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# Initialize chat history
chat_history = []

# Take domain and topic once
domain = input("Enter the domain: ")
topic = input("Enter the topic: ")

# Generate first explanation automatically
initial_prompt = chat_template.invoke({
    "domain": domain,
    "user_input": f"Explain what is {topic} in simple terms.",
    "chat_history": chat_history
})
initial_response = model.invoke(initial_prompt)

print("\nAI:", initial_response.content)
chat_history.append(HumanMessage(content=f"Explain what is {topic}"))
chat_history.append(AIMessage(content=initial_response.content))

# Start interactive chatbot
print("\n--- Chat started (type 'exit' to stop) ---\n")
def export_chat_to_pdf(chat_history, filename="chat_history.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for msg in chat_history:
        role = "User" if msg.type == "human" else "AI"
        text = f"{role}: {msg.content}\n"
        pdf.multi_cell(0, 10, text)

    pdf.output(filename)
    print(f"\n✅ Chat saved as PDF: {filename}")


while True:
    user_input = input("You: ")
    # if user_input.lower() in ["exit", "quit"]:
    #     print("Chat ended.")
    #     break
    if user_input.lower() in ["exit", "quit"]:
      filename = input("Enter a name for your PDF (e.g., physics_notes.pdf): ")
      export_chat_to_pdf(chat_history, filename)
      print("Chat ended.")
      break

    # Build prompt dynamically
    prompt = chat_template.invoke({
        "domain": domain,
        "user_input": user_input,
        "chat_history": chat_history
    })

    # Get model response
    result = model.invoke(prompt)
    print("AI:", result.content)

    # Update chat history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result.content))

# Print final chat history for debugging
print("\nFinal Chat History:")
for msg in chat_history:
    print(f"{msg.type.upper()}: {msg.content}")

