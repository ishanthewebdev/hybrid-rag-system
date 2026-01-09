# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from dotenv import load_dotenv
# import os
# from fpdf import FPDF



# load_dotenv()


# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     temperature=0.3,
# )

# model = ChatHuggingFace(llm=llm)


# chat_template = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful expert in {domain} who explains the given topic in simple terms under 50 words."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{user_input}")
# ])


# chat_history = []

# domain = input("Enter the domain: ")
# topic = input("Enter the topic: ")


# initial_prompt = chat_template.invoke({
#     "domain": domain,
#     "user_input": f"Explain what is {topic} in simple terms.",
#     "chat_history": chat_history
# })
# initial_response = model.invoke(initial_prompt)

# print("\nAI:", initial_response.content)
# chat_history.append(HumanMessage(content=f"Explain what is {topic}"))
# chat_history.append(AIMessage(content=initial_response.content))

# print("\n--- Chat started (type 'exit' to stop) ---\n")
# def export_chat_to_pdf(chat_history, filename="chat_history.pdf"):
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)

#     for msg in chat_history:
#         role = "User" if msg.type == "human" else "AI"
#         text = f"{role}: {msg.content}\n"
#         pdf.multi_cell(0, 10, text)

#     pdf.output(filename)
#     print(f"\n✅ Chat saved as PDF: {filename}")


# while True:
#     user_input = input("You: ")
#     # if user_input.lower() in ["exit", "quit"]:
#     #     print("Chat ended.")
#     #     break
#     if user_input.lower() in ["exit", "quit"]:
#       filename = input("Enter a name for your PDF (e.g., physics_notes.pdf): ")
#       export_chat_to_pdf(chat_history, filename)
#       print("Chat ended.")
#       break

  
#     prompt = chat_template.invoke({
#         "domain": domain,
#         "user_input": user_input,
#         "chat_history": chat_history
#     })

    
#     result = model.invoke(prompt)
#     print("AI:", result.content)

    
#     chat_history.append(HumanMessage(content=user_input))
#     chat_history.append(AIMessage(content=result.content))


# print("\nFinal Chat History:")
# for msg in chat_history:
#     print(f"{msg.type.upper()}: {msg.content}")


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from fpdf import FPDF



load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature=0.3,
)

model = ChatHuggingFace(llm=llm)

prompt=ChatPromptTemplate.from_messages([
    ('system',"you are a helpful and knowledgeble assitant, who can define {domain} in less than 50 words efficently "),
    MessagesPlaceholder(variable_name="chat_history"),
    ('user','{user_input}')

]  
)

chat_history=[]

domain=input("Enter the domain : ")
topic=input("Enter the topic : ")

# intial_prompt=prompt.invoke(
#     "domain":domain,
#     "user":f"Explain what is {topic} in simple terms.",
#     "chat_history":chat_history

# )

chain=prompt|model
result=chain.invoke({'domain':domain,'user_input':topic,'chat_history':chat_history})

chat_history.append(HumanMessage(content=f"Explain what is {topic}"))
chat_history.append(AIMessage(content=result.content))


print("\n--- Chat started (type 'exit' to stop) ---\n")

def chat_to_pdf(chat_history,file_name="chat_history.pdf"):
    pdf=FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for msg in chat_history:
        role = "User" if msg.type == "human" else "AI"
        text = f"{role}: {msg.content}\n"
        pdf.multi_cell(0, 10, text)

    pdf.output(file_name)
    print(f"\n✅ Chat saved as PDF: {file_name}")


while True:
    user_input=input("Do you have any question regarding this topics? :")
    if user_input.lower() in ["exit", "quit"]:
      file_name = input("Enter a name for your PDF (e.g., physics_notes.pdf): ")
      chat_to_pdf(chat_history, file_name)
      print("Chat ended.")
      break

    result1=result=chain.invoke({'domain':domain,'user_input':user_input,'chat_history':chat_history})
    print("AI:", result.content)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result.content))

print("\nFinal Chat History:")
for msg in chat_history:
    print(f"{msg.type.upper()}: {msg.content}")    








