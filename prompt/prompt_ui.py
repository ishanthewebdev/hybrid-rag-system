from langchain_huggingface import HuggingFaceEmbeddings ,ChatHuggingFace,HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.header("research tool")
# model=
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # temperature=0.1
    )
model = ChatHuggingFace(llm=llm)
# user_input=st.text_input("enter your prompt")

# if st.button('summarize'):
#     result=model.invoke(user_input)
#     st.write(result.content)

# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()
# model = ChatOpenAI()

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=['paper_input', 'style_input','length_input'] #variables define kar rhe what we are going to pass 
)




if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input, #prompts me desired value push kar rhe hai yaha se input variable the through
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)
      
