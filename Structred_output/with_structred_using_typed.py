from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict , Annotated

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # temperature=0.1
    )
model = ChatHuggingFace(llm=llm)

#schema

class Review(TypedDict):
    summary:str
    sentiment:str 
    #here is a example of annotated brefieng
    # summary:Annotated[str," A brief summary of the review"]
    # sentiment:Annotated[str,"return sentiment of the review either negative,postive or neutral"]
    #this helps to get complete output


structred_model= model.with_structured_output(Review)

result = structred_model.invoke("""  the hardware is great, but the software feels quite a bit laggy, it has a great UI but the slow user expirence. Hoping a software update can fix it , it has too many pre- installed apps that i cant remove
""")

print(result)
## sometimes LLM dosent get meanign of summary word and may not return anything like ye jo llm user kar rhe isme vo format nhi kar paaarha hai so we have to give
## Annotaed that means to specify what llm have to do clearly
