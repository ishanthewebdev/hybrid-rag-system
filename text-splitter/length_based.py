from langchain_text_splitters import CharacterTextSplitter

text=" hey hpw are you i am fine thanku"


splitter=CharacterTextSplitter(
    chunk_size=5,
    chunk_overlap=0,
    separator=''
)

result=splitter.split_text(text)
print(result[0])

