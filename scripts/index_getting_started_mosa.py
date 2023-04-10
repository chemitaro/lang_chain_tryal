from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter

loader = TextLoader('./docs/mosasaurus.txt')
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=0)
vectorstore = VectorstoreIndexCreator(
    text_splitter=text_splitter
)
index = vectorstore.from_loaders([loader])

query = "モササウルスの体の特徴は何ですか？"
print(index.query(query))