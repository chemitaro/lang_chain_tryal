from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter

loader = WebBaseLoader("https://python.langchain.com/")
data = [loader]

text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=0)
vectorstore = VectorstoreIndexCreator(
    text_splitter=text_splitter
)
index = vectorstore.from_loaders(data)

query = "Langchainの使い方を教えて"
print(index.query(query))
