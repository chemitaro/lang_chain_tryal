from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader("./docs/001016469.pdf")
data = [loader]

text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=0)
vectorstore = VectorstoreIndexCreator(
    text_splitter=text_splitter
)
index = vectorstore.from_loaders(data)

query = "壁紙の耐用年数は何年ですか？"

print(index.query(query))