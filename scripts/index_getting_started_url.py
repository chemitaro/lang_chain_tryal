from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter

urls = ['https://ja.wikipedia.org/wiki/%E3%83%A2%E3%82%B5%E3%82%B5%E3%82%A6%E3%83%AB%E3%82%B9']
loader = UnstructuredURLLoader(urls=urls)
data = [loader]  # 'load()'が返す'Document'オブジェクトをリストに格納
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=0)
vectorstore = VectorstoreIndexCreator(
    text_splitter=text_splitter
)
index = vectorstore.from_loaders(data)

query = "モササウルスの体の特徴は何ですか？"
print(index.query(query))
