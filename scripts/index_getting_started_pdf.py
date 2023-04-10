from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA, RetrievalQA
from langchain.chat_models import ChatOpenAI

paths = [
    "./docs/第１章_原状回復にかかるガイドライン.pdf",
    "./docs/第２章_トラブルの迅速な解決にかかる制度.pdf",
    "./docs/Ｑ＆Ａ.pdf",
    "./docs/第３章_原状回復にかかる判例の動向.pdf",
    "./docs/＜参考資料＞.pdf"
    ]
texts = []

for path in paths:
    loader = PyPDFLoader(path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = texts + (text_splitter.split_documents(documents))

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings)

chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever()
    )

query = "壁紙の耐用年数は何年ですか？"

answer = chain.run(query)

print(answer)