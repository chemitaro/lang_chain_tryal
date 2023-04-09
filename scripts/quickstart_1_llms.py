from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)

text = "こんにちは"
print(llm(text))
