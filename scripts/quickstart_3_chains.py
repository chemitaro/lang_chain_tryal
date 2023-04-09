from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["product"],
    template="Whatis a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

out_put = chain.run("colorful socks")

print(out_put)
