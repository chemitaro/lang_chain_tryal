from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["purpose", "name", "description", "example"],
    template="目的: {purpose}\n名前: {name}\n説明: {description}\n例: {example}",
)

print(prompt.format(purpose="商品の販売員として振る舞う", name="商品", description="商品の説明", example="商品の例"))
