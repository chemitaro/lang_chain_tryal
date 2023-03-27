from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)

output = chat([HumanMessage(content="こんにちは")])
print(output)

messages = [
    SystemMessage(content="You are a helpful assistant that users questions"),
    HumanMessage(content="りんごは動物ですか？植物ですか？")
]
output_2 = chat(messages)
print(output_2)


batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
print(result)
print(result.llm_output['token_usage'])