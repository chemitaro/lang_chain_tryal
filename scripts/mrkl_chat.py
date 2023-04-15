from langchain import OpenAI, LLMMathChain, SerpAPIWrapper, WikipediaAPIWrapper
from langchain.utilities import BashProcess, BingSearchAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.7)
llm1 = ChatOpenAI(temperature=0)
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm1, verbose=True)
wiki = WikipediaAPIWrapper()
bash = BashProcess()
bing = BingSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="useful for when you need to answer questions about history, science, and other topics"
    ),
    Tool(
        name="Bash",
        func=bash.run,
        description="It is used to generate and execute bash commands. A typical scenario is to allow the LLM to interact with the local file system. For this purpose, we provide a convenient utility to easily execute bash commands."
    ),
    Tool(
        name="Bing",
        func=bing.run,
        description="useful for when you need to answer questions about current events. You can ask questions in natural language."
    )
]

mrkl = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

output =mrkl.run("今日の重要なニュースを教えてください。")