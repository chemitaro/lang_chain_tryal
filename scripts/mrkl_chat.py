from langchain import OpenAI, LLMMathChain, SerpAPIWrapper, WikipediaAPIWrapper
from langchain.utilities import BashProcess
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.7)
llm1 = ChatOpenAI(temperature=0)
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm1, verbose=True)
wiki = WikipediaAPIWrapper()
bash = BashProcess()

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
        description="useful for when you need to answer questions involving the generation of bash commands, an LLM can often be employed. A typical scenario is enabling the LLM to interact with your local file system. To facilitate this, we offer a convenient utility for executing bash commands with ease."
    )
]

mrkl = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

output =mrkl.run("/Users/iwasawayuuta/workspace/python/lang_chain_trialのディレクトリに hoge.txt を作成して その中に hoge と書き込んでください")