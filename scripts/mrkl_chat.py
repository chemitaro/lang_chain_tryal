from langchain import OpenAI, LLMMathChain, SerpAPIWrapper, WikipediaAPIWrapper
from langchain.utilities import BashProcess, BingSearchAPIWrapper, PythonREPL, WolframAlphaAPIWrapper
from langchain.tools.human.tool import HumanInputRun
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
llm1 = ChatOpenAI(temperature=0)
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm1, verbose=True)
wiki = WikipediaAPIWrapper()
bash = BashProcess()
bing = BingSearchAPIWrapper(k=3)
human = HumanInputRun()
python_repl = PythonREPL()
wolfram = WolframAlphaAPIWrapper()

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
    ),
    Tool(
        name="human",
        func=human.run,
        description="You can ask a human for guidance when you think you got stuck or you are not sure what to do next. The input should be a question for the human."
    ),
    Tool(
        name="python",
        func=python_repl.run,
        description="It is used to execute python code. A typical scenario is to allow the LLM to interact with the local file system. For this purpose, we provide a convenient utility to easily execute python code."
    ),
    Tool(
        name="wolfram",
        func=wolfram.run,
        description="Useful when you need answers about numerical, physical, or statistical calculations. WolframAlpha is a computational knowledge engine that delivers accurate and relevant results in a variety of domains. Analyze, process, and visualize data in real time."
    )
]

mrkl = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

output =mrkl.run("直径50cmの球体を初速度100m/sで45度の角度で発射した際の最大高度は？")