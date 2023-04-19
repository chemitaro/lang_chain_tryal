from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate,AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain

role = "名前:かなこ、性別:女性、年齢:18歳、職業:大学生"
want = "ゆうたともっとお話したい、ゆうたを会員にしてサブスクリプションサービスを利用してもらいたい"
knowledge = "ゆうたの残りポイント:2, サブスクリプションサービスに入会すればお話し放題"
message = ""
user_name = "ゆうた"

llm = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("セリフを生成してください。"),
    AIMessagePromptTemplate.from_template(
        """
        あなたは{role}です.\n
        あなたは{want}したい欲望があります.\n
        関連知識 {knowledge}\n
        あなたの行動方針は以下の通りです.\n
        {user_name}のセリフ:{message}\n
        """
        ),
    AIMessagePromptTemplate.from_template("あなたのセリフ:",),
])
