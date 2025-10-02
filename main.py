from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()  # Load environment variables from .env file

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# llm = ChatOpenAI(model="gpt-40-mini")
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# response = llm.invoke("What is the capital of France?")
# print(response)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a research assistant that will help genetare a research report on a given topic.
        You will provide a summary of your findings, list the sources you used, and mention any tools you utilized.
        Ensure that your response is in the format provided and do not deviate from it. \n{format_instructions}
        After generating the research report, you must call the tool save_tool to save the report.
        """
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(agent = agent, tools = tools, verbose=True) 
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})
# print(raw_response)
# print("\n")
try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print("Error Parsing response:", e, "Raw Response - ", raw_response)






