from langgraph.graph import Graph
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import os
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Any
from langchain_community.tools import TavilySearchResults

# Load environment variables
load_dotenv()


class BlogState(BaseModel):
    topic: Any = Field(default="")
    content: Any = Field(default="")
    outline: Any = Field(default="")
    blog_post: Any = Field(default="")
    fetched_data: Any = Field(default="")


def get_llm():
    # Ensure the model name and setup are valid for ChatGroq
    return ChatGroq(model="llama3-8b-8192")


def outline_blog(state: BlogState):
    llm = get_llm()
    outline_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that outlines blog posts."),
            ("user", "Outline a blog post about {topic}.")
        ]
    )
    chain = outline_prompt | llm
    result = chain.invoke({"topic": state.topic})
    state.outline = result.content  # Update the state with the returned outline
    # print("Output from outlinte: ", state.outline)
    return state


def content_generator(state: BlogState):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert AI at creating blog content based on the given outline and the data fetched from the tavily "),
            ("user", "Outline: {outline}"),
            ("user", "Topic: {topic}"),
            ("user", "Fetched_data: {fetched_data}")
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"outline": state.outline, "topic": state.topic, "fetched_data": state.fetched_data})
    state.content = result.content  # Update the state with the returned content
    # print("Output from content: ", state.content)
    return state


def formatter_agent(state: BlogState):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert AI at formatting blog content. Content will be provided."),
            ("user", "Content: {content}"),
            ("user", "Topic: {topic}")
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"content": state.content, "topic": state.topic})
    state.blog_post = result.content  # Update the state with the formatted blog post
    # print("Output from format: ", state.blog_post)
    return state

# def tool_for_content(state: BlogState):
#     llm = get_llm()
#     tools = [TavilySearchResults(max_results=5, search_depth="advanced")]
#     llm = llm.bind_tools(tools)
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", """You are an expert AI that based on the outline and topic given. It will use the search tool given to fetch data from the internet.
#               The tool given you to is used to search the internet {agent_scratchpad}"""),
#             ("user", "Topic: {topic}"),
#             ("user", "Outline: {outline}"),
#         ]
#     )
#     agent = create_tool_calling_agent(llm, tools, prompt)

#     agent_executor = AgentExecutor(
#         agent=agent,
#         tools=tools,
#         verbose=True
#     )
    
#     state.fetched_data = agent_executor.invoke({"topic": state.topic, "outline": state.outline})
#     return state

def tool_for_content(state: BlogState):
    tool = TavilySearchResults(max_results=5, search_depth="advanced")
    state.fetched_data = tool.invoke({"query": state.topic})
    print("Output from tool:", state.fetched_data)
    return state


workflow = StateGraph(BlogState)

workflow.add_node("outline_blog", outline_blog)
workflow.add_node("content_generator", content_generator)
workflow.add_node("formatter_agent", formatter_agent)
workflow.add_node("tool_node", tool_for_content)

workflow.add_edge(START, "outline_blog")
workflow.add_edge("outline_blog", "tool_node")
workflow.add_edge("tool_node", "content_generator")
workflow.add_edge("content_generator", "formatter_agent")
workflow.add_edge("formatter_agent", END)

app = workflow.compile()

state_obj = BlogState()
state_obj.topic = "What is today's weather in pune?"


print(app.invoke(state_obj)["content"])

# AgentState
# Class TypeDict, Messages
# Custom States

# llm with bind tools
# agentExecutor