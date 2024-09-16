from typing_extensions import TypedDict
import random
from typing import Literal
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
import os, getpass
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from pprint import pprint
from langgraph.checkpoint.memory import MemorySaver

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

"""
Load in a chat model.
"""
llm = ChatOpenAI(model = 'gpt-4o-mini')

"""
Build Tools.
"""

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [multiply, add, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)


"""
1. Define the state. The state schema serves as the input schema for 
all nodes and edges.
"""
class State(TypedDict):
    graph_state: str

class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    pass

sys_message = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs")

"""
2. Nodes - python functions.
1st positional arg is the state.
Each node returns a new value of the state key.
"""

def tool_calling_llm(state: MessagesState):
    return{"messages": [llm_with_tools.invoke(state["messages"])]}

def assistant(state: MessagesState):
    return{"messages": [llm_with_tools.invoke([sys_message] + state["messages"])]}
    
"""
 3. Edges connect the nodes.
 Normal edges - to always go from a to b.
 Conditional edges - optionally route.
"""


"""
4. Construct the graph.
initialise graph with the state. Then add nodes and edges.
Start Node and End Node sends user input to the graph and
represents a terminal node respectively.
"""
memory = MemorySaver()
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))


builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
graph_memory = builder.compile(checkpointer=memory)

"""
Graph Invocation.
A compiled graph implements the runnable protocol.
invoke is the standard methods for this interface.
When using memory, need to specify a thread_id.
"""
config = {"configurable": {"thread_id": "1"}}

messages = graph_memory.invoke({"messages": ("user", "Add 3 and 4")}, config=config)
for m in messages["messages"]:
    m.pretty_print()

messages = graph_memory.invoke({"messages": ("user", "then multiply by 7")}, config=config)
for m in messages["messages"]:
    m.pretty_print()
