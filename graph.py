from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agents.supervisor import supervisor_node
from agents.researcher import researcher_node
from agents.critic import critic_node
from agents.executor import executor_node
from state import AgentState

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)
workflow.add_node("executor", executor_node)

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.next,
    {"researcher": "researcher", "critic": "critic", "executor": "executor", "__end__": END},
)

workflow.add_edge("researcher", "supervisor")
workflow.add_edge("critic", "supervisor")
workflow.add_edge("executor", "supervisor")
workflow.add_edge(START, "supervisor")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
