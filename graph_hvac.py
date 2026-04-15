from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents.hvac.supervisor import supervisor_node
from agents.hvac.diagnostician import diagnostician_node
from agents.hvac.safety_critic import safety_critic_node
from agents.hvac.executor import executor_node
from state import AgentState

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("diagnostician", diagnostician_node)
workflow.add_node("safety_critic", safety_critic_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.get("next", "diagnostician"),   # fallback nếu next chưa có
    {
        "diagnostician": "diagnostician",
        "safety_critic": "safety_critic",
        "executor": "executor",
        "__end__": END,
    }
)

workflow.add_edge("diagnostician", "supervisor")
workflow.add_edge("safety_critic", "supervisor")
workflow.add_edge("executor", "supervisor")

hvac_app = workflow.compile(checkpointer=MemorySaver())

print("✅ HVAC Troubleshooting Graph built successfully!")