from langgraph.graph import StateGraph, START, END

from app.graph.state import RAGState
from app.graph.nodes.retrieve import retrieve
from app.graph.nodes.context import build_context
from app.graph.nodes.generation import generate_answer


def build_graph():

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("build_context", build_context)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "build_context")
    graph.add_edge("build_context", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


rag_graph = build_graph()