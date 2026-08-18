from langchain_openai import ChatOpenAI

from app.config.settings import settings


llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0,
)


def generate_answer(state):

    prompt = f"""
You are a helpful assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context,
say that you don't have enough information.

Question:
{state["query"]}

Context:
{state["context"]}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
    }