def build_context(state):

    context = "\n\n".join(
        document.page_content
        for document in state["documents"]
    )

    return {
        "context": context,
    }