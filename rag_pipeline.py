from llm import generate_llm_response


def generate_answer(
    tokenizer,
    model,
    question,
    retrieved_documents
):

    if retrieved_documents:

        context = "\n\n".join(
            document.page_content
            for document, score in retrieved_documents
        )

        messages = [
            {
                "role": "system",
                "content": """
You are RAGBot, a friendly and intelligent AI assistant.

Answer naturally and clearly.

Use the provided company/document information when it
is relevant to the user's question.

If the information is not relevant, answer normally
using your general knowledge.

Never force unrelated document information into an answer.

If the user is just greeting you or having casual
conversation, respond naturally.
"""
            },
            {
                "role": "user",
                "content": f"""
Relevant information:

{context}

User question:

{question}
"""
            }
        ]

    else:

        messages = [
            {
                "role": "system",
                "content": """
You are RAGBot, a friendly and intelligent AI assistant.

Answer the user's question naturally and clearly.

You can answer general questions, explain concepts,
help with programming, have normal conversations,
and assist the user with everyday questions.
"""
            },
            {
                "role": "user",
                "content": question
            }
        ]

    return generate_llm_response(
        tokenizer,
        model,
        messages
    )