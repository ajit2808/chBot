import anthropic
from app.config import settings

client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

def generate_answer(context, question):
    prompt = f"""
You are an enterprise RAG assistant.

Answer ONLY using the provided context.
If the answer is not found in the context, say:
"Information not found in uploaded documents."

Context:
{context}

Question:
{question}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.content[0].text