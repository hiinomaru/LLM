from transformers import pipeline
from chroma_db import db

# retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# local llm
llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",#model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=256
)

query = "Funding opportunities for AI students in Austria"

# retrieve docs
docs = retriever.invoke(query)

# combine context
context = "\n\n".join([doc.page_content for doc in docs])

prompt = f"""
You are a scholarship search assistant.

ONLY use the provided context.

If the answer is not found in the context, say:
"Not found in database."

Do not use outside knowledge.

Context:
{context}

Question:
{query}

Answer:
"""

result = llm(prompt)

print(result[0]["generated_text"])