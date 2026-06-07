import json
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

embeddings = np.load(
    "vector_db/embeddings.npy"
)

with open(
    "vector_db/metadata.json",
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)


def search(query):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    q_embedding = response.data[0].embedding

    similarities = cosine_similarity(
        [q_embedding],
        embeddings
    )[0]

    idx = np.argmax(similarities)

    return metadata[idx]


def ask(question):

    context = search(question)

    prompt = f"""
    Kamu adalah chatbot sekolah.

    Informasi:
    {context['answer']}

    Pertanyaan:
    {question}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content