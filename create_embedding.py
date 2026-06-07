import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

with open("data/sekolah_faq.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

vectors = []

for item in docs:

    text = f"""
    Pertanyaan:
    {item['question']}

    Jawaban:
    {item['answer']}
    """

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    vectors.append(response.data[0].embedding)

np.save(
    "vector_db/embeddings.npy",
    np.array(vectors)
)

with open(
    "vector_db/metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(docs, f, ensure_ascii=False)

print("Embedding selesai")