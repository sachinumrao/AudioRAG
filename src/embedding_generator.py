import json
from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")
        pass

    def get_embedding(self, texts: List[str]):
        return self.model.encode(texts)


if __name__ == "__main__":
    # load embedding generator
    eg = EmbeddingGenerator()

    # load chunk data
    chunk_fname = "./../chunk_databases/karpathy_state_of_ai_chunks.json"
    with open(chunk_fname, "r") as f:
        chunk_data = json.load(f)

    embd = eg.get_embedding([chunk_data["chunk_1"]])
    print("Embedding Shape:", embd.shape)
