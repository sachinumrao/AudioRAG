import json
from typing import List

from nltk import sent_tokenize


def split_doc(text: str, method="sliding") -> List[str]:
    pass


def get_effective_token_length(text: str) -> int:
    tokenization_factor = 1.3
    return len(text.split()) * tokenization_factor


class DocumentSplitter:
    def __init__(self):
        pass

    def _sent_tokenize(self, text: str) -> List[str]:
        sents = sent_tokenize(text)
        return sents

    def get_split_docs(
        self, text: str, chunk_size=256, method="sliding", overlap_sents=1
    ):
        """create chunks of the document"""

        # convert text data to list of sentences
        sentences = self._sent_tokenize(text)

        # chunks list items into groups of chunk size length atmost
        chunk_id = 0
        chunk_bucket = []
        bucket_len = 0
        chunk_data = {}

        curr_sent_id = 0
        while curr_sent_id < len(sentences):
            curr_sent = sentences[curr_sent_id]
            n = get_effective_token_length(curr_sent)
            if bucket_len + n < chunk_size:
                bucket_len += n
                chunk_bucket.append(curr_sent)
                curr_sent_id += 1

            else:
                chunk_data[f"chunk_{chunk_id}"] = " ".join(chunk_bucket).strip()
                chunk_id += 1
                chunk_bucket = []
                bucket_len = 0
                curr_sent_id -= 1

        return chunk_data


if __name__ == "__main__":
    # load test doc
    test_file = "./../transcripts/State_of_GPT_|_BRK216HFS_base_transcription.txt"
    with open(test_file, "r") as f:
        test_data = f.read()

    # create document
    ds = DocumentSplitter()
    doc_chunks = ds.get_split_docs(test_data)

    # evlauate output of splitting
    for i in range(5):
        print(f"Chunk-{i}")
        print(doc_chunks[f"chunk_{i}"])
        print("-" * 80)

    output_filename = "./../chunk_databases/karpathy_state_of_ai_chunks.json"
    with open(output_filename, "w") as f:
        json.dump(doc_chunks, f)
