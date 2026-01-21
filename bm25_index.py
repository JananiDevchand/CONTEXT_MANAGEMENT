import re
from rank_bm25 import BM25Okapi

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self):
        self.docs = []        # tokenized documents
        self.ids = []         # episode IDs (same order as docs)
        self.bm25 = None

    def add(self, doc_id, text: str):
        """
        Add a document to the BM25 index.

        doc_id : episode_id (or any unique ID)
        text   : full episode text
        """
        tokens = tokenize(text)
        if not tokens:
            return

        self.docs.append(tokens)
        self.ids.append(doc_id)
        self.bm25 = BM25Okapi(self.docs)

    def search(self, query: str):
        """
        Returns a dict:
        {
            episode_id: bm25_score,
            ...
        }
        """
        if not self.bm25:
            return {}

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        return {
            self.ids[i]: float(scores[i])
            for i in range(len(scores))
        }
