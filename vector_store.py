import faiss
import numpy as np


class VectorStore:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.documents = []
        self.index = None

    def add_documents(self, documents):

        self.documents = documents

        texts = [
            document.page_content
            for document in documents
        ]

        embeddings = self.embedding_model.embed_documents(
            texts
        )

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(embeddings)

    def search(self, query, k=3):

        query_embedding = self.embedding_model.embed_query(
            query
        )

        query_embedding = np.array(
            [query_embedding],
            dtype="float32"
        )

        distances, indices = self.index.search(
            query_embedding,
            min(k, len(self.documents))
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            if index != -1:
                results.append(
                    (
                        self.documents[index],
                        float(distance)
                    )
                )

        return results