import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from rerankers import Reranker


class RAG:
    def __init__(self, faiss_index_path, code_df):
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.code_df = code_df
        self.reranker = Reranker(
            model_type="cross-encoder",
            model_name="microsoft/codebert-base"
        )

    def search(self, query_embedding, rerank=True, k=10):
        distances, indices = self.faiss_index.search(query_embedding, k)
        relevant_rows = self.code_df.iloc[indices[0]]
        if rerank:
            relevant_rows = self.rerank(query_embedding, relevant_rows, k)
        return relevant_rows

    def rerank(self, query, relevant_rows, k=10):
        results = self.reranker.rank(query, relevant_rows, doc_ids=[i for i in range(len(relevant_rows))])
        return [result.text for result in results]



def build_rag(
        code_snippets, documentations, code_df_path, faiss_index_path,
        model_name="microsoft/codebert-base", batch_size=32
):
    code_df = pd.DataFrame({
        "index": range(len(code_snippets)),
        "code": code_snippets,
        "documentation": documentations
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(code_snippets), batch_size):
            batch = code_snippets[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, faiss_index_path)
    code_df.to_csv(code_df_path, index=False)
