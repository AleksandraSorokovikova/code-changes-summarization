import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from rerankers import Reranker
from tqdm import tqdm
import torch.nn.functional as F


def mean_pooling(model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask[..., None].float()
    token_embeddings = model_output[0] * input_mask_expanded
    summed_embeddings = token_embeddings.sum(dim=1)
    mask_sums = input_mask_expanded.sum(dim=1)
    sentence_embeddings = summed_embeddings / mask_sums.clamp(min=1e-9)
    return sentence_embeddings


"""
for mixedbread-code-cross-encoder use
candidates = [
    re.sub(r'def \w+\(.*\)', 'def some_function(...)', candidate) for candidate in candidates
]
"""


class RAG:
    def __init__(
            self,
            faiss_index_path: str,
            code_df_path: str,
            emb_model_name: str = "microsoft/graphcodebert-base",
            reranker_model_name: str = "alexandraroze/codebert-cross-encoder",
            device: str = "cpu",
            num_of_docs_to_rerank: int = 20
    ):
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.code_df = pd.read_csv(code_df_path)
        self.reranker = Reranker(
            model_type="cross-encoder",
            model_name=reranker_model_name,
        )
        self.model = AutoModel.from_pretrained(emb_model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
        self.model.eval()
        self.device = device
        self.num_of_docs_to_rerank = num_of_docs_to_rerank

    def get_embedding(self, code_snippet: str) -> np.ndarray:
        with torch.no_grad():
            encoded_input = self.tokenizer(
                code_snippet, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
            model_output = self.model(**encoded_input)
            embedding = mean_pooling(model_output, encoded_input['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1).cpu().numpy()
        return embedding

    def rerank(self, query: str, relevant_rows: list[str], doc_ids: list[int]) -> list:
        results = self.reranker.rank(query, relevant_rows, doc_ids=doc_ids)
        return [result.doc_id for result in results]

    def search(self, code_snippet: str, top_k: int = 5) -> tuple[str, str] | tuple[list[str], list[str]]:
        query_embedding = self.get_embedding(code_snippet)
        distances, indices = self.faiss_index.search(query_embedding, self.num_of_docs_to_rerank)
        relevant_rows = self.code_df.iloc[indices[0]]["code"].tolist()
        relevant_ids = self.rerank(code_snippet, relevant_rows, doc_ids=indices[0])
        relevant_documentations = self.code_df.iloc[relevant_ids]["documentation"].tolist()[:top_k]
        relevant_codes = self.code_df.iloc[relevant_ids]["code"].tolist()[:top_k]

        if top_k == 1:
            return relevant_documentations[0], relevant_codes[0]

        return relevant_documentations, relevant_codes


def build_rag(
        code_snippets: list[str],
        documentations: list[str],
        code_df_path: str,
        faiss_index_path: str,
        model_name="microsoft/graphcodebert-base",
        batch_size=32
):
    code_df = pd.DataFrame({
        "index": range(len(code_snippets)),
        "code": code_snippets,
        "documentation": documentations
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.max_seq_length = 512
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(code_snippets), batch_size)):
            try:
                batch = code_snippets[i:i + batch_size]
                encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
                model_output = model(**encoded_input)
                batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)
            except Exception as e:
                print(e)
                del encoded_input, model_output, batch_embeddings
                torch.cuda.empty_cache()
                embeddings.append(np.zeros((len(batch), 768)))
                model.to(device)
                continue

    embeddings = np.vstack(embeddings)

    zero_indices = np.where(~embeddings.any(axis=1))[0]
    print(f"Found {len(zero_indices)} zero vectors")
    if zero_indices > 0:
        code_df = code_df.drop(zero_indices)
        embeddings = np.delete(embeddings, zero_indices, axis=0)
        print(len(code_df), embeddings.shape)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, faiss_index_path)
    code_df.to_csv(code_df_path, index=False)
