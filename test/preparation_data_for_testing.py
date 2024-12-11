import pandas as pd
from src.RAG import RAG
from src.config import *

test = RAG(
    faiss_index_path=FAISS_INDEX_PATH,
    code_df_path=CODE_DF_PATH,
    reranker_model_name=RERANKER_MODEL_NAME,
    change_def_to_some_function=CHANGE_DEF_TO_SOME_FUNCTION,
    emb_model_name=EMBEDDINGS_MODEL_NAME,
    num_of_docs_to_rerank=NUM_OF_DOCS_TO_RERANK,
    add_doc_to_code=ADD_DOC_TO_CODE
)

df = pd.read_csv('shaffled_data_5000_only_user_example.csv')

test_small_dataset = pd.DataFrame(columns=['user_code', 'similar_code', 'similar_documentation'])

for i in df['code'][:TEST_ITERATIONS]:
    similar_documentation, similar_code = test.search(i, top_k=RETRIEVE_TOP_K)
    test_small_dataset.loc[len(test_small_dataset)] = [i, similar_code[0], similar_documentation[0]]


# Created a dataset. User_code, similar code from RAG, similar documentation from RAG
test_small_dataset.to_csv('1000_user_similardoc_similarcode.csv', index=False)
