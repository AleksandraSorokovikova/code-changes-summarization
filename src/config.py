RETRIEVE_TOP_K=5

DATA_FOLDER_PATH="data"

FAISS_INDEX_PATH=f"{DATA_FOLDER_PATH}/faiss_index.faiss"

CODE_DF_PATH=f"{DATA_FOLDER_PATH}/code_df.csv"

# "alexandraroze/codebert-cross-encoder"
RERANKER_MODEL_NAME="alexandraroze/mixedbread-code-cross-encoder"

EMBEDDINGS_MODEL_NAME="microsoft/graphcodebert-base"

NUM_OF_DOCS_TO_RERANK=10

CHANGE_DEF_TO_SOME_FUNCTION=False

ADD_DOC_TO_CODE=False