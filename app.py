import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import streamlit as st
import difflib
from src.RAG import RAG
from src.config import *
from src.generate_documentation import generate_documentation, generate_documentation_diff_description

st.set_page_config(page_title="Code summarization", layout="wide")

def load_file(uploaded_file):
    if uploaded_file is not None:
        return uploaded_file.read().decode("utf-8").splitlines()
    return None


def load_text(text_input):
    if text_input:
        return text_input.splitlines()
    return None


def highlight_changes(old_code, new_code):
    diff = difflib.unified_diff(old_code, new_code, lineterm='', n=0)
    highlighted_code = "\n".join(line for line in diff)
    return highlighted_code


if "rag" not in st.session_state:
    st.session_state.rag = RAG(
        faiss_index_path=FAISS_INDEX_PATH,
        code_df_path=CODE_DF_PATH,
        reranker_model_name=RERANKER_MODEL_NAME,
        change_def_to_some_function=CHANGE_DEF_TO_SOME_FUNCTION,
        emb_model_name=EMBEDDINGS_MODEL_NAME,
        num_of_docs_to_rerank=NUM_OF_DOCS_TO_RERANK,
        add_doc_to_code=ADD_DOC_TO_CODE
    )
if "previous_code" not in st.session_state:
    st.session_state.previous_code = None
if "current_code" not in st.session_state:
    st.session_state.current_code = None
if "code_text" not in st.session_state:
    st.session_state.code_text = ""

st.title("Code summarization")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Submit code")

    uploaded_file = st.file_uploader("Upload file with code", type=["py", "txt"])
    code_text = st.text_area("Or write code in the text field", value=st.session_state.code_text)

    if st.button("Submit code"):
        if st.session_state.current_code:
            st.session_state.previous_code = st.session_state.current_code
        if uploaded_file:
            st.session_state.current_code = load_file(uploaded_file)
        elif code_text:
            st.session_state.current_code = load_text(code_text)
            st.session_state.code_text = ""

    if st.session_state.current_code:
        st.subheader("New version")
        st.code("\n".join(st.session_state.current_code), language="python")
    if st.session_state.previous_code:
        st.subheader("Old version")
        st.code("\n".join(st.session_state.previous_code), language="python")


with col2:
    st.header("Code changes")
    if st.session_state.current_code and st.session_state.previous_code:
        change_description = generate_documentation_diff_description(
            "\n".join(st.session_state.current_code),
            "\n".join(st.session_state.previous_code)
        )
        st.subheader("Changed description")
        st.markdown(change_description)

        highlighted_code = highlight_changes(st.session_state.previous_code, st.session_state.current_code)
        st.subheader("Changes log")
        st.code(highlighted_code, language="diff")


with col3:
    st.header("Documentation")
    if st.session_state.current_code:
        current_code = "\n".join(st.session_state.current_code)
        similar_documentation, similar_code = st.session_state.rag.search(current_code, top_k=RETRIEVE_TOP_K)
        for c, d in zip(similar_documentation, similar_code):
            print(c)
            print(d)
            print("______________________")
        documentation = generate_documentation(
            current_code, similar_documentation, similar_code
        )
        st.markdown(documentation)
