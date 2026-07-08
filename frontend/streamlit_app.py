import streamlit as st
from rag import rag_answer

st.title("RAG Assistant")

mode = st.sidebar.radio("Mode", ["Query Search", "CV Search"])
query = st.text_area("Question")

if st.button("Submit"):
    if mode == "Query Search":
        result = rag_answer.answer(query)
    else:
        pass
        #result = resume_rag.answer(query)

    st.markdown(result["answer"])