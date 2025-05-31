import streamlit as st
from utils.db_utils import init_db, insert_history, fetch_history
from utils.model_utils import load_model, generate_response
from utils.retrieval_utils import PDFRetriever

st.set_page_config(page_title="Fine-tuned Chatbot")

st.title("Fine-tuned Chatbot")

DB_PATH = "finetuned_history.db"
conn = init_db(DB_PATH)

if "finetuned_retriever" not in st.session_state:
    st.session_state.finetuned_retriever = PDFRetriever("data/pdfs")

if "finetuned_model" not in st.session_state:
    model, tokenizer = load_model("BioMistral/BioMistral-7B")
    st.session_state.finetuned_model = model
    st.session_state.finetuned_tokenizer = tokenizer

st.sidebar.header("Conversation History")
for q, a, src, ts in fetch_history(conn):
    st.sidebar.markdown(f"**Q:** {q}\n\n**A:** {a}\n\n*Source: {src}*\n---")

question = st.text_input("Ask a question")
if st.button("Submit") and question:
    # retrieve for source only
    results = st.session_state.finetuned_retriever.query(question, k=1)
    source = results[0][1] if results else ""
    answer = generate_response(
        st.session_state.finetuned_model,
        st.session_state.finetuned_tokenizer,
        question,
    )
    insert_history(conn, question, answer, source)
    st.markdown("### Response")
    st.write(answer)
    st.markdown(f"*Source: {source}*")
