import streamlit as st
from utils.db_utils import init_db, insert_history, fetch_history
from utils.model_utils import load_model, generate_response
from utils.retrieval_utils import PDFRetriever

st.set_page_config(page_title="RAG Chatbot")

st.title("RAG Chatbot")

# initialize db and retriever
DB_PATH = "rag_history.db"
conn = init_db(DB_PATH)

if "retriever" not in st.session_state:
    st.session_state.retriever = PDFRetriever("data/pdfs")

if "model" not in st.session_state:
    model, tokenizer = load_model("BioMistral/BioMistral-7B")
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer

# sidebar history
st.sidebar.header("Conversation History")
for q, a, src, ts in fetch_history(conn):
    st.sidebar.markdown(f"**Q:** {q}\n\n**A:** {a}\n\n*Source: {src}*\n---")

question = st.text_input("Ask a question")
if st.button("Submit") and question:
    results = st.session_state.retriever.query(question, k=3)
    context = "\n".join([text for text, _ in results])
    source = results[0][1] if results else ""
    prompt = f"Answer the question using the context below.\nContext:\n{context}\nQuestion: {question}\nAnswer:"
    answer = generate_response(st.session_state.model, st.session_state.tokenizer, prompt)
    insert_history(conn, question, answer, source)
    st.markdown("### Response")
    st.write(answer)
    st.markdown(f"*Source: {source}*")
