# Streamlit Multipage Chatbot

This app demonstrates two approaches for building a chatbot with the
`BioMistral/BioMistral-7B` model.

- **RAG Chatbot** – uses retrieval augmented generation with a FAISS
  vector store built from local PDFs.
- **Fine-tuned Chatbot** – uses the same model after fine-tuning on the
  dataset.

Both pages store conversation history in separate SQLite databases and
show the source PDF for each answer.

## Setup
1. Place your research paper PDFs in `data/pdfs`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run Home.py
   ```

If the model weights are not present locally, `transformers` will attempt
to download them from Hugging Face on first run.
