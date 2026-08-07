import streamlit as st

from src.rag.config import settings
from src.rag.ingest import ingest
from src.rag.pipeline import RAGPipeline

st.set_page_config(page_title="RAG + Hallucination Detector", layout="wide")
st.title("RAG with Groundedness / Hallucination Detection")

with st.sidebar:
    st.header("Ingest documents")
    st.caption(f"Drop .pdf/.txt/.md files into `{settings.data_dir}` then click ingest.")
    if st.button("Ingest documents"):
        with st.spinner("Ingesting..."):
            count = ingest(settings.data_dir)
        st.success(f"Ingested {count} chunks.")

    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, settings.top_k)


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


question = st.text_input("Ask a question about your documents")

if question:
    pipeline = load_pipeline()
    with st.spinner("Retrieving and generating..."):
        result = pipeline.ask(question, top_k=top_k)

    st.subheader("Answer")
    st.write(result["answer"])

    g = result["groundedness"]
    color = "green" if g["label"] == "grounded" else "red"
    st.subheader("Groundedness")
    st.markdown(f":{color}[**{g['label']}**] — overall entailment score: {g['overall_score']:.2f}")
    for s in g["sentences"]:
        st.progress(min(max(s["entailment"], 0.0), 1.0), text=f"{s['entailment']:.2f} — {s['sentence']}")

    st.subheader("Retrieved sources")
    for hit in result["sources"]:
        with st.expander(f"{hit['source']} (distance={hit['distance']:.4f})"):
            st.write(hit["text"])
