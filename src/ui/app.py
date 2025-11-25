import streamlit as st
import os
import tempfile
from langchain_ollama import OllamaLLM
from src.core.rag_pipeline import RAGPipeline
from src.core.graph_gen import extract_graph_data, generate_network_graph

# Page Config
st.set_page_config(page_title="InsightSense", layout="wide")

st.title("InsightSense 🧠")
st.subheader("LLM-Based Research Assistant with Knowledge Graph Generation")

# Initialize Session State
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
if "llm" not in st.session_state:
    st.session_state.llm = OllamaLLM(model="llama3")

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("1. Upload Research Paper")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        if st.button("Process PDF"):
            with st.spinner("Ingesting, Chunking, and Indexing..."):
                msg = st.session_state.rag.process_pdf(tmp_path)
                st.success(msg)
                os.remove(tmp_path) # Clean up

# --- Main Tabs ---
tab1, tab2 = st.tabs(["💬 Q&A Chat", "🕸️ Summary & Graph"])

# --- Tab 1: Q&A ---
with tab1:
    question = st.text_input("Ask a question about the paper:")
    if st.button("Get Answer"):
        if not st.session_state.rag.vector_store:
            st.error("Please process a PDF first.")
        else:
            # 1. Retrieve
            docs = st.session_state.rag.get_context(question)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            # 2. Augment & Generate
            prompt = f"""
            Answer the question based ONLY on the context provided below.
            Context: {context_text}
            Question: {question}
            """
            response = st.session_state.llm.invoke(prompt)
            st.write(response)
            
            with st.expander("View Retrieved Context (Evidence)"):
                for i, doc in enumerate(docs):
                    st.caption(f"Chunk {i+1}: {doc.page_content[:200]}...")

# --- Tab 2: Summary & Graph ---
with tab2:
    st.write("Generate a summary based on your custom format and visualize it.")
    
    user_format = st.text_area(
        "Define your summary format:", 
        value="1. Key Methodology\n2. Main Results\n3. Future Work"
    )
    
    if st.button("Generate Summary & Graph"):
        if not st.session_state.rag.vector_store:
            st.error("Please process a PDF first.")
        else:
            with st.spinner("Generating Summary..."):
                # 1. Generate Summary
                # (In a real app, you might retrieve specific 'summary' chunks, 
                # here we assume we want a general summary based on what we know or the first few pages)
                # For better results, you might want to pass the abstract or intro chunks specifically.
                prompt = f"""
                Summarize the uploaded paper following this EXACT format:
                {user_format}
                """
                summary = st.session_state.llm.invoke(prompt)
                st.markdown("### Generated Summary")
                st.write(summary)
            
            with st.spinner("Building Knowledge Graph..."):
                # 2. Generate Graph
                relations = extract_graph_data(st.session_state.llm, summary)
                if relations:
                    img_bytes = generate_network_graph(relations)
                    st.image(img_bytes, caption="Knowledge Graph of Summary")
                    
                    st.download_button(
                        label="Download Graph Image",
                        data=img_bytes,
                        file_name="knowledge_graph.png",
                        mime="image/png"
                    )
                else:
                    st.warning("Could not extract relationships for graph generation.")