"""
Advanced RAG System  - Streamlit Web UI
==========================================

Professional web interface with real-time chat and document management.

Run with:
    streamlit run app.py

Make sure to have the main AdvancedRAGSystem code in a file named 'advanced_rag.py'
"""

import streamlit as st
import os
from datetime import datetime
from pathlib import Path

# Import the RAG system (assuming it's in advanced_rag.py)
# If not, copy the previous code to 'advanced_rag.py'
try:
    from advanced_rag import AdvancedRAGSystem, Config
    SYSTEM_AVAILABLE = True
except:
    SYSTEM_AVAILABLE = False
    st.error("⚠️ Please save the Advanced RAG System code as 'advanced_rag.py' in the same directory")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Advanced RAG System 2025",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #ffc107;
        margin: 0.25rem 0;
    }
    .stat-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

if 'system' not in st.session_state:
    st.session_state.system = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = []

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR - CONFIGURATION & DOCUMENT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # API Token
    with st.expander("🔑 Hugging Face Token", expanded=not st.session_state.system):
        hf_token = st.text_input(
            "Enter your token",
            type="password",
            help="Get your token from https://huggingface.co/settings/tokens"
        )
        
        if st.button("Initialize System", disabled=not hf_token):
            if SYSTEM_AVAILABLE:
                with st.spinner("Initializing Advanced RAG System..."):
                    try:
                        st.session_state.system = AdvancedRAGSystem(token=hf_token)
                        st.success("✅ System initialized!")
                    except Exception as e:
                        st.error(f"❌ Initialization failed: {e}")
            else:
                st.error("System code not available")
    
    st.markdown("---")
    
    # Document Upload
    st.markdown("## 📁 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or TXT files to add to the knowledge base"
    )
    
    if st.button("Process Documents", disabled=not uploaded_files or not st.session_state.system):
        if uploaded_files and st.session_state.system:
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files temporarily
                    temp_dir = Path("temp_uploads")
                    temp_dir.mkdir(exist_ok=True)
                    
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = temp_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(str(file_path))
                    
                    # Ingest documents
                    st.session_state.system.ingest_documents(file_paths)
                    st.session_state.documents_loaded.extend([f.name for f in uploaded_files])
                    
                    st.success(f"✅ Processed {len(uploaded_files)} documents!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing documents: {e}")
    
    # Show loaded documents
    if st.session_state.documents_loaded:
        st.markdown("### 📚 Loaded Documents")
        for doc in st.session_state.documents_loaded:
            st.markdown(f"- {doc}")
    
    st.markdown("---")
    
    # Advanced Options
    with st.expander("🔧 Advanced Options"):
        use_multi_query = st.checkbox("Multi-Query Retrieval", value=True,
                                      help="Generate multiple query variations (improves accuracy)")
        use_reranking = st.checkbox("Re-ranking", value=True,
                                   help="Re-rank results using cross-encoder (40% better)")
        show_sources = st.checkbox("Show Source Details", value=True)
        show_queries = st.checkbox("Show Generated Queries", value=False)
    
    # Reset button
    if st.button("🔄 Reset Conversation"):
        if st.session_state.system:
            st.session_state.system.reset_conversation()
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Stats
    if st.session_state.system:
        st.markdown("### 📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(st.session_state.documents_loaded))
        with col2:
            st.metric("Messages", len(st.session_state.chat_history))

# ═══════════════════════════════════════════════════════════════════════════
# MAIN AREA - HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-header">🤖 Advanced RAG System 2025</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">State-of-the-art Retrieval-Augmented Generation with Multi-Query, Hybrid Search & Re-ranking</div>', unsafe_allow_html=True)

# System status indicator
if st.session_state.system:
    st.success("✅ System Active | Models: meta-llama/Llama-3.1-8B (LLM) + all-MiniLM-L6-v2 (Embeddings)")
else:
    st.warning("⚠️ Please initialize the system in the sidebar")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN AREA - CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

# Display chat history
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        # User message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong><br>
            {message['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant message
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong><br>
            {message['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        # Sources
        if show_sources and 'sources' in message:
            with st.expander(f"📚 Sources ({message['num_sources']} documents)"):
                for i, doc in enumerate(message['sources'], 1):
                    source = doc.metadata.get('filename', 'Unknown')
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i}:</strong> {source}<br>
                        <em>{doc.page_content[:200]}...</em>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Generated queries
        if show_queries and 'queries_used' in message and len(message['queries_used']) > 1:
            with st.expander(f"🔍 Generated Queries ({len(message['queries_used'])})"):
                for i, query in enumerate(message['queries_used'], 1):
                    st.markdown(f"{i}. {query}")

# Chat input
st.markdown("---")

if st.session_state.system:
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Add user message to history
        with st.spinner("🤔 Thinking..."):
            try:
                # Query the system
                result = st.session_state.system.query(
                    user_input,
                    use_multi_query=use_multi_query,
                    use_reranking=use_reranking
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': user_input,
                    'answer': result['answer'],
                    'sources': result['sources'],
                    'num_sources': result['num_sources'],
                    'queries_used': result['queries_used'],
                    'timestamp': datetime.now().isoformat()
                })
                
                # Rerun to update display
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
else:
    st.info("👈 Initialize the system in the sidebar to start chatting")

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **2025 Features:**
    - ✅ Multi-Query Retrieval
    - ✅ Hybrid Search
    - ✅ Re-ranking
    """)

with col2:
    st.markdown("""
    **Technologies:**
    - LangChain
    - Hugging Face
    - ChromaDB
    """)

with col3:
    st.markdown("""
    **Links:**
    - [GitHub](#)
    - [Documentation](#)
    - [Report Issue](#)
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built with ❤️ using state-of-the-art 2025 techniques</div>", unsafe_allow_html=True)
