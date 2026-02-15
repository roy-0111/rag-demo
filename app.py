import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# page styling
st.set_page_config(page_title="Financial RAG Demo", layout="centered")

st.title("Financial Analytics RAG System")
st.markdown("### Structure-Aware QA for Corporate Disclosures")

# load the API Key from Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Cache the model loading so it doesn't reload on every click
@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    template = """Use the context below to answer the question. Keep your answer factual and precise.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Load the system
try:
    with st.spinner("Loading AI Models and Vector Database..."):
        qa_chain = load_rag_system()
    st.success("System Online and Ready.")
except Exception as e:
    st.error(f"Failed to load the system. Error: {e}")

#  User Interface
query = st.text_input("Ask a question about the financial report (e.g., 'What is the net liquidity position?' or 'Summarize lease liabilities.'):")

if st.button("Run Analysis"):
    if query:
        with st.spinner("Searching millions of parameters..."):
            result = qa_chain.invoke({"query": query})
            
            st.markdown("### 🤖 AI Analysis")
            st.info(result['result'])
            
            st.markdown("### 📚 Retrieved Context & Citations")
            for i, doc in enumerate(result['source_documents']):
                company = doc.metadata.get('company', 'Unknown')
                page = doc.metadata.get('page', '?')
                with st.expander(f"Citation {i+1}: {company} - Page {page}"):
                    st.write(doc.page_content[:1000] + "...")
    else:
        st.warning("Please enter a question first.")
