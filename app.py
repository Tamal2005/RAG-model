import streamlit as st
from streamlit_chat import message
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'
# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="RAG Chat Interface",
    page_icon="💬",
    layout="wide"
)

CHROMA_PATH = "chroma_db"

# ------------------------------------------------------------
# INITIALIZATION FUNCTIONS (LOAD ONCE)
# ------------------------------------------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

@st.cache_resource
def load_db(_embeddings):
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

@st.cache_resource
def load_llm(model_name, temperature):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    return HuggingFacePipeline(pipeline=pipe)

def build_qa(llm, retriever):
    prompt_template = """
You are a knowledgeable assistant specialized in answering questions about machine learning and statistics using the provided context and your own expertise. Use the context to provide accurate and concise answers.

Context:
{context}

Question:
{question}

Answer:
"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa


# -------------------------------
# UI INITIALIZATION
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")


    temperature = st.slider("Temperature", 0.0, 1.5, 0.3)

    top_k = st.slider("Top-K (Retrieval)", 1, 10, 2)

    st.write("---")
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------
# LOAD RAG COMPONENTS
# ------------------------------------------------------------
embeddings = load_embeddings()
db = load_db(embeddings)
llm = load_llm('Qwen/Qwen2.5-0.5B-Instruct', temperature)

retriever = db.as_retriever(search_kwargs={"k": top_k})
qa_chain = build_qa(llm, retriever)

# ------------------------------------------------------------
# RAG RESPONSE FUNCTION
# ------------------------------------------------------------
def rag_response(prompt):
    result = qa_chain.invoke({"query": prompt})
    raw_answer = result['result']
    
    # ✅ Clean up the answer
    # Remove everything before "Answer:"
    if "Answer:" in raw_answer:
        answer = raw_answer.split("Answer:")[-1].strip()
    else:
        answer = raw_answer
    return answer

# -------------------------------
# Chat UI Header
# -------------------------------
st.title("💬 RAG Chat Interface")
st.caption("Ask about Machine learning or statistics!")

# -------------------------------
# Display Chat History
# -------------------------------
for msg in st.session_state.messages:
    message(msg["content"], is_user=msg["role"] == "user")

# -------------------------------
# Chat Input
# -------------------------------
user_input = st.chat_input("Type your question...")

# -------------------------------
# MAIN CHAT PROCESSING
# -------------------------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    message(user_input, is_user=True)

    with st.spinner("Thinking with RAG..."):
        ai_response = rag_response(user_input)

    # Typing animation
    placeholder = st.empty()
    typed = ""
    for ch in ai_response:
        typed += ch
        placeholder.markdown(f"```\n{typed}\n```")
        time.sleep(0.002)
    placeholder.empty()

    message(ai_response, is_user=False)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
