import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# # 📂 Directory for PDFs
# PDF_DIRECTORY = "./peatland_data/"

# # 📍 Persistent Chroma DB Path
CHROMA_DB_PATH = "./chroma_db"

# # 🔹 **Load PDFs and Process Documents**
# @st.cache_data(show_spinner="📂 Loading and processing PDFs...")
# def load_pdfs_from_directory(pdf_directory):
#     all_docs = []
    
#     for filename in os.listdir(pdf_directory):
#         if filename.endswith(".pdf"):
#             pdf_path = os.path.join(pdf_directory, filename)
#             st.write(f"📂 Loading: {filename}")

#             loader = PyPDFLoader(pdf_path)
#             data = loader.load()

#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#             docs = text_splitter.split_documents(data)

#             all_docs.extend(docs)

#     return all_docs

# docs = load_pdfs_from_directory(PDF_DIRECTORY)
# st.write(f"✅ **Total documents loaded:** {len(docs)}")


# 🔹 **Load or Create Chroma Vector Database**
@st.cache_resource(show_spinner="⚡ Loading Chroma vector store...")
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists(CHROMA_DB_PATH):
        st.success("✅ Using existing Chroma database.")
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    else:
        st.warning("⚠️ No existing Chroma DB found. Creating one...")
        # docs = load_pdfs_from_directory(PDF_DIRECTORY)  # Load only if DB is missing
        # vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=CHROMA_DB_PATH)
        # st.success("✅ Chroma database created and saved!")
        # return vectorstore

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# 🔹 **Setup Gemini LLM**
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=500)

# 🔹 **Prompt Template**
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, make up the answer "
    "using your own knowledge. Must use bullet points and must give answer. "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 🔹 **Streamlit UI**
st.title("🌍 AI-Powered Peatland & Carbon Footprint Assistant")
st.write("💡 Ask questions about **peatlands, carbon footprint, NDVI, NDWI, NDMI, and more!**")

user_query = st.text_input("🔹 Enter your question:")
if st.button("Ask AI"):
    if user_query:
        response = rag_chain.invoke({"input": user_query})
        st.subheader("🤖 AI Response")
        st.write(response["answer"])
    else:
        st.warning("⚠️ Please enter a question before clicking 'Ask AI'.")


def knowledge_base():
    import os
    import streamlit as st
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_chroma import Chroma
    from dotenv import load_dotenv

    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    CHROMA_DB_PATH = "./chroma_db"

    @st.cache_resource(show_spinner="⚡ Loading Chroma vector store...")
    def get_vectorstore():
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        if os.path.exists(CHROMA_DB_PATH):
            return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        else:
            st.warning("⚠️ No existing Chroma DB found. Creating one...")

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=500)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, make up the answer "
        "using your own knowledge. Must use bullet points. "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    st.title("🌍 AI-Powered Peatland & Carbon Footprint Assistant")
    st.write("💡 Ask questions about **peatlands, carbon footprint, NDVI, NDWI, NDMI, and more!**")

    user_query = st.text_input("🔹 Enter your question:")
    if st.button("Ask AI"):
        if user_query:
            response = rag_chain.invoke({"input": user_query})
            st.subheader("🤖 AI Response")
            st.write(response["answer"])
        else:
            st.warning("⚠️ Please enter a question before clicking 'Ask AI'.")