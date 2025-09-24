import os
import gradio as gr
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever

os.environ['groq_api_key'] = "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"

# ----------------------------
# Load documents
# ----------------------------
document_path = "C:/Users/USER/Downloads/LUX TECH ACADEMY/RAG CHATBOT/document"

loader = PyPDFDirectoryLoader(document_path)
documents = loader.load()
#print(len(documents))

# ----------------------------
# Define splitters
# ----------------------------
# For parent docs (large chunks)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

# For child docs (small chunks used for embeddings/retrieval)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=300
)

# ----------------------------
# Embeddings + Vector Store
# ----------------------------
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="sample_collection",
    embedding_function=embeddings_model,
)

# ------------------------------
# ParentDocumentRetriever
# ----------------------------
from langchain.storage import InMemoryByteStore
# Create a simple in-memory docstore for parent docs
docstore = InMemoryByteStore()

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    byte_store=docstore,     
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents (they are automatically split into parent + child chunks)
retriever.add_documents(documents)
#print("Documents added to ParentDocumentRetriever")

# ----------------------------
# LLM
# ----------------------------
LLM = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("groq_api_key"),
    temperature=0.5
)

# ----------------------------
# Prompt Template
# ----------------------------
rag_prompt_template = PromptTemplate(
    input_variables=["question", "history", "knowledge"],
    template="""
    You are a contextual assistant. Answer the question using only the knowledge provided.
    Do not say "according to documents" or similar.

    Question: {question}

    Conversation history: {history}

    Knowledge:
    {knowledge}
    """
)

# ----------------------------
# Ask function
# ----------------------------
def ask_question(question, history=None):
    history = history or []

    # Retrieve parent docs (not just child snippets)
    docs = retriever.get_relevant_documents(question)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    # Format the prompt
    rag_prompt = rag_prompt_template.format(
        question=question,
        history=history,
        knowledge=knowledge
    )

    # Stream response
    response = ""
    for chunk in LLM.stream(rag_prompt):
        response += chunk.content
    return response

# ----------------------------
# Chatbot function
# ----------------------------
def chatbot_fn(question, history):
    history.append({"role": "user", "content": question})
    answer = ask_question(question, history)
    history.append({"role": "assistant", "content": answer})
    return history, history

# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Financial Inclusion in Kenya Chatbot - Survey results and analysis from FinAccess 2009")
    chatbot = gr.Chatbot(type="messages")  
    msg = gr.Textbox(label="Ask a question")
    state = gr.State([])

    def respond(message, history):
        return chatbot_fn(message, history)

    msg.submit(respond, [msg, state], [chatbot, state])

demo.launch(share = True)
