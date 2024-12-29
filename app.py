import gradio as gr
from document_processing import DocumentProcessor
from llm_model import LLM
from rag_chain import RAGChain
from agent import initialize_agent
import os
import shutil

# Configuration
DOC_DIR = "/mnt/c/Users/Anushtup/OneDrive/Documents/BITS/obsidian_home"  # Directory containing your documents
VECTORSTORE_PATH = "vectorstore_db"
# MODEL_PATH = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MODEL_PATH = "/home/anushtup/mistral-7b-instruct-v0.2.Q5_K_M.gguf"  # Replace with your chosen model

# Initialize components
document_processor = DocumentProcessor(DOC_DIR, VECTORSTORE_PATH)
llm_instance = LLM(MODEL_PATH)
rag_chain_instance = RAGChain(llm_instance.get_llm(), document_processor.get_retriever())

# Initialize components
document_processor = DocumentProcessor(DOC_DIR, VECTORSTORE_PATH)
llm_instance = LLM(MODEL_PATH)
# rag_chain_instance = RAGChain(llm_instance.get_llm(), document_processor.get_retriever())
rag_chain_instance = None  # Initialize later
agent_executor = initialize_agent(llm_instance.get_llm(), document_processor.get_retriever())

def update_vectorstore():
    global rag_chain_instance
    # Delete existing vector store
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)  # Use shutil.rmtree to delete the directory

    # Recreate vector store
    document_processor.vectorstore = document_processor.load_or_create_vectorstore()
    retriever = document_processor.get_retriever()
    rag_chain_instance = RAGChain(llm_instance.get_llm(), retriever)

    return "Vector store updated!"

# Gradio chat interface
def query_documents(question, history):
    global rag_chain_instance
    if rag_chain_instance is None:
      retriever = document_processor.get_retriever()
      rag_chain_instance = RAGChain(llm_instance.get_llm(), retriever)
    response = rag_chain_instance.invoke_chain(question)
    history.append((question, response["answer"]))
    if len(history) > 3:
        history = history[-3:]
    return history, history

def run_agent(task, history):
    global agent_executor
    response = agent_executor.invoke({"input": task})
    history.append((task, response["output"]))
    return history, history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Khoj-Lite", value=[], elem_id="chatbot")
    msg = gr.Textbox(label="Question", placeholder="Ask a question about your documents")
    agent_task = gr.Textbox(label="Agent Task", placeholder="Enter a task for the agent")

    with gr.Row():
        update_button = gr.Button("Update Vector Store")
        agent_button = gr.Button("Run Agent")

    update_status = gr.Textbox(label="Update Status")

    clear = gr.ClearButton([msg, chatbot, agent_task])

    def respond(message, chat_history):
        full_response = query_documents(message, chat_history)
        chat_history, bot_message = full_response
        return "", chat_history

    def respond_agent(task, chat_history):
        full_response = run_agent(task, chat_history)
        chat_history, bot_message = full_response
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    agent_task.submit(respond_agent, [agent_task, chatbot], [agent_task, chatbot])
    update_button.click(update_vectorstore, inputs=[], outputs=update_status)
    agent_button.click(respond_agent, [agent_task, chatbot], [agent_task, chatbot])

if __name__ == "__main__":
    demo.launch()