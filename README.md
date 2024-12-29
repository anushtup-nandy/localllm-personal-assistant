# Khoj-Lite: A Local-First, Document-Aware AI Assistant

Khoj-Lite is a personal AI assistant inspired by the [Khoj project](https://github.com/khoj-ai/khoj) but designed to be simpler and more lightweight. It allows you to interact with your documents using natural language, leveraging the power of a locally running large language model (LLM).

## Features

*   **Document Indexing and Retrieval:** Indexes your documents (text files, PDFs, Markdown, etc.) using FAISS for efficient similarity search.
*   **Local LLM:** Uses a quantized Mistral 7B model (or other models from TheBloke on Hugging Face) for natural language understanding and generation.
*   **Retrieval-Augmented Generation (RAG):** Combines the LLM with document retrieval to provide contextually relevant answers to your questions.
*   **Gradio Interface:** Provides a user-friendly web interface for interaction.
*   **Obsidian Integration:** Specifically designed to work with your Obsidian notes (can be configured for other directories).
*   **Privacy-Focused:** Runs entirely locally; your data never leaves your computer.

## Requirements

*   Python 3.10+
*   CUDA-enabled GPU (recommended for faster inference)

## Installation

1. **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd personal_assistant
    ```

2. **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *   **Note:** This command installs most of the necessary dependencies. You might need to install additional packages for specific file types. See the "Additional Dependencies" section below.

4. **Download a quantized LLM:**

    *   Download a quantized Mistral 7B model (GGUF format) from TheBloke on Hugging Face. For example: [TheBloke/Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
    *   Place the downloaded model file in the project directory.

5. **Configure `app.py`:**

    *   Update the following variables in `app.py`:
        *   `DOC_DIR`: The path to your Obsidian vault (or the directory containing your documents).
        *   `VECTORSTORE_PATH`: The directory where the vector store will be saved (default: `vectorstore_db`).
        *   `MODEL_PATH`: The path to your downloaded LLM model file.

## Additional Dependencies

To handle various file types, you might need to install additional dependencies:

*   **For PDF processing:**

    ```bash
    pip install "unstructured[pdf]"
    sudo apt-get update  # On Linux
    sudo apt-get install -y poppler-utils libmagic-dev # On Linux
    ```

*   **For other file types (Markdown, images, HTML, etc.):**

    ```bash
    pip install "unstructured[all-docs]"
    ```

    Refer to the [unstructured documentation](https://unstructured-io.github.io/unstructured/installation/full_installation.html) for a complete list of dependencies.

*   **System Dependencies:**

    You might also need to install system-level dependencies like `tesseract-ocr` (for OCR) and `pandoc` (for document conversion). For example, on Ubuntu/Debian:

    ```bash
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr pandoc
    ```

## Usage

1. **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. **Run the application:**

    ```bash
    python app.py
    ```

3. **Open your browser and go to the provided URL (usually `http://127.0.0.1:7860`).**

4. **Interact with the assistant through the Gradio interface.**

## Updating the Knowledge Base

Currently, the vector store (knowledge base) is created once and then loaded on subsequent runs. To update the knowledge base with changes to your documents:

1. **Click the "Update Vector Store" button in the Gradio interface.** This will delete the existing vector store and recreate it from scratch, indexing the updated documents.

## Future Enhancements

*   **Agentic Capabilities:**
    *   Reasoning and planning.
    *   Goal setting and proactive actions.
    *   Integration with external tools (calendar, to-do list, etc.).
*   **Notion Integration:**
    *   Connect to the Notion API to index and query your Notion pages.
*   **Continuous Learning:**
    *   Implement feedback mechanisms to learn from user interactions.
    *   Explore reinforcement learning for optimization.
*   **Voice Interaction:**
    *   Add speech-to-text and text-to-speech capabilities.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.