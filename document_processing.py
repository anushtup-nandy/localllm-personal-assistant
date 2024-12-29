import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from unstructured.partition.auto import partition
from unstructured.file_utils.filetype import FileType
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, doc_dir, vectorstore_path="vectorstore_db", embedding_model_name="sentence-transformers/all-mpnet-base-v2"):
        if not os.path.exists(doc_dir):
            raise ValueError(f"Document directory does not exist: {doc_dir}")

        self.doc_dir = doc_dir
        self.vectorstore_path = vectorstore_path
        self.embedding_model_name = embedding_model_name
        self.vectorstore = self.load_or_create_vectorstore()

    def extract_metadata(self, filename):
        metadata = {}
        try:
            with open(filename, 'r') as f:
                content = f.read()

            # Extract tags using a regular expression (modify if needed)
            tags = re.findall(r"#(\w+)", content)
            metadata['tags'] = tags

            # Get creation and modification times
            metadata['created_at'] = os.path.getctime(filename)
            metadata['modified_at'] = os.path.getmtime(filename)

            # Extract backlinks (links to other notes) - example using [[ ]]
            backlinks = re.findall(r"\[\[(.*?)\]\]", content)
            metadata['backlinks'] = backlinks

        except Exception as e:
            logger.error(f"Error extracting metadata from {filename}: {e}")

        return metadata

    def load_and_split_documents(self):
        logger.info(f"Loading documents from: {self.doc_dir}")
        loader = DirectoryLoader(self.doc_dir, recursive=True, silent_errors=True)
        docs = []
        for doc in loader.load():
            try:
                elements = partition(filename=doc.metadata['source'])
                text_content = "\n".join([str(el) for el in elements])

                # Extract metadata
                metadata = self.extract_metadata(doc.metadata['source'])
                metadata['source'] = doc.metadata['source']

                # Create a new document with extracted text and metadata
                extracted_doc = {
                    'page_content': text_content,
                    'metadata': metadata
                }
                docs.append(extracted_doc)
                logger.info(f"Successfully processed: {doc.metadata['source']}")

            except Exception as e:
                if "Invalid file" in str(e):
                    logger.warning(f"Skipping unsupported file type: {doc.metadata['source']}")
                else:
                    logger.error(f"Error processing {doc.metadata['source']}: {e}")

        logger.info(f"Loaded {len(docs)} documents")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.create_documents([doc['page_content'] for doc in docs], metadatas=[doc['metadata'] for doc in docs])
        logger.info(f"Created {len(splits)} text splits")
        return splits

    def load_or_create_vectorstore(self):
        index_path = os.path.join(self.vectorstore_path, "index.faiss")
        pkl_path = os.path.join(self.vectorstore_path, "index.pkl")

        if os.path.exists(index_path) and os.path.exists(pkl_path):
            logger.info(f"Loading existing vector store from {self.vectorstore_path}")
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            vectorstore = FAISS.load_local(self.vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully")
        else:
            logger.info(f"Vector store not found at {self.vectorstore_path}")
            logger.info("Creating new vector store...")

            # Create directory if it doesn't exist
            os.makedirs(self.vectorstore_path, exist_ok=True)

            splits = self.load_and_split_documents()
            if not splits:
                raise ValueError("No documents were loaded to create the vector store")

            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

            logger.info(f"Saving vector store to {self.vectorstore_path}")
            vectorstore.save_local(self.vectorstore_path)
            logger.info("Vector store created and saved successfully")

        return vectorstore

    def get_retriever(self):
        return self.vectorstore.as_retriever()