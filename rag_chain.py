from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class RAGChain:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.rag_chain_with_source = self.create_rag_chain()

    def create_rag_chain(self):
        template = """You are a knowledgeable assistant tasked with providing detailed and informative answers.
        Use the following pieces of retrieved context to answer the question thoroughly and comprehensively.
        
        Important Guidelines:
        - Provide detailed explanations with relevant examples when possible
        - Include specific information from the context
        - Break down complex ideas into clear, understandable parts
        - If relevant, mention relationships to other topics or concepts
        - If you don't know something or if the context doesn't provide enough information, be honest about it
        - Aim to be both informative and engaging in your response
        
        Question: {question}
        
        Context: {context}
        
        Detailed Answer: Let me provide a comprehensive response to your question."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | self.llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {
                "context": self.retriever,
                "question": RunnablePassthrough(),
            }
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source

    def invoke_chain(self, question):
        response = self.rag_chain_with_source.invoke(question)
        return response