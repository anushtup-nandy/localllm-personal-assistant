from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain import hub
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class NoteSearchResult(BaseModel):
    """Structure for search results"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float = Field(default=0.0)

class SearchObsidianNotesTool(BaseTool):
    name = "search_obsidian_notes"
    description = "Searches through Obsidian notes to find relevant information. Returns top matches with relevance scores."
    document_processor = None

    def __init__(self, document_processor):
        super().__init__()
        self.document_processor = document_processor

    def _run(self, query: str, top_k: int = 5) -> List[NoteSearchResult]:
        """Search notes with improved relevance scoring"""
        if not self.document_processor:
            raise ValueError("Document processor not initialized")
        
        docs = self.document_processor.vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            NoteSearchResult(
                content=doc[0].page_content,
                metadata=doc[0].metadata,
                relevance_score=doc[1]
            ) for doc in docs
        ]

class GetBacklinksTool(BaseTool):
    name = "get_backlinks"
    description = "Retrieves notes that link to a specific note with additional context"
    document_processor = None

    def __init__(self, document_processor):
        super().__init__()
        self.document_processor = document_processor

    def _run(self, note_name: str) -> Dict[str, Any]:
        """Enhanced backlink retrieval with context"""
        if not self.document_processor:
            raise ValueError("Document processor not initialized")

        backlinks = {}
        for doc in self.document_processor.vectorstore.get():
            if 'metadata' in doc and 'backlinks' in doc['metadata']:
                if note_name in doc['metadata']['backlinks']:
                    backlinks[doc['metadata']['source']] = {
                        'context': doc['page_content'][:200],  # Preview
                        'tags': doc['metadata'].get('tags', []),
                        'created_at': doc['metadata'].get('created_at'),
                        'modified_at': doc['metadata'].get('modified_at')
                    }
        return backlinks

class SummarizeNotesTool(BaseTool):
    name = "summarize_notes"
    description = "Generates both detailed and concise summaries of notes with key points"
    llm = None

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def _run(self, notes: List[str], mode: str = "detailed") -> Dict[str, str]:
        """Improved summarization with multiple modes"""
        if not self.llm:
            raise ValueError("LLM not initialized")

        prompts = {
            "detailed": """Provide a comprehensive summary of the following notes, including:
                         - Main themes and concepts
                         - Key relationships between ideas
                         - Important details and examples
                         Notes: {notes}""",
            "concise": """Create a brief, focused summary of the key points from these notes:
                         Notes: {notes}"""
        }
        
        prompt = prompts.get(mode, prompts["detailed"])
        summary = self.llm.generate_response(prompt.format(notes="\n\n".join(notes)))
        
        return {
            "summary": summary,
            "mode": mode,
            "source_count": len(notes)
        }

def initialize_agent(llm, document_processor):
    """Initialize agent with improved error handling and customization"""
    try:
        tools = [
            SearchObsidianNotesTool(document_processor),
            GetBacklinksTool(document_processor),
            SummarizeNotesTool(llm)
        ]

        # Custom prompt with better instructions
        prompt = hub.pull("hwchase17/react-json")
        
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            max_iterations=5,  # Prevent infinite loops
            early_stopping_method="force",
            handle_parsing_errors=True
        )

        return agent_executor
    except Exception as e:
        raise RuntimeError(f"Failed to initialize agent: {str(e)}")