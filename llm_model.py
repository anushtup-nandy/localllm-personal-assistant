from ctransformers import AutoModelForCausalLM, AutoConfig
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM as LangChainLLM
from pydantic import Field, BaseModel

class LLM(LangChainLLM, BaseModel):
    model_path: str = Field(description="Path to the model file")
    model: Any = Field(default=None, description="The loaded CTransformers model")
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        conf = AutoConfig.from_pretrained(model_path)
        conf.config.max_new_tokens = 2048
        conf.config.context_length = 4096
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            config=conf
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        response = self.model(prompt)
        return response

    @property
    def _llm_type(self) -> str:
        return "ctransformers"

    def get_llm(self):
        return self