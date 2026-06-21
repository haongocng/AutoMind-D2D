from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from load_cfg import DEEPINFRA_API_KEY, DEEPINFRA_API_BASE
from logger import setup_logger
#To use openai api key
import os
# os.environ["OPENAI_API_KEY"] = "your_api_key_here"

class LanguageModelManager:
    def __init__(self):
        self.logger = setup_logger()
        self.llm = None
        self.power_llm = None
        self.json_llm = None
        self.initialize_llms()

    def initialize_llms(self):
        try:
            #model 
            #Qwen/Qwen3-Next-80B-A3B-Instruct
            #meta-llama/Llama-3.3-70B-Instruct
            #google/gemini-2.5-pro
            #gpt-4o
            self.llm = ChatOpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_API_BASE,
                model="Qwen/Qwen2.5-72B-Instruct",
            )
            
            self.power_llm = ChatOpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_API_BASE,
                model="Qwen/Qwen2.5-72B-Instruct",
            )

            self.json_llm = ChatOpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_API_BASE,
                model="Qwen/Qwen2.5-72B-Instruct",
            )

            #  self.llm = ChatOpenAI(
            #     model="gpt-4o",
            #     temperature=0
            # )
            
            # self.power_llm = ChatOpenAI(
            #     model="gpt-4o",
            #     temperature=0.5
            # )

            # self.json_llm = ChatOpenAI(
            #     model="gpt-4o",
            #     temperature=0,
            #     model_kwargs={"response_format": {"type": "json_object"}}
            # )
            self.logger.info("DeepInfra language models initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error creating LLM: {str(e)}")
            raise

    def get_models(self):
        return {
            "llm": self.llm,
            "power_llm": self.power_llm,
            "json_llm": self.json_llm
        }