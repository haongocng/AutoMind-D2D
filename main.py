import os
import re
from logger import setup_logger
from langchain_core.messages import HumanMessage

from load_cfg import LANGCHAIN_API_KEY, WORKING_DIRECTORY, GOOGLE_API_KEY
from core.workflows import WorkflowManager
from core.language_models import LanguageModelManager
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

class MultiAgentSystem:
    def __init__(self):
        self.logger = setup_logger()
        self.setup_environment()
        self.lm_manager = LanguageModelManager()
        self.workflow_manager = WorkflowManager(
            language_models=self.lm_manager.get_models(),
            working_directory=WORKING_DIRECTORY
        )

    def setup_environment(self):
        """Initialize environment variables"""
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_TRACING"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "Multi-Agents for DA"


        if not os.path.exists(WORKING_DIRECTORY):
            os.makedirs(WORKING_DIRECTORY)
            self.logger.info(f"Created working directory: {WORKING_DIRECTORY}")

    def parse_input(self, user_input: str):
        """Extracts datapath from user input."""
        match = re.search(r"datapath:(.*)", user_input)
        if match:
            return match.group(1).strip()
        return None

    def run(self, user_input: str) -> None:
        """Run the multi-agent system with user input"""
        datapath = self.parse_input(user_input)
        if not datapath:
            self.logger.error("Could not find 'datapath:' in user input.")
            return

        graph = self.workflow_manager.get_graph()
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "sender": "",
            "datapath": datapath,
            "eda_report": "",
            "statistic_report": "",
            "cluster_report": "",
            "visualization_report": "",
            "hypothesis_report": "",
            "reasoning_report": "",
            "total_summary_report": "",
            "final_report": "",

            "transformed_datapath": "",
            "feature_engineering_report": "",
            "model_selection_report": "",
            "model_training_report": "",
            "trained_model_path": "",
            "model_evaluation_report": "",
            "prediction_report": "",
        }

        events = graph.stream(
            initial_state,
            {"recursion_limit": 50},
            stream_mode="values"
        )
        
        for event in events:
            self.logger.debug(f"Graph stream event: {event}")
            if "messages" in event:
                message = event["messages"][-1]
                if not isinstance(message, tuple) and hasattr(message, 'content'):
                    if message.content:
                        self.logger.info(f"===== Terminal Output =====\nName: {getattr(message, 'name', 'N/A')}\n{message.content}\n=========================")
                if isinstance(message, tuple):
                    print(message, end='', flush=True)
                else:
                    message.pretty_print()

def main():
    """Main entry point"""
    system = MultiAgentSystem()
    
    # user_input = '''
    # datapath:drinks_data.csv
    # Please analyze the 'drinks_data.csv' dataset. Your task is to cluster each instance in the data form into some group, then profile each group based on their digital readiness.
    # '''
    # system.run(user_input)

    # user_input_classification = '''
    # datapath: edudata_english.csv
    # Please analyze the 'edudata_english.csv' dataset and perform a full classification analysis on 'edudata_english.csv' dataset. Now your task is to predict the 'I am willing to share my digital skills with other students' field in the data.
    # Do the workflow: DataExplorer -> FeatureEngineering -> ModelSelection -> ModelTraining-> ModelEvaluation -> Prediction -> Synthesis. 
    # Note that: the final evaluation report MUST include the following metrics, calculated on the test set: Accuracy, Precision, Recall, F1-Score. You must use the `execute_code` tool to run all your code.
    # '''
    # system.run(user_input_classification)

    # user_input_fake_news = '''
    # full_train_datapath: train.csv (if want to read the full train data, please use 'encoding="utf-8"' and 'sep='\t' to load the data)
    # sample_train_datapath: sampled_dataset.csv
    # test_datapath: test.csv
    # task: Binary Classification
    # target_column: label
    # objective: >
    #     Build a model to classify news articles as fake or real. The target 'label' is 1 for fake and 0 for true.
    #     IMPORTANT STRATEGY: This is a large dataset. To ensure efficiency, please adhere to the following workflow:
    #     1. First,  For all initial analysis steps (Data Exploration, Statistics, Feature Engineering, and Model Selection), all agents MUST use the smaller 'sampled_data.csv'. This will speed up the process significantly. (Please MUSTN'T PRINT the data to the console to avoid overload.)
    #     2. For the final model training step, the ModelTraining agent MUST load the FULL 'train.csv' dataset. It should then re-apply the feature engineering steps that were developed on the sample data to the full dataset before training the final model.
    #     3. After the final model is trained on the full data, use it to predict the labels for the articles in 'test.csv'.
    #     The final evaluation metric is Accuracy. You must use the `execute_code` tool to run all your code.
    # Note that: if all agents need to read_document to check the data, they must read a small sample in sampled_data.csv, not the full data.
    # '''
    # system.run(user_input_fake_news)

    user_input_heart_disease = '''
    datapath: heart_train.csv
    test_datapath: heart_test.csv
    task: Binary Classification
    target_column: HeartDisease
    objective: Build a model using train.csv to predict the presence of heart disease (target 'HeartDisease' = 1 for presence, 0 for absence). After training, use the model to make predictions on heart_test.csv (datapath: "heart_test.csv"). The final evaluation metric is Accuracy.
    '''
    system.run(user_input_heart_disease)

    # user_input_shop_customer = '''
    # datapath: Customers.csv
    # task: Clustering
    # objective: Analyze the dataset to segment customers into distinct groups based on their characteristics like Age, Annual Income, and Spending Score. After clustering, provide a detailed profile for each customer segment, describing their key attributes in final insights report.
    # '''
    # system.run(user_input_shop_customer)

if __name__ == "__main__":
    main()