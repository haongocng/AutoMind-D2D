from langchain_core.messages import AIMessage, HumanMessage
from core.state import State
import logging
from langchain.agents import AgentExecutor
import json 

logger = logging.getLogger(__name__)

def agent_node(state: State, agent: AgentExecutor, name: str) -> dict:
    """
    Invokes the agent, updates the state with the agent's output, and
    trims the message history to prevent it from getting too long.
    """
    logger.info(f"--- Executing agent: {name} ---")
    try:
        state_for_agent = state.copy()
        
        original_messages = list(state_for_agent["messages"])

        context_str = "\n--- Current System State ---\n"
        for key, value in state_for_agent.items():
            if key not in ["messages", "sender", "intermediate_steps", "agent_scratchpad"] and value:
                if isinstance(value, (dict, list)):
                    context_str += f"{key}: {json.dumps(value, indent=2)}\n" 
                else:
                    context_str += f"{key}: {value}\n"
        context_str += "----------------------------\n"

        state_for_agent["messages"].insert(0, HumanMessage(content=context_str, name="SystemContext"))

        # state_for_agent["messages"] = state["messages"][-6:] 


        result = agent.invoke(state_for_agent)
        
        output = result.get("output", "")
        
        ai_message = AIMessage(content=output, name=name)
        
        state["messages"] = original_messages 
        state["messages"].append(ai_message)
        state["sender"] = name
        
        if name == "DataExplorer":
            state["eda_report"] = output
            logger.info("State updated with EDA Report.")
        elif name == "DataStatistic":
            state["statistic_report"] = output
            logger.info("State updated with Statistic Report.")
        elif name == "DataCluster":
            state["cluster_report"] = output
            logger.info("State updated with Cluster Report.")
        elif name == "DataVisualization":
            state["visualization_report"] = output
            logger.info("State updated with Visualization Report.")
        elif name == "HypothesisGenerator":
            state["hypothesis_report"] = output
            logger.info("State updated with Hypothesis Report.")
        elif name == "Reasoner":
            state["reasoning_report"] = output
            logger.info("State updated with Reasoning Report.")
        elif name == "QualityReview":
            state["total_summary_report"] = output
            logger.info("State updated with Total Summary Report.")
        elif name == "Synthesis":
            state["final_report"] = output
            logger.info("State updated with Final Insights Report.")
        elif name == "FeatureEngineering":
            state["feature_engineering_report"] = output
            if "transformed_data.csv" in output: 
                state["transformed_datapath"] = "transformed_data.csv"
            logger.info("State updated with Feature Engineering Report and transformed data path.")
        elif name == "ModelSelection":
            state["model_selection_report"] = output
            logger.info("State updated with Model Selection Report.")
        elif name == "ModelTraining":
            state["model_training_report"] = output
            if "trained_classification_model.pkl" in output:
                state["trained_model_path"] = "trained_classification_model.pkl"
            logger.info("State updated with Model Training Report and trained model path.")
        elif name == "ModelEvaluation":
            state["model_evaluation_report"] = output
            logger.info("State updated with Model Evaluation Report.")
        elif name == "Prediction":
            state["prediction_report"] = output
            logger.info("State updated with Prediction Report.")
            
        logger.info(f"--- Finished agent: {name} ---")
        return state

    except Exception as e:
        logger.error(f"Error occurred while processing agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error in agent {name}: {str(e)}", name="error")
        state["messages"].append(error_message)
        return state

logger.info("Agent processing node module initialized")