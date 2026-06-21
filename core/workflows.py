from langgraph.graph import StateGraph, END, START
from core.state import State
from core.node import agent_node 
from agents.explorer_agent import create_data_explorer_agent
from agents.statistic_agent import create_data_statistic_agent
from agents.cluster_agent import create_data_cluster_agent
from agents.visualization_agent import create_visualization_agent
from agents.hypothesis_agent import create_hypothesis_generator_agent
from agents.reasoner_agent import create_reasoner_agent
from agents.review_agent import create_quality_review_agent
from agents.synthesis_agent import create_synthesis_agent

from agents.feature_engineering_agent import create_feature_engineering_agent
from agents.model_selection_agent import create_model_selection_agent
from agents.model_training_agent import create_model_training_agent
from agents.model_evaluation_agent import create_model_evaluation_agent
from agents.prediction_agent import create_prediction_agent
from agents.planner_agent import create_planner_agent 

import json
import re 

class WorkflowManager:
    def __init__(self, language_models, working_directory):
        self.language_models = language_models
        self.working_directory = working_directory
        self.members = [
            "Planner", "DataExplorer", "DataStatistic", "DataCluster", "DataVisualization",
            "FeatureEngineering", "ModelSelection", "ModelTraining","ModelEvaluation", "Prediction",
            "HypothesisGenerator", "Reasoner", "QualityReview", "Synthesis"
        ]
        self.agents = self.create_agents()
        self.workflow = self.setup_dynamic_workflow()

    def create_agents(self):
        """Create all system agents for the workflow"""
        llm = self.language_models["llm"]
        power_llm = self.language_models["power_llm"]
        
        agents = {
            "planner": create_planner_agent(power_llm, self.members, self.working_directory),
            "explorer": create_data_explorer_agent(llm, self.members, self.working_directory),
            "statistic": create_data_statistic_agent(llm, self.members, self.working_directory),
            "cluster": create_data_cluster_agent(llm, self.members, self.working_directory),
            "visualizer": create_visualization_agent(llm, self.members, self.working_directory),
            "feature_engineer": create_feature_engineering_agent(power_llm, self.members, self.working_directory),
            "model_selection": create_model_selection_agent(power_llm, self.members, self.working_directory),
            "model_training": create_model_training_agent(power_llm, self.members, self.working_directory),
            "model_evaluation": create_model_evaluation_agent(power_llm, self.members, self.working_directory),
            "prediction": create_prediction_agent(power_llm, self.members, self.working_directory),
            "hypothesis": create_hypothesis_generator_agent(llm, self.members, self.working_directory),
            "reasoner": create_reasoner_agent(power_llm, self.members, self.working_directory),
            "quality": create_quality_review_agent(power_llm, self.members, self.working_directory),
            "synthesis": create_synthesis_agent(power_llm, self.members, self.working_directory),
        }
        return agents

    def _route_next_step(self, state: State) -> str:
        """
        Determines the next agent to execute based on the Planner's decision.
        This function will be used as the conditional edge.
        """
        sender = state.get("sender", "")
        last_message = state["messages"][-1]
        
        if sender != "Planner" and "Error" in last_message.content:
            print(f"Error detected from {sender}. Routing back to Planner for re-evaluation.")
            return "Planner"
        elif sender == "Planner":
            planner_output_str = last_message.content
            planner_output = None
            extracted_json_str = None

            json_array_pattern = re.compile(r"\[\s*(?:\{.*?\})\s*(?:,\s*\{.*?\})*\s*\]", re.DOTALL)
            
            match = json_array_pattern.search(planner_output_str)

            if match:
                extracted_json_str = match.group(0) 
                try:
                    planner_output = json.loads(extracted_json_str)
                    print(f"Successfully extracted and parsed JSON from Planner output.")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse extracted JSON: {e}. Extracted string: {extracted_json_str[:200]}...")
            
            if planner_output and isinstance(planner_output, list) and len(planner_output) > 0 and "Agent" in planner_output[0]:
                if planner_output[0]["Agent"] == "END":
                    print("Planner signaled END. Ending workflow.")
                    return "END"
                next_agent_name = planner_output[0]["Agent"]
                print(f"Planner decided next agent: {next_agent_name}")
                return next_agent_name
            else:
                print(f"Planner output not valid JSON array or 'Agent' key missing. Rerouting to Planner. Raw output: {planner_output_str[:500]}...")
                return "Planner"
        elif sender == "":
            print("Initial state or unhandled state. Routing to Planner.")
            return "Planner"
        else:
            print(f"Agent {sender} finished successfully. Routing back to Planner for next step decision.")
            return "Planner"


    def setup_dynamic_workflow(self):
        """Set up the dynamic workflow graph using Planner Agent."""
        graph = StateGraph(State)

        graph.add_node("Planner", lambda state: agent_node(state, self.agents["planner"], "Planner"))
        graph.add_node("DataExplorer", lambda state: agent_node(state, self.agents["explorer"], "DataExplorer"))
        graph.add_node("DataStatistic", lambda state: agent_node(state, self.agents["statistic"], "DataStatistic"))
        graph.add_node("DataCluster", lambda state: agent_node(state, self.agents["cluster"], "DataCluster"))
        graph.add_node("DataVisualization", lambda state: agent_node(state, self.agents["visualizer"], "DataVisualization"))
        graph.add_node("FeatureEngineering", lambda state: agent_node(state, self.agents["feature_engineer"], "FeatureEngineering"))
        graph.add_node("ModelSelection", lambda state: agent_node(state, self.agents["model_selection"], "ModelSelection"))
        graph.add_node("ModelTraining", lambda state: agent_node(state, self.agents["model_training"], "ModelTraining"))
        graph.add_node("ModelEvaluation", lambda state: agent_node(state, self.agents["model_evaluation"], "ModelEvaluation"))
        graph.add_node("Prediction", lambda state: agent_node(state, self.agents["prediction"], "Prediction"))
        graph.add_node("HypothesisGenerator", lambda state: agent_node(state, self.agents["hypothesis"], "HypothesisGenerator"))
        graph.add_node("Reasoner", lambda state: agent_node(state, self.agents["reasoner"], "Reasoner"))
        graph.add_node("QualityReview", lambda state: agent_node(state, self.agents["quality"], "QualityReview"))
        graph.add_node("Synthesis", lambda state: agent_node(state, self.agents["synthesis"], "Synthesis"))

        graph.add_edge(START, "Planner")

        for agent_name in [
            "DataExplorer", "DataStatistic", "DataCluster", "DataVisualization",
            "FeatureEngineering", "ModelSelection", "ModelTraining", "ModelEvaluation", "Prediction",
            "HypothesisGenerator", "Reasoner", "QualityReview", "Synthesis"]:
            graph.add_edge(agent_name, "Planner")

        graph.add_conditional_edges(
            "Planner",
            self._route_next_step,
            {
                "DataExplorer": "DataExplorer",
                "DataStatistic": "DataStatistic",
                "DataCluster": "DataCluster",
                "DataVisualization": "DataVisualization",
                "FeatureEngineering": "FeatureEngineering",
                "ModelSelection": "ModelSelection",
                "ModelTraining": "ModelTraining",
                "ModelEvaluation": "ModelEvaluation",
                "Prediction": "Prediction",
                "HypothesisGenerator": "HypothesisGenerator",
                "Reasoner": "Reasoner",
                "QualityReview": "QualityReview",
                "Synthesis": "Synthesis",
                "END": END,
                "Planner": "Planner",
            }
        )

        return graph.compile()

    def get_graph(self):
        return self.workflow