from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI 
from typing import List
from langchain.tools import tool
import os
from logger import setup_logger
from core.state import NoteState
from langchain.output_parsers import PydanticOutputParser

logger = setup_logger()

@tool
def list_directory_contents(directory: str = './storage/') -> str:
    """Lists the contents of the specified directory."""
    try:
        from load_cfg import WORKING_DIRECTORY
        logger.info(f"Listing contents of directory: {WORKING_DIRECTORY}")
        contents = os.listdir(WORKING_DIRECTORY)
        logger.debug(f"Directory contents: {contents}")
        return f"Directory contents:\n" + "\n".join(contents)
    except Exception as e:
        logger.error(f"Error listing directory contents: {str(e)}")
        return f"Error listing directory contents: {str(e)}"

def create_agent(
    # llm: ChatGoogleGenerativeAI,
    llm: ChatOpenAI,
    tools: list,
    system_message: str,
    team_members: list[str],
    working_directory: str = './storage/'
) -> AgentExecutor:
    """Creates an agent with the given language model, tools, and system message."""
    logger.info("Creating agent")
    
    tools = list(tools)
    if list_directory_contents not in tools:
        tools.append(list_directory_contents)

    tool_names = ", ".join([t.name for t in tools])
    team_members_str = ", ".join(team_members)
    initial_directory_contents = list_directory_contents.invoke(working_directory)

    system_prompt_text = (
        "You are a specialized AI assistant in a data analysis team, operating as part of a multi-agent workflow."
        "Your primary goal is to execute your specific task and provide a clear output for the next agent in the chain."
        "If you can't fully complete a task, explain what you've done and what's needed next."
        "Always aim for accurate and clear outputs. "
        f"You have access to the following tools: {tool_names}. "
        f"Your specific role: {system_message}\n"
        "Work autonomously according to your specialty, using the tools available to you. "
        "Do not ask for clarification."
        f"You are one of the following team members: {team_members_str}.\n"
        f"The initial contents of your working directory are:\n{initial_directory_contents}\n"
        "Use the ListDirectoryContents tool to check for updates in the directory contents when needed."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder(variable_name="messages"),
        ("ai", "Context from previous steps:\n"
             "Input Data Path: {{datapath}}\n"
             "EDA Report: {{eda_report}}\n"
             "Statistics Report: {{statistic_report}}\n"
             "Clustering Report: {{cluster_report}}\n"
             "Visualization Report: {{visualization_report}}\n"
             "Feature Engineering Report: {{feature_engineering_report}}\n"
             "Transformed Data Path: {{transformed_datapath}}\n"
             "Model Selection Report: {{model_selection_report}}\n"
             "Model Training Report: {{model_training_report}}\n"
             "Trained Model Path: {{trained_model_path}}\n"
             "Model Evaluation Report: {{model_evaluation_report}}\n"
             "Prediction Report: {{prediction_report}}\n"
             "Hypothesis Report: {{hypothesis_report}}\n"
             "Reasoning Report: {{reasoning_report}}\n"
             "Total Summary Report: {{total_summary_report}}\n"
             "Final Report: {{final_report}}\n"
             "Current Sender: {{sender}}"
            ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    
    logger.info("Agent created successfully")
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def create_note_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> AgentExecutor:
    """Creates a Note Agent that updates the entire state."""
    logger.info("Creating note agent")
    parser = PydanticOutputParser(pydantic_object=NoteState)
    output_format = parser.get_format_instructions()
    
    prompt_str = (
        system_prompt + 
        "\n\nPlease format your entire response as a single, valid JSON object that conforms to the following Pydantic schema:\n" + 
        output_format
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_str),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    logger.info("Note agent created successfully")
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

logger.info("Agent creation module initialized")