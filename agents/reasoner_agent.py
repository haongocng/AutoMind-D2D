from create_agent import create_agent
from tools.FileEdit import read_document, create_document, collect_data
from tools.basetool import execute_code

def create_reasoner_agent(llm, members, working_directory):
    tools = [read_document, create_document, collect_data]
    
    system_prompt = """
    You are a Reasoner Agent. Your task is to answer the specific questions raised in the hypothesis report.

    Your tasks are:
    1.  Use the `read_document` tool to read the questions from `hypothesis_report.md`.
    2.  To formulate your answers, you can use `read_document` to access any other previous reports (`eda_report.md`, `statistic_report.md`, etc.) or `collect_data` to re-examine the original dataset.
    3.  Formulate clear, data-driven answers to each question.
    4.  Use the `create_document` tool to save your answers in a Markdown file named `reasoning_report.md`.

    Constraints:
    - Address each question from the hypothesis report systematically.
    - Your answers must be supported by the data and the prior analyses.
    - **IMPORTANT**: You are part of a multi-step workflow. Do NOT use any tags like "<final_answer>" or state that the task is complete. Your only output should be your designated report.
    """
    
    return create_agent(
        llm,
        tools,
        system_prompt,
        members,
        working_directory
    )