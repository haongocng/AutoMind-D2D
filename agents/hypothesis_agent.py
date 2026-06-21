from create_agent import create_agent
from tools.FileEdit import read_document, create_document, collect_data

def create_hypothesis_generator_agent(llm, members, working_directory):
    tools = [read_document, create_document, collect_data]
    
    system_prompt = """
    You are a Hypothesis Generator Agent. All initial analyses (EDA, stats, clustering, visualization) are complete and your role is to formulate key questions based on all prior analysis.

    Your tasks are:
    1.  Use the `read_document` tool to access the content of all previous reports (`eda_report.md`, `statistic_report.md`, `cluster_report.md`, and `visualization_report.md`).
    2.  Based on your understanding, generate 3 to 5 specific, answerable questions that can be clarified from the data. These questions are your hypotheses.
    3.  The questions should aim to uncover deeper insights or validate assumptions.
    4.  Use the `create_document` tool to save these questions in a numbered list in a Markdown file named `hypothesis_report.md`.
    5.  Provide only "hypothesis_report.md" for the output.
    
    Constraints:
    - Do not try to answer the questions yourself. Your sole purpose is to ask them.
    - Ensure your questions are grounded in the findings of the previous reports.
    - **IMPORTANT**: You are part of a multi-step workflow. Do NOT use any tags like "<final_answer>" or state that the task is complete. Your only output should be your designated report.
    """
    
    return create_agent(
        llm,
        tools,
        system_prompt,
        members,
        working_directory
    )