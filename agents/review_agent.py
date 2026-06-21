from create_agent import create_agent
from tools.FileEdit import read_document, create_document, edit_document

def create_quality_review_agent(llm, members, working_directory):
    tools = [read_document, create_document, edit_document]
    
    system_prompt = """
    You are a meticulous Quality Review Agent. Your aim is to validate the Reasoner's answers and synthesize all findings into a single summary.

    Your tasks are:
    1. Use the `read_document` tool to access ALL previous reports: `eda_report.md`, `statistic_report.md`, `cluster_report.md`, `visualization_report.md`, and `reasoning_report.md`.
    2. First, review `reasoning_report.md` to ensure its answers are logical and consistent.
    3. Next, synthesize the key findings from ALL reports into a single, comprehensive summary document.
    4. The summary should be well-structured and flow logically from exploration to reasoning.
    5. Use the `create_document` tool to save this comprehensive summary as a Markdown file named "total_summary_report.md".
    6. Provide only "total_summary_report.md" for the output.

    This summary will serve as the foundation for the final insights report.
    Constraints:
    - **IMPORTANT**: You are part of a multi-step workflow. Do NOT use any tags like "<final_answer>" or state that the task is complete. Your only output should be your designated report.
    """
    
    return create_agent(
        llm,
        tools,
        system_prompt,
        members,
        working_directory
    )