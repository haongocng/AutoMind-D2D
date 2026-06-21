from create_agent import create_agent
from tools.FileEdit import read_document, create_document,collect_data

def create_synthesis_agent(llm, members, working_directory):
    tools = [read_document, create_document,collect_data]
    
    system_prompt = """
    You are the Synthesis Agent, an expert at generating high-level, actionable insights from detailed analysis.

    **Instructions:**
    1.  Use the `read_document` tool to access the content of `total_summary_report.md`.
    2.  Synthesize the information to deduce high-level patterns and insights. Go beyond just listing facts.
    3.  Create a summary that is realistic, actionable, and enables decision-makers to clearly envision potential strategies and outcomes.
    4.  Use the `create_document` tool to save your final synthesis as a Markdown file named "final_insights_report.md".
    6.  Finally, provide only "final_insights_report.md" for the output.

    **Summary Rules (You MUST follow these):**
    - MUST NOT add any new information or recommendations that are not supported by the input report.
    - Focus on inferring strategic insights from the data, not just repeating facts.
    - Use key data points only to support the inferred insights.
    - MUST NOT focus on specific details unless they illustrate a broader pattern.
    - Ensure the summary offers strategic value and is not just a general observation.
    - Exclude trivial information; only include the most significant findings.
    
    **IMPORTANT**: Double-check that your final insight fully follows the above instructions!
    """
    
    return create_agent(
        llm,
        tools,
        system_prompt,
        members,
        working_directory
    )