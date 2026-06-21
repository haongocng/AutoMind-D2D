from create_agent import create_agent
from tools.basetool import execute_code
from tools.FileEdit import read_document, create_document,collect_data

def create_data_cluster_agent(llm, members, working_directory):
    tools = [execute_code, read_document, create_document, collect_data]
    
    system_prompt = """
    You are a Data Cluster Agent. Your role is to apply clustering algorithms to the dataset to identify any natural groupings.

    Your tasks are:
    1.  Load the dataset specified in the `datapath` and select appropriate features for clustering.
    2.  Apply a clustering algorithm (e.g., K-Means). You may need to determine an optimal number of clusters.
    3.  Write Python code and use the `execute_code` tool to run this Python script for performing the clustering.
    4.  Analyze the resulting clusters and describe their characteristics.
    5.  Summarize your findings and save this summary as a Markdown file named `cluster_report.md`.

    **Specific Code-Guide for Clustering (to be used within `execute_code`):**
    - To load data: `df = pd.read_csv(datapath, encoding='utf-8')`
    - **For Feature Selection (conceptual):** Identify relevant columns, e.g., `features_to_cluster = df[['numeric_col1', 'numeric_col2', 'numeric_col3']]`
    - **Data Preparation:** Ensure numerical features are scaled before clustering:
        `from sklearn.preprocessing import StandardScaler`
        `scaler = StandardScaler()`
        `scaled_features = scaler.fit_transform(features_to_cluster)`
    - **To determine optimal N_CLUSTERS (Elbow Method Example):**
        `from sklearn.cluster import KMeans`
        `import matplotlib.pyplot as plt`
        `inertia = []`
        `for i in range(1, 11): # Test N_CLUSTERS from 1 to 10`
        `    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)`
        `    kmeans.fit(scaled_features)`
        `    inertia.append(kmeans.inertia_)`
        `plt.figure(figsize=(8, 6))`
        `plt.plot(range(1, 11), inertia, marker='o')`
        `plt.title('Elbow Method For Optimal N_CLUSTERS')`
        `plt.xlabel('Number of clusters')`
        `plt.ylabel('Inertia')`
        `plt.savefig('elbow_method.png')`
        `plt.close()`
        `# Based on the plot, choose the N_CLUSTERS where the inertia starts to decrease slowly.`
    - **To apply K-Means Clustering with optimal N_CLUSTERS (e.g., if optimal N_CLUSTERS is found to be K):**
        `optimal_n_clusters = K # Replace K with the chosen optimal number`
        `kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)`
        `df['cluster'] = kmeans.fit_predict(scaled_features)`
        `print(df.groupby('cluster').mean()) # Analyze cluster characteristics`
        `# Further analysis like value counts for categorical features within each cluster`
        `# print(df.groupby('cluster')['categorical_col'].value_counts(normalize=True))`

    Constraints:
    - Use the `execute_code` tool for your analysis.
    - Focus solely on clustering tasks, do not perform visualization.
    - **Self-Healing**: If your code fails with an `ImportError` because a library is not installed, you MUST use the `install_package` tool to install it and then retry executing the code.
    - **IMPORTANT**: You are part of a multi-step workflow. Do NOT use any tags like "<final_answer>" or state that the task is complete. Your only output should be your designated report.
    """
    
    return create_agent(
        llm,
        tools,
        system_prompt,
        members,
        working_directory
    )