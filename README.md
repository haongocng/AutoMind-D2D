# Multi-Agent System for Automated Data Analysis
---
## INTRODUCTIONS:
This project implements a multi-agent system using Python and the LangGraph library to perform a comprehensive, step-by-step data analysis on a given dataset. The system is designed to automate the entire analytical workflow, from initial data exploration to final insight synthesis, by leveraging a team of specialized AI agents.

---
## INSTRUCTIONS:

### 1. Clone project

```bash
git clone https://github.com/Orjnzz/Multi-agents-for-DA.git
cd Multi-agents-for-DA
```

### 2. Set up environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3. Create and activate Conda environment:
```bash
conda create -n data_assistant
conda activate data_assistant
```
Note: If system doesn't have Anaconda, only need to install Miniconda for experience

### 4. Install
```bash
pip install -r requirements.txt
```

### 5. Input data
Put your data (csv) from dataset folder into storage
(example: heart_test.csv & heart_train.csv in storage)

### 6. Prompt
Change the prompt for specific task
(example: Currently for heart disease prediction in main.py)

### 7. Run
```bash
python main.py
```
### Note: rename "sample.env" to ".env" file and fill api_key of LLM, CONDA environment