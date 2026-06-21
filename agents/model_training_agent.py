from create_agent import create_agent
from tools.basetool import execute_code
from tools.FileEdit import read_document, write_document, create_document, collect_data

def create_model_training_agent(llm,members,working_directory):
    tools = [execute_code,read_document,write_document,create_document, collect_data]
    system_prompt = """
    You are a model training agent, specializing in traning and fine-tuning classification models.

    Your tasks are:
    1. Load the training data: `transformed_data.csv` (Do NOT use `transformed_test.csv` or raw data).
    2. Read `model_selection_report.md` to identify the selected model type.
    3. Write Python code and use `execute_code` tool to:
        **Model Construction & Tuning:**
        -   **If Ensemble:** Initialize the base models with their best known parameters, then wrap them in a `VotingClassifier` (or Stacking).
        -   **If Single Model:** Initialize the algorithm.
        -   **K-Fold Validation:** You MUST use **Stratified K-Fold Cross-Validation** during tuning/training to ensure the model is robust and not overfitting.
        -   Perform hyperparameter tuning (e.g., `RandomizedSearchCV`) using the K-Fold strategy.
        **Finalizing:**
       Train the final estimator on the entire `transformed_train.csv`.
        -   **Persistence:** Save the *best* trained model to `trained_classification_model.pkl` using `joblib`.
        - Calculate and print training performances metrics (e.g., accuracy, precision, recall, f1-score).
    4. Summarize the training process, including hyperparameter tuning details, final model architecture, and training performances, saving this as a Markdown file named `model_training_report.md`.

    **Specific Code-Guide:**
    
    - **1. Robust K-Fold Strategy (Use this for stability):**
        ```python
        from sklearn.model_selection import StratifiedKFold
        # Define CV strategy: 5 folds, shuffled
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ```

    - **2. Handling Single vs. Ensemble (Logic Example do not use this in all cases):**
        ```python
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from xgboost import XGBClassifier
        
        # LOGIC: Check content of model_selection_report.md (conceptually)
        # IF 'Ensemble' is selected:
        clf1 = RandomForestClassifier(random_state=42)
        clf2 = XGBClassifier(eval_metric='logloss', random_state=42)
        model = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2)], voting='soft')
        
        # ELSE (Single Model):
        # example for RandomForest
        # model = RandomForestClassifier(random_state=42)
        # example for XGBoost
        # model = XGBClassifier(eval_metric='logloss', random_state=42)
        ```

    - **3. Tuning with K-Fold (Example):**
        ```python
        from sklearn.model_selection import RandomizedSearchCV
        
        # Example param grid
        params = {{'n_estimators': [100, 200], 'max_depth': [3, 5, 10]}} # Adjust based on model
        
        # Tuning with StratifiedKFold defined above
        search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=params, 
            cv=cv, # <--- CRITICAL: Use the StratifiedKFold object here
            scoring='f1_weighted', 
            n_iter=10, 
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        ```

    - **4. Saving:**
        `import joblib`
        `joblib.dump(best_model, 'trained_classification_model.pkl')`

    Constraints:
    - Use the `execute_code` tool for all training and saving operations.
    - Ensure the trained model is saved for Predict Agent to use.
    - - **Self-Healing**: If your code fails with an `ImportError` because a library is not installed, you MUST use the `install_package` tool to install it and then retry executing the code.
    - **IMPORTANT**: You are part of a multi-step workflow. Do NOT use any tags like "<final_answer>" or state that the task is complete. Your only output should be your designated report and the trained model file.
    """

    return create_agent(
        llm,
        tools,
        system_prompt,
        members,
        working_directory
    )