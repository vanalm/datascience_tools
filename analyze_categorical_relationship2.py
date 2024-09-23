import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

def analyze_categorical_relationship(df, col1, col2):
    """
    This function analyzes the relationship between two categorical variables in a dataframe using various 
    statistical methods, probability calculations, and visualizations. It performs the Chi-Square test of 
    independence, generates joint, marginal, and conditional probabilities, and creates visual summaries 
    with interpretations.

    Parameters:
    ----------
    df : pandas.DataFrame
        The dataframe containing the data to be analyzed.
    col1 : str
        The name of the first categorical column (e.g., 'Gender', 'Churn').
    col2 : str
        The name of the second categorical column (e.g., 'Contract', 'Dependents').

    Returns:
    -------
    None
        The function prints the results of statistical tests, generates summary graphs, and provides 
        a simple interpretation of the relationship between the two categorical variables.

    Workflow:
    --------
    1. Verifies that both columns are of categorical data types.
    2. Creates and displays a contingency table for the two variables.
    3. Performs the Chi-Square test of independence and prints the results.
    4. Calculates and prints joint, marginal, and conditional probabilities.
    5. Generates visualizations: Heatmap, Count Plot, Stacked Bar Plot, and Mosaic Plot.
    6. Provides a simple summary and interpretation of findings.

    Raises:
    ------
    AssertionError
        If the data types of the provided columns are not categorical.
    Exception
        Catches and prints any unexpected errors that occur during the function execution.

    Example:
    --------
    >>> analyze_categorical_relationship(df, 'Gender', 'Churn')
    Checking data types for columns 'Gender' and 'Churn'...
    ...
    The variables 'Gender' and 'Churn' are not significantly related (fail to reject null hypothesis).
    """

    try:
        # Verify data types
        print(f"\nChecking data types for columns '{col1}' and '{col2}'...")
        assert pd.api.types.is_categorical_dtype(df[col1]) or pd.api.types.is_object_dtype(df[col1]), f"{col1} must be categorical."
        assert pd.api.types.is_categorical_dtype(df[col2]) or pd.api.types.is_object_dtype(df[col2]), f"{col2} must be categorical."

        # Create a contingency table
        print(f"\nCreating a contingency table for '{col1}' and '{col2}'...")
        contingency_table = pd.crosstab(df[col1], df[col2])
        print("Contingency Table:\n", contingency_table)

        # Perform Chi-Square test of independence
        print("\nPerforming Chi-Square test of independence...")
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Chi-Square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4f}")
        print(f"Degrees of Freedom: {dof}")

        # Interpretation of Chi-Square result
        if p < 0.05:
            print(f"\nThe variables '{col1}' and '{col2}' are significantly related (reject null hypothesis).")
        else:
            print(f"\nThe variables '{col1}' and '{col2}' are not significantly related (fail to reject null hypothesis).")

        # Calculate joint, marginal, and conditional probabilities
        print("\nCalculating joint, marginal, and conditional probabilities...")
        joint_prob = contingency_table / contingency_table.values.sum()
        marginal_prob_row = contingency_table.sum(axis=1) / contingency_table.values.sum()
        marginal_prob_col = contingency_table.sum(axis=0) / contingency_table.values.sum()
        conditional_prob = contingency_table.div(contingency_table.sum(axis=1), axis=0)
        
        print("\nJoint Probability Table:\n", joint_prob)
        print("\nMarginal Probability (Row):\n", marginal_prob_row)
        print("\nMarginal Probability (Column):\n", marginal_prob_col)
        print("\nConditional Probability (Given Row):\n", conditional_prob)

        # Visualization: Heatmap of the Contingency Table
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
        plt.title(f'Heatmap of {col1} vs {col2}')
        plt.show()

        # Visualization: Count Plot
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col1, hue=col2, data=df)
        plt.title(f'Count Plot of {col1} by {col2}')
        plt.xlabel(col1)
        plt.ylabel('Count')
        plt.show()

        # Visualization: Stacked Bar Plot
        contingency_table.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'Stacked Bar Plot of {col1} by {col2}')
        plt.xlabel(col1)
        plt.ylabel('Count')
        plt.legend(title=col2)
        plt.show()

        # Visualization: Mosaic Plot
        mosaic(df, [col1, col2])
        plt.title(f'Mosaic Plot of {col1} vs {col2}')
        plt.show()

        # Summary Interpretation
        print("\nSummary:")
        if p < 0.05:
            print(f"There is a significant relationship between '{col1}' and '{col2}'.")
            print("This means that the distribution of one variable is dependent on the other.")
        else:
            print(f"There is no significant relationship between '{col1}' and '{col2}'.")
            print("This means that the distribution of one variable is independent of the other.")

    except AssertionError as e:
        print(f"Assertion Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example of function usage:
# analyze_categorical_relationship(df, 'Gender', 'Churn')