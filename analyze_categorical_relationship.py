import pandas as pd
import scipy.stats as stats

def analyze_categorical_relationship(df, col1, col2):
    """
    This function analyzes the relationship between two categorical variables in a dataframe using the 
    Chi-Square test of independence. It determines whether the distribution of one categorical variable 
    is independent of the other.

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
        The function prints the results of the Chi-Square test, including the test statistic, p-value, 
        and an interpretation of whether the two variables are significantly associated.

    Workflow:
    --------
    1. Verifies that both columns are of categorical data types (either 'object' or 'category').
    2. Creates a contingency table to represent the frequency distribution of the variables.
    3. Checks for expected frequencies to ensure they are appropriate for the Chi-Square test.
    4. Performs the Chi-Square test of independence.
    5. Outputs the Chi-Square statistic, p-value, and an interpretation of whether the variables are related.

    Raises:
    ------
    AssertionError
        If the data types of the provided columns are not categorical.
    ValueError
        If the contingency table has expected frequencies less than 5, which may invalidate the Chi-Square test.
    Exception
        Catches and prints any unexpected errors that occur during the function execution.

    Example:
    --------
    >>> compare_categorical_relationship(df, 'Gender', 'Churn')
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

        # Check expected frequencies
        print("\nChecking expected frequencies...")
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        if (expected < 5).any():
            raise ValueError("One or more expected frequencies are below 5, which may invalidate the Chi-Square test.")

        # Perform Chi-Square test
        print("\nPerforming Chi-Square test of independence...")
        print(f"Chi-Square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4f}")
        print(f"Degrees of Freedom: {dof}")

        # Interpretation
        if p < 0.05:
            print(f"\nThe variables '{col1}' and '{col2}' are significantly related (reject null hypothesis).")
        else:
            print(f"\nThe variables '{col1}' and '{col2}' are not significantly related (fail to reject null hypothesis).")

    except AssertionError as e:
        print(f"Assertion Error: {e}")
    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example of function usage:
# compare_categorical_relationship(df, 'Gender', 'Churn')