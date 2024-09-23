



import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def compare_means(df, continuous_col, group_col):
    """

    This function takes two columns from a dataframe—a continuous variable and a categorical variable—
    and tests if the continuous variable differs significantly across the categories of the categorical variable
    
    It also verifies that the data meets the assumptions for ANOVA and performs either ANOVA or Kruskal-Wallis test
    
    more information on ANOVA here: https://www.youtube.com/watch?v=NF5_btOaCig
    
    This function analyzes the relationship between a continuous variable and a categorical grouping variable 
    in a dataframe. It checks the assumptions for performing ANOVA and, depending on the results, conducts 
    either a one-way ANOVA or a non-parametric Kruskal-Wallis test to determine if there are significant 
    differences in the continuous variable across the groups.

    Parameters:
    ----------
    df : pandas.DataFrame
        The dataframe containing the data to be analyzed.
    continuous_col : str
        The name of the column containing the continuous variable (e.g., 'tenure', 'MonthlyCharges').
    group_col : str
        The name of the column containing the categorical grouping variable (e.g., 'Contract', 'Gender').

    Returns:
    -------
    None
        The function prints the results of the assumption checks, the statistical test results, and 
        an interpretation of whether there is a significant difference between the group means or medians.

    Workflow:
    --------
    1. Verifies that the provided columns are of appropriate types (continuous and categorical).
    2. Visualizes the distribution of the continuous variable across different groups using a boxplot.
    3. Checks for normality of the continuous variable within each group using the Shapiro-Wilk test.
    4. Checks for homogeneity of variances across groups using Levene's test.
    5. Based on the assumption checks:
       - If normality and equal variances are satisfied, performs a one-way ANOVA.
       - If assumptions are not satisfied, performs the non-parametric Kruskal-Wallis test.
    6. Outputs the test statistics, p-values, and an interpretation of the results.

    Raises:
    ------
    AssertionError
        If the data types of the provided columns are not as expected or if there are fewer than three groups.
    Exception
        Catches and prints any unexpected errors that occur during the function execution.

    Example:
    --------
    >>> compare_means(df, 'tenure', 'Contract')
    Checking data types for columns 'tenure' and 'Contract'...
    ...
    The mean of the continuous variable is significantly different across groups (reject null hypothesis).

    """
    try:
        # Verify data types
        print(f"\nChecking data types for columns '{continuous_col}' and '{group_col}'...")
        assert pd.api.types.is_numeric_dtype(df[continuous_col]), f"{continuous_col} must be numeric."
        assert pd.api.types.is_categorical_dtype(df[group_col]) or pd.api.types.is_object_dtype(df[group_col]), f"{group_col} must be categorical."

        # Check unique values in the group column
        unique_groups = df[group_col].unique()
        print(f"\nNumber of unique groups in '{group_col}': {len(unique_groups)}")
        print(f"Unique values: {unique_groups}")
        
        assert len(unique_groups) >= 3, "There must be at least three groups for ANOVA."

        # Visualize distribution
        print("\nVisualizing distributions for the groups...")
        sns.boxplot(x=group_col, y=continuous_col, data=df)
        plt.title(f'Distribution of {continuous_col} by {group_col}')
        plt.xlabel(group_col)
        plt.ylabel(continuous_col)
        plt.show()

        # Check normality assumption using Shapiro-Wilk test
        print("\nChecking normality assumption using Shapiro-Wilk test for each group...")
        for group in unique_groups:
            group_data = df[df[group_col] == group][continuous_col]
            stat, p_value = stats.shapiro(group_data)
            print(f"Group '{group}' - Shapiro-Wilk test p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"Warning: The distribution of {continuous_col} in group '{group}' is not normal (p-value < 0.05).")

        # Check homogeneity of variances using Levene's test
        print("\nChecking homogeneity of variances using Levene's test...")
        group_data_list = [df[df[group_col] == group][continuous_col] for group in unique_groups]
        levene_stat, levene_p = stats.levene(*group_data_list)
        print(f"Levene's test p-value: {levene_p:.4f}")
        if levene_p < 0.05:
            print("Warning: The variances across groups are not equal (p-value < 0.05).")

        # Choose the appropriate test based on assumptions
        print("\nDetermining the appropriate statistical test based on assumptions...")
        if all(p_value > 0.05 for group in unique_groups) and levene_p > 0.05:
            print("Assumptions met for ANOVA. Performing one-way ANOVA...")
            f_stat, anova_p = stats.f_oneway(*group_data_list)
            print(f"\nANOVA results:\nF-statistic: {f_stat:.4f}, p-value: {anova_p:.4f}")
            if anova_p < 0.05:
                print("The mean of the continuous variable is significantly different across groups (reject null hypothesis).")
            else:
                print("The mean of the continuous variable is not significantly different across groups (fail to reject null hypothesis).")
        else:
            print("Assumptions not met for ANOVA. Performing Kruskal-Wallis H test...")
            h_stat, kruskal_p = stats.kruskal(*group_data_list)
            print(f"\nKruskal-Wallis H test results:\nH-statistic: {h_stat:.4f}, p-value: {kruskal_p:.4f}")
            if kruskal_p < 0.05:
                print("The median of the continuous variable is significantly different across groups (reject null hypothesis).")
            else:
                print("The median of the continuous variable is not significantly different across groups (fail to reject null hypothesis).")

    except AssertionError as e:
        print(f"Assertion Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    try:
        # Load sample data
        df = pd.read_csv('your_dataset.csv')  # Replace with your actual dataset

        # User input for columns to use
        continuous_col = input("Enter the name of the continuous variable column: ")
        group_col = input("Enter the name of the categorical group column: ")

        # Run the analysis function
        check_assumptions_and_compare_means(df, continuous_col, group_col)

    except FileNotFoundError:
        print("The file was not found. Please ensure the dataset path is correct.")
    except Exception as e:
        print(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()