"""
eda_main.py

This script contains the main function that orchestrates the data analysis
workflow by utilizing helper functions from eda_utils.py.

Author: Jake van Almelo & GPT-4
Date: 2024-09-24
"""

import pandas as pd
import logging
from eda_utils import (
    perform_missing_value_analysis,
    perform_outlier_detection,
    detect_variable_types,
    analyze_numeric_vs_numeric,
    analyze_numeric_vs_categorical,
    analyze_categorical_vs_numeric,
    analyze_categorical_vs_categorical,
)
from tqdm import tqdm  # Import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def generate_data_report(df: pd.DataFrame, target_variable: str) -> None:
    """
    Generates a comprehensive data report for the given dataframe and target variable.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    target_variable (str): The name of the target variable in the dataset.

    Returns:
    None
    """
    try:
        logging.info("Starting data quality assessment...")
        # Data Quality Assessment
        perform_missing_value_analysis(df)
        perform_outlier_detection(df)

        logging.info("Performing descriptive statistics...")
        # Descriptive Statistics
        variable_types = detect_variable_types(df)
        numerical_vars = variable_types['numerical']
        categorical_vars = variable_types['categorical']

        # Descriptive statistics for numerical variables
        if numerical_vars:
            logging.info("Descriptive statistics for numerical variables:")
            logging.info(f"\n{df[numerical_vars].describe()}\n")

        # Descriptive statistics for categorical variables
        if categorical_vars:
            logging.info("Descriptive statistics for categorical variables:")
            # Wrap the loop with tqdm for progress feedback
            for var in tqdm(categorical_vars, desc="Processing Categorical Variables"):
                logging.info(f"\nVariable: {var}")
                logging.info(f"\n{df[var].value_counts()}\n")

        logging.info("Performing bivariate analysis...")
        # Bivariate Analysis
        perform_bivariate_analysis(df, target_variable)

        logging.info("Data report generation completed.")
    except Exception as e:
        logging.error(f"Error in generate_data_report: {e}")


def perform_bivariate_analysis(df: pd.DataFrame, target_variable: str) -> None:
    """
    Performs bivariate analysis between all variables and the target variable.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    target_variable (str): The name of the target variable.

    Returns:
    None
    """
    try:
        variable_types = detect_variable_types(df)
        numerical_vars = variable_types['numerical']
        categorical_vars = variable_types['categorical']

        # Remove target variable from lists if present
        if target_variable in numerical_vars:
            numerical_vars.remove(target_variable)
        if target_variable in categorical_vars:
            categorical_vars.remove(target_variable)

        # Determine type of target variable
        if target_variable in df.select_dtypes(include=['int64', 'float64']).columns:
            target_type = 'numerical'
        else:
            target_type = 'categorical'

        # Iterate over variables and perform analysis
        # Wrap the loop with tqdm for progress feedback
        for var in tqdm(df.columns, desc="Bivariate Analysis"):
            if var == target_variable:
                continue

            logging.info(f"\nAnalyzing {var} vs {target_variable}")

            # Determine variable type
            if var in numerical_vars and target_type == 'numerical':
                analyze_numeric_vs_numeric(df, var, target_variable)
            elif var in numerical_vars and target_type == 'categorical':
                analyze_numeric_vs_categorical(df, var, target_variable)
            elif var in categorical_vars and target_type == 'numerical':
                analyze_categorical_vs_numeric(df, var, target_variable)
            elif var in categorical_vars and target_type == 'categorical':
                analyze_categorical_vs_categorical(df, var, target_variable)
    except Exception as e:
        logging.error(f"Error in perform_bivariate_analysis: {e}")


if __name__ == "__main__":
    # Example usage
    try:
        # Load your dataset
        df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

        # Set your target variable
        target_variable = 'Churn'  # Replace with your actual target variable

        # Call the main function
        generate_data_report(df, target_variable)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
