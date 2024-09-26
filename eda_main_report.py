# eda_main_report.py
"""
eda_main.py

This script contains the main function that orchestrates the data analysis
workflow by utilizing helper functions from eda_utils.py.

Author: Jake van Almelo & GPT-4
Date: 2024-09-24
"""

import os
import pandas as pd
import logging
from eda_utils_report import (
    perform_missing_value_analysis,
    perform_outlier_detection,
    detect_variable_types,
    analyze_numeric_vs_numeric,
    analyze_numeric_vs_categorical,
    analyze_categorical_vs_numeric,
    analyze_categorical_vs_categorical,
)
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Create a directory to save the report
if not os.path.exists('reports'):
    os.makedirs('reports')


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
        missing_value_outputs = perform_missing_value_analysis(df)
        outlier_detection_outputs = perform_outlier_detection(df)

        logging.info("Performing descriptive statistics...")
        # Descriptive Statistics
        variable_types = detect_variable_types(df)
        numerical_vars = variable_types['numerical']
        categorical_vars = variable_types['categorical']

        # Descriptive statistics for numerical variables
        numerical_describe = None
        if numerical_vars:
            logging.info("Descriptive statistics for numerical variables:")
            numerical_describe = df[numerical_vars].describe()
            logging.info(f"\n{numerical_describe}\n")

        # Descriptive statistics for categorical variables
        categorical_describe = {}
        if categorical_vars:
            logging.info("Descriptive statistics for categorical variables:")
            for var in tqdm(categorical_vars, desc="Processing Categorical Variables"):
                value_counts = df[var].value_counts()
                categorical_describe[var] = value_counts
                logging.info(f"\nVariable: {var}")
                logging.info(f"\n{value_counts}\n")

        logging.info("Performing bivariate analysis...")
        # Bivariate Analysis
        bivariate_outputs = perform_bivariate_analysis(df, target_variable)

        logging.info("Compiling the report...")
        # Compile the report
        compile_report(
            missing_value_outputs,
            outlier_detection_outputs,
            numerical_describe,
            categorical_describe,
            bivariate_outputs,
            target_variable
        )

        logging.info("Data report generation completed.")
    except Exception as e:
        logging.error(f"Error in generate_data_report: {e}")


def perform_bivariate_analysis(df: pd.DataFrame, target_variable: str) -> list:
    """
    Performs bivariate analysis between all variables and the target variable.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    target_variable (str): The name of the target variable.

    Returns:
    list: A list of dictionaries containing analysis outputs for each variable.
    """
    outputs = []
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
        for var in tqdm(df.columns, desc="Bivariate Analysis"):
            if var == target_variable:
                continue

            logging.info(f"\nAnalyzing {var} vs {target_variable}")

            analysis_result = {'variable': var, 'target': target_variable}

            # Determine variable type
            if var in numerical_vars and target_type == 'numerical':
                result = analyze_numeric_vs_numeric(df, var, target_variable)
                analysis_result['type'] = 'numeric_vs_numeric'
            elif var in numerical_vars and target_type == 'categorical':
                result = analyze_numeric_vs_categorical(df, var, target_variable)
                analysis_result['type'] = 'numeric_vs_categorical'
            elif var in categorical_vars and target_type == 'numerical':
                result = analyze_categorical_vs_numeric(df, var, target_variable)
                analysis_result['type'] = 'categorical_vs_numeric'
            elif var in categorical_vars and target_type == 'categorical':
                result = analyze_categorical_vs_categorical(df, var, target_variable)
                analysis_result['type'] = 'categorical_vs_categorical'
            else:
                result = None
                analysis_result['type'] = 'unknown'

            analysis_result['result'] = result
            outputs.append(analysis_result)
    except Exception as e:
        logging.error(f"Error in perform_bivariate_analysis: {e}")
    return outputs


def compile_report(
    missing_value_outputs,
    outlier_detection_outputs,
    numerical_describe,
    categorical_describe,
    bivariate_outputs,
    target_variable
):
    """
    Compiles all analysis outputs into an HTML report using Jinja2 templates.

    Parameters:
    All outputs from the analysis functions.

    Returns:
    None
    """
    try:
        # Set up Jinja2 environment
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('report_template.html')

        # Render the template with the analysis outputs
        html_out = template.render(
            missing_value_outputs=missing_value_outputs,
            outlier_detection_outputs=outlier_detection_outputs,
            numerical_describe=numerical_describe,
            categorical_describe=categorical_describe,
            bivariate_outputs=bivariate_outputs,
            target_variable=target_variable
        )

        # Save the report
        report_filename = 'reports/analysis_report.html'
        with open(report_filename, 'w') as f:
            f.write(html_out)

        logging.info(f"Report saved to {report_filename}")
    except Exception as e:
        logging.error(f"Error in compile_report: {e}")


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
