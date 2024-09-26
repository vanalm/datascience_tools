"""
eda_utils_report.py

This module contains helper functions for performing data quality assessment,
descriptive statistics, bivariate analysis, and generating visualizations,
tailored for report generation.

Author: Jake van Almelo & GPT-4
Date: 2024-09-24
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Create a directory to save plots
if not os.path.exists('plots'):
    os.makedirs('plots')

def sanitize_filename(name):
    """
    Sanitizes a string to be used in a filename.

    Parameters:
    name (str): The string to sanitize.

    Returns:
    str: A sanitized string safe for filenames.
    """
    # Remove invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    return name

def perform_missing_value_analysis(df: pd.DataFrame) -> dict:
    """
    Analyzes missing values in the dataset and generates visualizations.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.

    Returns:
    dict: A dictionary containing paths to generated plots and missing values table.
    """
    outputs = {}
    try:
        # Calculate missing values
        missing_values = df.isnull().sum()
        missing_percent = 100 * missing_values / len(df)
        missing_table = pd.concat([missing_values, missing_percent], axis=1)
        missing_table.columns = ['Missing Values', 'Percentage']
        missing_table = missing_table[missing_table['Missing Values'] > 0]

        # Save missing values table
        outputs['missing_table'] = missing_table

        # Print summary table
        if not missing_table.empty:
            logging.info("Missing Values Summary:")
            logging.info(f"\n{missing_table}\n")

            # Visualize missing values with a bar chart
            plt.figure(figsize=(10, 6))
            sns.barplot(x=missing_table.index, y='Missing Values', data=missing_table)
            plt.xticks(rotation=90)
            plt.title('Missing Values per Variable')
            plt.tight_layout()
            filename_barplot = 'plots/missing_values_barplot.png'
            plt.savefig(filename_barplot)
            plt.close()
            outputs['missing_values_barplot'] = filename_barplot

            # Visualize missing data matrix
            msno.matrix(df)
            filename_matrix = 'plots/missing_data_matrix.png'
            plt.savefig(filename_matrix)
            plt.close()
            outputs['missing_data_matrix'] = filename_matrix

            # Visualize missing data heatmap
            msno.heatmap(df)
            filename_heatmap = 'plots/missing_data_heatmap.png'
            plt.savefig(filename_heatmap)
            plt.close()
            outputs['missing_data_heatmap'] = filename_heatmap
        else:
            logging.info("No missing values found in the dataset.")
    except Exception as e:
        logging.error(f"Error in perform_missing_value_analysis: {e}")
    return outputs

def perform_outlier_detection(df: pd.DataFrame) -> dict:
    """
    Detects outliers in numerical variables using the IQR method and generates visualizations.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.

    Returns:
    dict: A dictionary containing paths to generated plots and outlier counts.
    """
    outputs = {'outlier_plots': [], 'outlier_counts': {}}
    try:
        numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Use tqdm to wrap the loop for progress tracking
        for var in tqdm(numerical_vars, desc="Outlier Detection"):
            plt.figure(figsize=(12, 5))

            # Boxplot to visualize outliers
            plt.subplot(1, 2, 1)
            sns.boxplot(y=df[var])
            plt.title(f'Box Plot of {var}')

            # Histogram to visualize distribution
            plt.subplot(1, 2, 2)
            sns.histplot(df[var].dropna(), kde=True)
            plt.title(f'Distribution of {var}')

            plt.tight_layout()
            # Sanitize variable name
            var_sanitized = sanitize_filename(var)
            # Save the plot
            filename = f'plots/outlier_detection_{var_sanitized}.png'
            plt.savefig(filename)
            plt.close()
            outputs['outlier_plots'].append({'variable': var, 'plot': filename})

            # Detect outliers using IQR
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)]

            outlier_count = outliers.shape[0]
            outputs['outlier_counts'][var] = outlier_count

            logging.info(f"Outliers detected in '{var}': {outlier_count} observations")
        logging.info("Outlier detection completed. Plots saved in 'plots/' directory.")
    except Exception as e:
        logging.error(f"Error in perform_outlier_detection: {e}")
    return outputs

def detect_variable_types(df: pd.DataFrame) -> dict:
    """
    Classifies variables in the dataframe into numerical and categorical types.

    Parameters:
    df (pd.DataFrame): The dataset to classify.

    Returns:
    dict: A dictionary with keys 'numerical' and 'categorical' containing lists of variable names.
    """
    try:
        numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_vars = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Additional check for numerical variables with few unique values
        for var in numerical_vars.copy():
            if df[var].nunique() < 10:
                numerical_vars.remove(var)
                categorical_vars.append(var)

        return {'numerical': numerical_vars, 'categorical': categorical_vars}
    except Exception as e:
        logging.error(f"Error in detect_variable_types: {e}")
        return {}

def analyze_numeric_vs_numeric(df: pd.DataFrame, var1: str, var2: str) -> dict:
    """
    Performs analysis between two numerical variables.

    Parameters:
    df (pd.DataFrame): The dataset.
    var1 (str): The first numerical variable.
    var2 (str): The second numerical variable.

    Returns:
    dict: A dictionary containing paths to generated plots and statistical results.
    """
    outputs = {}
    try:
        # Drop missing values
        data = df[[var1, var2]].dropna()

        # Scatter plot with regression line
        sns.jointplot(x=var1, y=var2, data=data, kind='reg')
        plt.suptitle(f'Scatter Plot of {var1} vs {var2}', y=1.02)
        # Sanitize variable names
        var1_sanitized = sanitize_filename(var1)
        var2_sanitized = sanitize_filename(var2)
        # Save the plot
        filename = f'plots/numeric_vs_numeric_{var1_sanitized}_vs_{var2_sanitized}.png'
        plt.savefig(filename)
        plt.close()
        outputs['plot'] = filename

        # Calculate correlation coefficients
        pearson_corr, pearson_p = stats.pearsonr(data[var1], data[var2])
        spearman_corr, spearman_p = stats.spearmanr(data[var1], data[var2])

        outputs['pearson'] = {'correlation': pearson_corr, 'p_value': pearson_p}
        outputs['spearman'] = {'correlation': spearman_corr, 'p_value': spearman_p}
    except Exception as e:
        logging.error(f"Error in analyze_numeric_vs_numeric: {e}")
    return outputs

def analyze_numeric_vs_categorical(df: pd.DataFrame, numeric_var: str, categorical_var: str) -> dict:
    """
    Analyzes the relationship between a numerical variable and a categorical variable.

    Parameters:
    df (pd.DataFrame): The dataset.
    numeric_var (str): The name of the numerical variable.
    categorical_var (str): The name of the categorical variable.

    Returns:
    dict: A dictionary containing paths to generated plots and statistical results.
    """
    outputs = {}
    try:
        # Drop missing values
        data = df[[numeric_var, categorical_var]].dropna()

        # Number of categories
        categories = data[categorical_var].unique()
        num_categories = len(categories)

        # Visualizations
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=categorical_var, y=numeric_var, data=data)
        plt.title(f'Box Plot of {numeric_var} by {categorical_var}')
        # Sanitize variable names
        numeric_var_sanitized = sanitize_filename(numeric_var)
        categorical_var_sanitized = sanitize_filename(categorical_var)
        filename_boxplot = f'plots/{numeric_var_sanitized}_by_{categorical_var_sanitized}_boxplot.png'
        plt.savefig(filename_boxplot)
        plt.close()
        outputs['boxplot'] = filename_boxplot

        plt.figure(figsize=(10, 6))
        sns.violinplot(x=categorical_var, y=numeric_var, data=data)
        plt.title(f'Violin Plot of {numeric_var} by {categorical_var}')
        filename_violinplot = f'plots/{numeric_var_sanitized}_by_{categorical_var_sanitized}_violinplot.png'
        plt.savefig(filename_violinplot)
        plt.close()
        outputs['violinplot'] = filename_violinplot

        # Descriptive statistics
        desc_stats = data.groupby(categorical_var)[numeric_var].describe()
        outputs['desc_stats'] = desc_stats

        # Assumption checks
        # Normality test for each category
        normality = []
        p_values_normality = {}
        logging.info(f"Performing normality tests for each category of '{categorical_var}'...")
        for category in tqdm(categories, desc="Normality Tests"):
            stat, p = stats.shapiro(data[data[categorical_var] == category][numeric_var])
            normality.append(p > 0.05)
            p_values_normality[category] = p

        outputs['normality'] = p_values_normality

        # Homogeneity of variance
        grouped_data = [data[data[categorical_var] == cat][numeric_var] for cat in categories]
        stat, p = stats.levene(*grouped_data)
        equal_variance = p > 0.05
        outputs['homogeneity'] = {'statistic': stat, 'p_value': p, 'equal_variance': equal_variance}

        # Choose the appropriate test
        if num_categories == 2:
            # Two categories
            if all(normality) and equal_variance:
                # Independent Samples T-Test
                stat, p = stats.ttest_ind(*grouped_data)
                test_name = 'Independent Samples T-Test'
            else:
                # Mann-Whitney U Test
                stat, p = stats.mannwhitneyu(*grouped_data)
                test_name = 'Mann-Whitney U Test'
        else:
            # More than two categories
            if all(normality) and equal_variance:
                # One-Way ANOVA
                stat, p = stats.f_oneway(*grouped_data)
                test_name = 'One-Way ANOVA'
            else:
                # Kruskal-Wallis Test
                stat, p = stats.kruskal(*grouped_data)
                test_name = 'Kruskal-Wallis Test'

        outputs['test'] = {
            'name': test_name,
            'statistic': stat,
            'p_value': p
        }

        # Effect size
        if num_categories == 2:
            # Calculate Cohen's d
            mean_diff = grouped_data[0].mean() - grouped_data[1].mean()
            pooled_sd = np.sqrt((grouped_data[0].std()**2 + grouped_data[1].std()**2) / 2)
            cohen_d = mean_diff / pooled_sd
            outputs['effect_size'] = {'Cohen_d': cohen_d}
        else:
            # Calculate Eta Squared for ANOVA
            if test_name == 'One-Way ANOVA':
                ss_between = sum([len(g) * (g.mean() - data[numeric_var].mean())**2 for g in grouped_data])
                ss_total = sum((data[numeric_var] - data[numeric_var].mean())**2)
                eta_squared = ss_between / ss_total
                outputs['effect_size'] = {'Eta_squared': eta_squared}
            else:
                outputs['effect_size'] = None
    except Exception as e:
        logging.error(f"Error in analyze_numeric_vs_categorical: {e}")
    return outputs

def analyze_categorical_vs_numeric(df: pd.DataFrame, categorical_var: str, numeric_var: str) -> dict:
    """
    Analyzes the relationship between a categorical variable and a numerical variable.

    Parameters:
    df (pd.DataFrame): The dataset.
    categorical_var (str): The name of the categorical variable.
    numeric_var (str): The name of the numerical variable.

    Returns:
    dict: A dictionary containing paths to generated plots and statistical results.
    """
    # Call the numeric vs categorical analysis with swapped arguments
    return analyze_numeric_vs_categorical(df, numeric_var, categorical_var)

def analyze_categorical_vs_categorical(df: pd.DataFrame, var1: str, var2: str) -> dict:
    """
    Analyzes the relationship between two categorical variables.

    Parameters:
    df (pd.DataFrame): The dataset.
    var1 (str): The first categorical variable.
    var2 (str): The second categorical variable.

    Returns:
    dict: A dictionary containing paths to generated plots and statistical results.
    """
    outputs = {}
    try:
        # Drop missing values
        data = df[[var1, var2]].dropna()

        # Contingency table
        contingency_table = pd.crosstab(data[var1], data[var2])
        outputs['contingency_table'] = contingency_table

        logging.info(f"Contingency Table between '{var1}' and '{var2}':\n")
        logging.info(f"\n{contingency_table}\n")

        # Visualizations
        plt.figure(figsize=(10, 6))
        contingency_table.plot(kind='bar', stacked=True)
        plt.title(f'Stacked Bar Chart of {var1} vs {var2}')
        plt.xlabel(var1)
        plt.ylabel('Count')
        plt.legend(title=var2)
        plt.tight_layout()
        # Sanitize variable names
        var1_sanitized = sanitize_filename(var1)
        var2_sanitized = sanitize_filename(var2)
        # Save the plot
        filename = f'plots/categorical_vs_categorical_{var1_sanitized}_vs_{var2_sanitized}.png'
        plt.savefig(filename)
        plt.close()
        outputs['plot'] = filename

        # Chi-Squared Test
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        outputs['chi_squared'] = {
            'chi2_statistic': chi2,
            'p_value': p,
            'degrees_of_freedom': dof,
            'expected_frequencies': expected
        }

        # Calculate Cramér's V
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))
        outputs['cramers_v'] = cramers_v

        # Interpretation of Cramér's V
        if cramers_v < 0.1:
            strength = "Weak"
        elif cramers_v < 0.3:
            strength = "Moderate"
        elif cramers_v < 0.5:
            strength = "Relatively Strong"
        else:
            strength = "Strong"

        outputs['association_strength'] = strength
    except Exception as e:
        logging.error(f"Error in analyze_categorical_vs_categorical: {e}")
    return outputs
