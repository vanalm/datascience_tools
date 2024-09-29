# sequential_analysis_with_target.py

"""
Sequential Attribute Analysis with Target Variable

This script performs analysis on each attribute sequentially, exploring the relationship
between each predictor variable and the target variable, and compiles the results into an HTML report.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from jinja2 import Environment, FileSystemLoader
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set up directories
REPORT_DIR = 'reports'
PLOTS_DIR = os.path.join(REPORT_DIR, 'plots')
TEMPLATE_PATH = '.'

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load dataset
# DATA_PATH = './data/house.csv'
DATA_PATH = './data/churn.csv'
df = pd.read_csv(DATA_PATH)
# Specify the target variable
# TARGET_VARIABLE = 'price'  # Replace with your target variable
TARGET_VARIABLE = 'Churn'  # Replace with your target variable

# Drop 'customerID' if present
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Handle duplicates
df.drop_duplicates(inplace=True)


# Handle missing values (you can implement imputation if needed)
df.dropna(subset=[TARGET_VARIABLE], inplace=True)

def sanitize_filename(name):
    """Sanitizes a string to be used in a filename."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.replace(' ', '_')
    return name

def detect_variable_types(df):
    """Detects numerical and categorical variables."""
    numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_vars = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    return numerical_vars, categorical_vars

def scientific_notation(value, precision=2):
    """Formats a number in scientific notation."""
    try:
        if value is None or np.isnan(value):
            return "N/A"
        elif abs(value) < 1e-4:
            return f"{value:.{precision}e}"
        else:
            return f"{value:.{precision}f}"
    except:
        return value

def analyze_numerical_variables(df, target_variable):
    """Performs efficient analysis of numerical variables."""
    outputs = {}
    numerical_vars, _ = detect_variable_types(df)

    # Remove target variable if it's numerical
    if target_variable in numerical_vars:
        numerical_vars.remove(target_variable)

    # Encode categorical target if necessary
    if not pd.api.types.is_numeric_dtype(df[target_variable]):
        df['__target_encoded'] = df[target_variable].astype('category').cat.codes
        target_for_corr = '__target_encoded'
        outputs['target_encoded'] = True
    else:
        target_for_corr = target_variable
        outputs['target_encoded'] = False

    # Correlation matrix
    corr_matrix = df[numerical_vars + [target_for_corr]].corr()
    outputs['corr_matrix'] = corr_matrix

    # Extract correlations with target variable
    target_corr = corr_matrix[target_for_corr].drop(target_for_corr).sort_values(key=lambda x: abs(x), ascending=False)
    outputs['target_corr'] = target_corr

    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    filename_corr_heatmap = os.path.join(PLOTS_DIR, 'correlation_matrix.png')
    plt.savefig(filename_corr_heatmap)
    plt.close()
    outputs['corr_heatmap'] = os.path.relpath(filename_corr_heatmap, REPORT_DIR)

    # Visualize correlations with target variable
    plt.figure(figsize=(8, len(target_corr) * 0.5))
    sns.barplot(x=target_corr.values, y=target_corr.index)
    plt.title(f'Correlation of Numerical Variables with {target_variable}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Variables')
    plt.tight_layout()
    filename_corr_barplot = os.path.join(PLOTS_DIR, 'correlation_with_target.png')
    plt.savefig(filename_corr_barplot)
    plt.close()
    outputs['corr_barplot'] = os.path.relpath(filename_corr_barplot, REPORT_DIR)

    # Generate summary statistics grouped by target variable
    group_stats = df.groupby(target_variable)[numerical_vars].agg(['mean', 'median', 'std']).round(2)
    outputs['group_stats'] = group_stats

    # Convert to HTML
    outputs['group_stats_html'] = group_stats.to_html(classes='table table-striped')

    # Identify top variables with highest correlation
    top_variables = target_corr.head(5).index.tolist()

    # Generate strip plots for top variables
    stripplots = []
    for var in top_variables:
        plt.figure(figsize=(8, 6))
        sns.stripplot(x=target_variable, y=var, data=df, jitter=True, alpha=0.5)
        plt.title(f'{var} by {target_variable}')
        plt.tight_layout()
        filename_stripplot = os.path.join(PLOTS_DIR, f'stripplot_{sanitize_filename(var)}_by_{sanitize_filename(target_variable)}.png')
        plt.savefig(filename_stripplot)
        plt.close()
        stripplots.append({'variable': var, 'plot': os.path.relpath(filename_stripplot, REPORT_DIR)})
    outputs['stripplots'] = stripplots

    # Clean up temporary column if added
    if outputs.get('target_encoded'):
        df.drop(columns=['__target_encoded'], inplace=True)

    return outputs

def analyze_numerical_attribute(df, column):
    """Performs analysis on a numerical attribute."""
    outputs = {}
    data = df[column].dropna()
    if data.empty:
        logging.warning(f"No data available for numerical analysis of {column}.")
        return outputs

    # Descriptive statistics
    desc_stats = data.describe().round(2)
    outputs['desc_stats'] = desc_stats.to_frame()

    # Convert to HTML
    outputs['desc_stats_html'] = outputs['desc_stats'].to_html(classes='table table-striped', header=False)

    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    filename_hist = os.path.join(PLOTS_DIR, f'histogram_{sanitize_filename(column)}.png')
    plt.savefig(filename_hist)
    plt.close()
    outputs['histogram'] = os.path.relpath(filename_hist, REPORT_DIR)

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data)
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.tight_layout()
    filename_boxplot = os.path.join(PLOTS_DIR, f'boxplot_{sanitize_filename(column)}.png')
    plt.savefig(filename_boxplot)
    plt.close()
    outputs['boxplot'] = os.path.relpath(filename_boxplot, REPORT_DIR)

    return outputs

def analyze_categorical_attribute(df, column):
    """Performs analysis on a categorical attribute."""
    outputs = {}
    data = df[column].dropna()
    if data.empty:
        logging.warning(f"No data available for categorical analysis of {column}.")
        return outputs

    # Frequency counts
    freq_counts = data.value_counts()
    outputs['freq_counts'] = freq_counts.to_frame()

    # Convert to HTML (limit to top 20 categories to avoid lengthy tables)
    outputs['freq_counts_html'] = outputs['freq_counts'].head(20).to_html(classes='table table-striped', header=False)
    if len(outputs['freq_counts']) > 20:
        outputs['freq_counts_html'] += '<p>Only showing top 20 categories.</p>'

    # Bar chart
    plt.figure(figsize=(10, 6))
    sns.countplot(y=data, order=freq_counts.index[:20])
    plt.title(f'Bar Chart of {column}')
    plt.xlabel('Count')
    plt.ylabel(column)
    plt.tight_layout()
    filename_barchart = os.path.join(PLOTS_DIR, f'barchart_{sanitize_filename(column)}.png')
    plt.savefig(filename_barchart)
    plt.close()
    outputs['barchart'] = os.path.relpath(filename_barchart, REPORT_DIR)

    return outputs

def analyze_relationship(df, predictor, target):
    """Analyzes the relationship between a predictor and the target variable."""
    outputs = {}
    data = df[[predictor, target]].dropna()

    if data.empty or data[predictor].nunique() < 2 or data[target].nunique() < 2:
        logging.warning(f"Not enough data variation to perform analysis between {predictor} and {target}.")
        return outputs

    predictor_type = 'numerical' if pd.api.types.is_numeric_dtype(data[predictor]) else 'categorical'
    target_type = 'numerical' if pd.api.types.is_numeric_dtype(data[target]) else 'categorical'

    # Sanitize variable names
    predictor_sanitized = sanitize_filename(predictor)
    target_sanitized = sanitize_filename(target)

    # Case A: Numerical X and Numerical Y
    if predictor_type == 'numerical' and target_type == 'numerical':
        # Scatter Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data[predictor], y=data[target])
        sns.regplot(x=data[predictor], y=data[target], scatter=False, color='red')
        plt.title(f'Scatter Plot of {predictor} vs {target}')
        plt.xlabel(predictor)
        plt.ylabel(target)
        plt.tight_layout()
        filename_scatter = os.path.join(PLOTS_DIR, f'scatter_{predictor_sanitized}_vs_{target_sanitized}.png')
        plt.savefig(filename_scatter)
        plt.close()
        outputs['scatter_plot'] = os.path.relpath(filename_scatter, REPORT_DIR)

        # Correlation
        pearson_corr, pearson_p = stats.pearsonr(data[predictor], data[target])
        spearman_corr, spearman_p = stats.spearmanr(data[predictor], data[target])

        outputs['pearson'] = {'correlation': pearson_corr, 'p_value': pearson_p}
        outputs['spearman'] = {'correlation': spearman_corr, 'p_value': spearman_p}

    # Case B: Numerical X and Categorical Y
    elif predictor_type == 'numerical' and target_type == 'categorical':
        outputs.update(analyze_numeric_vs_categorical(data, predictor, target))

    # Case C: Categorical X and Numerical Y
    elif predictor_type == 'categorical' and target_type == 'numerical':
        outputs.update(analyze_numeric_vs_categorical(data, target, predictor, swap=True))

    # Case D: Categorical X and Categorical Y
    elif predictor_type == 'categorical' and target_type == 'categorical':
        outputs.update(analyze_categorical_vs_categorical(data, predictor, target))

    else:
        logging.warning(f"Unknown variable types for {predictor} and {target}.")

    return outputs

def analyze_numeric_vs_categorical(data, numeric_var, categorical_var, swap=False):
    """Analyzes numerical vs categorical relationship."""
    outputs = {}
    categories = data[categorical_var].unique()
    num_categories = len(categories)

    if num_categories < 2:
        logging.warning(f"Not enough categories in {categorical_var} for analysis.")
        return outputs

    # Limit categories for visualization if too many
    if num_categories > 20:
        top_categories = data[categorical_var].value_counts().head(20).index
        data = data[data[categorical_var].isin(top_categories)]
        categories = top_categories
        num_categories = len(categories)
        logging.info(f"Limited to top 20 categories for '{categorical_var}'")

    # Sanitize variable names
    numeric_var_sanitized = sanitize_filename(numeric_var)
    categorical_var_sanitized = sanitize_filename(categorical_var)

    # Strip Plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(x=categorical_var, y=numeric_var, data=data, jitter=True, alpha=0.5)
    plt.title(f'Strip Plot of {numeric_var} by {categorical_var}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename_stripplot = os.path.join(PLOTS_DIR, f'stripplot_{numeric_var_sanitized}_by_{categorical_var_sanitized}.png')
    plt.savefig(filename_stripplot)
    plt.close()
    outputs['stripplot'] = os.path.relpath(filename_stripplot, REPORT_DIR)

    # Descriptive statistics
    desc_stats = data.groupby(categorical_var)[numeric_var].describe().round(2)
    outputs['desc_stats'] = desc_stats

    # Convert to HTML
    outputs['desc_stats_html'] = outputs['desc_stats'].to_html(classes='table table-striped', index=True)

    # Statistical Test
    grouped_data = [group[numeric_var].values for name, group in data.groupby(categorical_var)]
    if num_categories == 2:
        # Independent T-test
        stat, p_value = stats.ttest_ind(*grouped_data, equal_var=False)
        test_name = 'Independent T-test'
        outputs['test'] = {
            'name': test_name,
            'statistic': stat,
            'p_value': p_value
        }
        # Effect Size (Cohen's d)
        diff = np.mean(grouped_data[0]) - np.mean(grouped_data[1])
        pooled_sd = np.sqrt(((len(grouped_data[0])-1)*np.var(grouped_data[0], ddof=1) + (len(grouped_data[1])-1)*np.var(grouped_data[1], ddof=1)) / (len(grouped_data[0]) + len(grouped_data[1]) - 2))
        cohen_d = diff / pooled_sd
        outputs['effect_size'] = {'Cohen_d': cohen_d}
    elif num_categories > 2:
        # One-way ANOVA
        stat, p_value = stats.f_oneway(*grouped_data)
        test_name = 'One-way ANOVA'
        outputs['test'] = {
            'name': test_name,
            'statistic': stat,
            'p_value': p_value
        }
        # Effect Size (Eta squared)
        ss_between = sum(len(g) * (np.mean(g) - data[numeric_var].mean())**2 for g in grouped_data)
        ss_total = sum((data[numeric_var] - data[numeric_var].mean())**2)
        eta_squared = ss_between / ss_total
        outputs['effect_size'] = {'Eta_squared': eta_squared}

    return outputs

def analyze_categorical_vs_categorical(data, var1, var2):
    """Analyzes the relationship between two categorical variables."""
    outputs = {}

    # Contingency table
    contingency_table = pd.crosstab(data[var1], data[var2])
    outputs['contingency_table'] = contingency_table

    # Convert to HTML (limit to avoid large tables)
    outputs['contingency_table_html'] = contingency_table.head(20).to_html(classes='table table-striped', index=True)
    if contingency_table.shape[0] > 20:
        outputs['contingency_table_html'] += '<p>Only showing top 20 rows.</p>'

    # Chi-Squared Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    outputs['chi_squared'] = {
        'chi2_statistic': chi2,
        'p_value': p,
        'degrees_of_freedom': dof,
    }

    # Calculate Cramér's V
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1))/(n - 1))
    rcorr = r - ((r - 1)**2)/(n - 1)
    kcorr = k - ((k - 1)**2)/(n - 1)
    if min(rcorr, kcorr) > 0:
        cramers_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    else:
        cramers_v = np.nan  # Cannot compute Cramér's V
    outputs['cramers_v'] = cramers_v

    # Interpretation of Cramér's V
    if not np.isnan(cramers_v):
        if cramers_v < 0.1:
            strength = "Weak"
        elif cramers_v < 0.3:
            strength = "Moderate"
        elif cramers_v < 0.5:
            strength = "Relatively Strong"
        else:
            strength = "Strong"
    else:
        strength = "Undefined"
    outputs['association_strength'] = strength

    # Visualization (Stacked Bar Chart for small number of categories)
    if contingency_table.shape[0] <= 10 and contingency_table.shape[1] <= 10:
        plt.figure(figsize=(10, 6))
        contingency_table.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title(f'Stacked Bar Chart of {var1} vs {var2}')
        plt.xlabel(var1)
        plt.ylabel('Count')
        plt.legend(title=var2)
        plt.tight_layout()
        filename = os.path.join(PLOTS_DIR, f'stacked_bar_{sanitize_filename(var1)}_vs_{sanitize_filename(var2)}.png')
        plt.savefig(filename)
        plt.close()
        outputs['plot'] = os.path.relpath(filename, REPORT_DIR)
    else:
        outputs['plot'] = None  # Skip plot if too many categories

    return outputs

def generate_report(df, target_variable):
    """Generates an HTML report of the sequential attribute analysis with target variable relationships."""
    env = Environment(loader=FileSystemLoader(TEMPLATE_PATH))
    # Add custom filters
    env.filters['scientific_notation'] = scientific_notation
    template = env.get_template('templates/report_template.html')

    attribute_outputs = []

    # Identify data types
    numerical_vars, categorical_vars = detect_variable_types(df)

    # Remove target variable from lists
    if target_variable in numerical_vars:
        numerical_vars.remove(target_variable)
    if target_variable in categorical_vars:
        categorical_vars.remove(target_variable)

    target_type = 'numerical' if target_variable in numerical_vars else 'categorical'

    # Perform efficient numerical analysis at the top
    logging.info("Performing numerical analysis")
    numerical_analysis = analyze_numerical_variables(df, target_variable)

    # Sequentially analyze each attribute
    for column in tqdm(df.columns, desc="Analyzing Attributes"):
        if column == target_variable:
            continue

        logging.info(f"Analyzing {column}")
        output = {'attribute': column}

        if column in numerical_vars:
            analysis = analyze_numerical_attribute(df, column)
            output['type'] = 'numerical'
        elif column in categorical_vars:
            analysis = analyze_categorical_attribute(df, column)
            output['type'] = 'categorical'
        else:
            logging.warning(f"Unknown data type for column {column}. Skipping.")
            continue

        # Analyze relationship with target variable
        relationship = analyze_relationship(df, column, target_variable)
        output['analysis'] = analysis
        output['relationship'] = relationship
        attribute_outputs.append(output)

    # Render the report
    html_out = template.render(
        numerical_analysis=numerical_analysis,
        attributes=attribute_outputs,
        target_variable=target_variable
    )

    # Save the report
    report_filename = os.path.join(REPORT_DIR, 'auto_eda_report.html')
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_out)

    logging.info(f"Report saved to {report_filename}")

if __name__ == "__main__":
    generate_report(df, TARGET_VARIABLE)
