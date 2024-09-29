# AutoEDA

AutoEDA is an automated exploratory data analysis (EDA) tool that sequentially analyzes each attribute of a dataset, generates descriptive statistics, visualizations, and assesses relationships with a target variable. The results are compiled into an HTML report.

## Project Structure


## Installation

1. Clone the repository:
    ```bash
    git clone https://your-repository-link
    cd autoeda
    ```

2. (Optional) Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your dataset in the `data/` directory and rename it to `your_data.csv` or update the `DATA_PATH` variable in the `autoeda_run.py` file to point to your dataset.

2. Run the EDA script:
    ```bash
    python autoeda_run.py
    ```

3. Once the script finishes, check the `reports/` folder for your generated report `your_output_report.html` and the `plots/` subfolder for the auto-generated plots.

## Features

- **Sequential Attribute Analysis**: The tool analyzes each attribute in the dataset individually and generates descriptive statistics, histograms, boxplots, and more.
  
- **Correlation Analysis**: The script provides correlation analysis for numerical variables and their relationships with the target variable.
  
- **Relationship Analysis**: It analyzes the relationship between each predictor variable and the target variable, providing visualizations such as scatter plots, box plots, and strip plots depending on the variable types.

- **Customizable Reports**: The HTML report format is based on a customizable template stored in the `templates/` folder.

## File Descriptions

- **autoeda_run.py**: The main script that runs the entire EDA process.
  
- **report_template.html**: The HTML template used to generate the final EDA report.

- **requirements.txt**: List of dependencies required to run the project.

## Known Issues & Improvements

- **Long Processing Times**: Large datasets or datasets with many variables can lead to long processing times. Optimizing this script for speed and efficiency could improve its performance.

- **Numerical vs Categorical Analysis**: If a numerical variable has too many unique values, the strip plot might be too crowded. This could be mitigated by summarizing data in ranges or binning the values.

- **Handling of Missing Values**: Currently, the script drops records with missing target variables. Implementing various imputation strategies could be useful.

- **P-Value Warnings**: Ensure that the dataset is appropriate for parametric/non-parametric tests. The script currently applies general statistical methods that may not be suitable for all datasets.
  
- **Correlation Matrix Display Issues**: Ensure the dataset has sufficient numerical columns for correlation analysis, or handle this in the report if there are fewer variables.
  
- **Customization**: Modify `report_template.html` to customize how the report looks and which elements are included in the output.

## Future Enhancements

- **Parallel Processing**: Implementing parallel processing could help reduce runtime when dealing with large datasets.
  
- **Advanced Statistical Tests**: Add more advanced statistical testing and diagnostics based on the type of dataset being analyzed.
  
- **Interactive Visualizations**: Integrate tools like Plotly or Bokeh to generate interactive graphs for deeper analysis.
