# Life Expectancy Prediction Project

## Overview
This Life Expectancy Prediction project aims to explore and analyze the impact of various health-related and socioeconomic factors on life expectancy. Utilizing a comprehensive dataset, the project employs machine learning techniques, particularly linear regression models, to predict life expectancy and understand the significance of each factor. The project includes data preprocessing, feature selection, model training, evaluation, and interactive visualizations.

## Key Features
 - **Data Processing**: Robust preprocessing of the dataset, including handling missing values, normalization, and feature engineering.
 - **Simple Linear Regression**: Implementation of simple linear regression to understand the impact of a single factor on life expectancy.
 - **Multiple Linear Regression**: Advanced analysis using multiple linear regression to predict life expectancy based on multiple factors.
 - **Feature Selection**: Utilization of statistical techniques to identify and select significant features influencing life expectancy.
 - **Visualization**: Comprehensive visualizations including scatter plots and geographical maps to illustrate the actual vs. predicted life expectancy and the distribution of life expectancy across different regions.
 - **Interactive Web Interface**: A user-friendly interface built with Gradio, allowing users to interact with the model and visualize predictions in real-time.

## Installation and Setup
Clone the repository to your local machine:

```
git clone https://github.com/your-username/life-expectancy-prediction.git
cd life-expectancy-prediction
```

Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage
Execute the **main.py** script with the desired flags to activate different functionalities:

```
python3 main.py --simple_lr  # Runs simple linear regression
python3 main.py --multi_lr   # Runs multiple linear regression
python3 main.py --feature_selection # Performs feature selection
python3 main.py --correlation # Calculates feature correlations
python3 main.py --visualize  # Visualizes results (use with --multi_lr or --simple_lr)
python3 main.py --interface  # Launches the Gradio interface
```

## Detailed Description of Scripts
 - **data_processing.py**: Contains functions for loading data from CSV files, preprocessing the data (including scaling and encoding), and splitting the data into training and test sets.
 - **simple_lr.py**: Custom implementation of simple linear regression, demonstrating the basic principles of linear regression.
 - **multi_lr.py**: Advanced implementation of multiple linear regression to incorporate multiple predictors in the model.
 - **draw_map.py** and **draw_scatter.py**: These scripts are responsible for generating visualizations. draw_map.py produces a world map indicating life expectancy, while draw_scatter.py creates scatter plots to compare actual and predicted life expectancy values.
 - **interface.py**: Constructs an interactive web interface using Gradio, enabling users to input data and receive predictions directly from the web application.
 - **main.py**: The central script that orchestrates the execution of the project's functionalities based on user input.
