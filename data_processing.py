import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def feature_selection(data):
    data = data.copy()
    data.fillna(data.mean(numeric_only=True), inplace=True)

    numeric_data = data.select_dtypes(include=[np.number])
    target_variable = numeric_data['Life expectancy']
    features_data = numeric_data.drop(columns=['Life expectancy'])

    if features_data.isnull().any().any():
        print("Warning: Missing values detected. Filling with mean values.")
        features_data.fillna(features_data.mean(), inplace=True)

    k_best_selector = SelectKBest(score_func=f_regression, k=10)
    selected_features = k_best_selector.fit_transform(features_data, target_variable)
    print("Selected Features: ", features_data.columns[k_best_selector.get_support()])


def calculate_correlations(data):
    correlations = data.corrwith(data['Life expectancy'], numeric_only=True)
    correlations = correlations.sort_values(ascending=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm')
    plt.title("Correlation with 'Life expectancy'")
    plt.xlabel("Correlation Coefficient")
    plt.show()


def simple_preprocessing(data):
    data = data.copy()
    print("Initial data shape:", data.shape)

    mean_values = data.mean(numeric_only=True)
    data.fillna(mean_values, inplace=True)

    X = data[['Schooling']]
    y = data[['Life expectancy']]
    print("Shape of X:", X.shape, "Type of X:", type(X))
    print("Shape of y:", y.shape, "Type of y:", type(y))

    processed_data = pd.concat([X, y], axis=1)
    processed_data.to_csv("data/expectancy_single_processed.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.25, shuffle=True)
    print("Shapes of X_train, X_test, y_train, y_test:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # Debug print

    return X_train, X_test, y_train, y_test, data


def multi_preprocessing(data):
    data = data.copy()

    mean_values = data.mean(numeric_only=True)
    data.fillna(mean_values, inplace=True)

    all_numerical_columns = ['Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling']
    numerical_columns = ['Adult Mortality', ' HIV/AIDS', ' thinness  1-19 years', ' thinness 5-9 years', 'Schooling', 'Income composition of resources', ' BMI ', 'Polio', 'GDP']
    numerical_df = data[numerical_columns]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numerical_df)
    scaled_df = pd.DataFrame(scaled_data, columns=numerical_columns)

    encoder = OneHotEncoder(sparse_output=False)
    encoded_status = encoder.fit_transform(data[['Status']])
    status_columns = encoder.get_feature_names_out(['Status'])
    encoded_df = pd.DataFrame(encoded_status, columns=status_columns)

    processed_data = pd.concat([scaled_df, data["Life expectancy"]], axis=1)
    processed_data.to_csv("data/expectancy_multi_processed.csv", index=False)

    X = processed_data.drop(["Life expectancy"], axis=1)
    y = data['Life expectancy']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.25, shuffle=True)

    return X_train, X_test, y_train, y_test, data, scaler
