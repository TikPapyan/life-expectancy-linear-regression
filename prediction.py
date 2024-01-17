from data_processing import load_data, multi_preprocessing
from multi_lr import MultipleLinearRegression

file_path = "data/expectancy.csv"
data = load_data(file_path)

X_train, X_test, y_train, y_test, processed_data, scaler = multi_preprocessing(data)

multi_model = MultipleLinearRegression()
trained_model = multi_model.evaluate_scratch(X_train, X_test, y_train, y_test)


def calculate_predicted_expectancy(adult_mortality, hiv_aids, thinness_1_19, thinness_5_9, schooling, income, bmi, polio, gdp):
    input_features = [adult_mortality, hiv_aids, thinness_1_19, thinness_5_9, schooling, income, bmi, polio, gdp]
    scaled_features = scaler.transform([input_features])

    predicted_expectancy = trained_model.predict(scaled_features)
    predicted_expectancy_original_scale = predicted_expectancy[0]

    return f"{round(predicted_expectancy_original_scale)}"

if __name__ == "__main__":
    calculate_predicted_expectancy()
