import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


class SimpleLinearRegression:
    def __init__(self):
        self.slope_ = None
        self.intercept_ = None
        self.residual_ = None
        self.RSS = None
        self.TSS = None
        self.r2score_ = None


    def fit(self, x, y):
        if isinstance(x, pd.DataFrame):
            x = x.squeeze()
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        self.slope_ = numerator / denominator
        self.intercept_ = y_mean - self.slope_ * x_mean

        assert np.isscalar(self.slope_), "Slope is not a scalar value."
        assert np.isscalar(self.intercept_), "Intercept is not a scalar value."

        y_pred = self.intercept_ + self.slope_ * x
        self.residual_ = y - y_pred
        
        self.RSS = np.sum(self.residual_ ** 2)
        self.TSS = np.sum((y - y_mean) ** 2)
        self.r2score_ = 1 - (self.RSS / self.TSS)


    def predict(self, x):
        return self.intercept_ + self.slope_ * x


    def evaluate_scratch(self, X_train, X_test, y_train, y_test):
        model = SimpleLinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.squeeze()
        if isinstance(pred, pd.Series):
            pred = pred.values

        mse = np.mean((y_test - pred) ** 2)

        print("Simple Linear Regression from scratch")
        print(f"Simple linear equation: y = {model.intercept_:.2f} + {model.slope_:.2f}x")
        print(f"Slope: {model.slope_:.2f}")
        print(f"Intercept: {model.intercept_:.2f}")
        print(f"Residual sum of squares (RSS): {model.RSS:.2f}")
        print(f"Total sum of squares (TSS): {model.TSS:.2f}")
        print(f"Coefficient of determination (R^2): {model.r2score_:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")

        plt.scatter(X_test, y_test, label='Observed values')
        plt.plot(X_test, pred, color='red', marker='o', label='Regression line')
        plt.xlabel('Adult Mortality')
        plt.ylabel('Life Expectancy')
        plt.title('Simple Linear Regression')
        plt.legend()
        plt.show()
    
    
    def evaluate_sklearn(self, X_train, X_test, y_train, y_test):
        model = LinearRegression()
        model.fit(X_train.values.reshape(-1, 1), y_train)
        y_pred = model.predict(X_test.values.reshape(-1, 1))
        mse = ((y_test - y_pred) ** 2).mean()

        residuals = y_test - y_pred
        rss = sum(residuals ** 2)

        mean_y = y_test.mean()
        tss = sum((y_test - mean_y) ** 2)

        print("\nSimple Linear Regression Sklearn \n")
        print(f"Simple linear equation: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x")
        print(f"Slope: {model.coef_}")
        print(f"Intercept: {model.intercept_:.2f}")
        print(f"Residual Sum of Squares (RSS): {rss:.2f}")
        print(f"Total Sum of Squares (TSS): {tss:.2f}")
        print(f"Coefficient of determination (R^2): {model.score(X_test.values.reshape(-1, 1), y_test):.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")

        plt.scatter(X_test, y_test, color='blue', label='Observed')
        plt.scatter(X_test, y_pred, color='red', marker='o', label='Regression line')
        plt.xlabel('Adult Mortality')
        plt.ylabel('Life Expectancy')
        plt.title('Simple Linear Regression')
        plt.legend()
        plt.show()
