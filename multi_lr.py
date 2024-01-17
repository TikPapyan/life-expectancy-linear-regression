import numpy as np

from sklearn.linear_model import LinearRegression


class MultipleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.residual_ = None
        self.RSS = None
        self.TSS = None
        self.r2score_ = None
        

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X_with_ones = np.hstack((ones, X))        
        coefficients = np.linalg.inv(X_with_ones.T @ X_with_ones) @ X_with_ones.T @ y
        self.intercept_ = coefficients[0]
        self.coef_ = coefficients[1:]
        
        y_pred = X_with_ones @ coefficients
        self.residual_ = y - y_pred
        
        self.RSS = np.sum(self.residual_ ** 2)
        self.TSS = np.sum((y - np.mean(y)) ** 2)
        self.r2score_ = 1 - (self.RSS / self.TSS)


    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X_with_ones = np.hstack((ones, X))
        
        return X_with_ones @ np.hstack((self.intercept_, self.coef_))


    def evaluate_scratch(self, X_train, X_test, y_train, y_test):
        model = MultipleLinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = ((y_test - pred) ** 2).mean()

        coef_tuples = [(X_train.columns[i], coef) for i, coef in enumerate(model.coef_)]
        coef_tuples.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Multiple Linear Regression from scratch\n")
        print(f"Intercept: {model.intercept_:.2f}")
        print(f"Residual sum of squares (RSS): {model.RSS:.2f}")
        print(f"Total sum of squares (TSS): {model.TSS:.2f}")
        print(f"Coefficient of determination (R^2): {model.r2score_:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        
        return model


    def evaluate_sklearn(self, X_train, X_test, y_train, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = ((y_test - y_pred) ** 2).mean()

        residuals = y_test - y_pred
        rss = sum(residuals ** 2)

        mean_y = y_test.mean()
        tss = sum((y_test - mean_y) ** 2)

        print("\nMultiple Linear Regression Sklearn\n")
        print(f"Intercept: {model.intercept_:.2f}")
        print(f"Residual Sum of Squares (RSS): {rss:.2f}")
        print(f"Total Sum of Squares (TSS): {tss:.2f}")
        print(f"Coefficient of determination (R^2): {model.score(X_test, y_test):.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
