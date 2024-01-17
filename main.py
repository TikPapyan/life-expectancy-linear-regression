import argparse

from data_processing import load_data, feature_selection, calculate_correlations, simple_preprocessing, multi_preprocessing
from simple_lr import SimpleLinearRegression
from multi_lr import MultipleLinearRegression
from draw_map import draw_map
from draw_scatter import draw_scatter
from interface import iface


def main(args):
    data_initial = load_data('data/expectancy.csv')

    if any([args.simple_lr, args.multi_lr, args.feature_selection, args.correlation, args.visualize]):
        data = data_initial.copy()

    X_train, X_test, y_train, y_test, trained_model = None, None, None, None, None

    if args.simple_lr:
        X_train, X_test, y_train, y_test, _ = simple_preprocessing(data)
        simple_model = SimpleLinearRegression()
        simple_model.evaluate_scratch(X_train, X_test, y_train, y_test)

    if args.multi_lr:
        X_train, X_test, y_train, y_test, _, scaler = multi_preprocessing(data)
        multi_model = MultipleLinearRegression()
        trained_model = multi_model.evaluate_scratch(X_train, X_test, y_train, y_test)

    if args.feature_selection:
        feature_selection(data)

    if args.correlation:
        calculate_correlations(data)

    if args.visualize:
        if trained_model is not None and X_test is not None and y_test is not None:
            draw_map(data_initial, X_test, y_test, trained_model)
            draw_scatter(data_initial, X_test, y_test, trained_model)
        else:
            print("Visualization requires a trained model and test data. Please ensure you run with --multi_lr or --simple_lr.")

    if args.interface:
        iface.launch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Life Expectancy Prediction Script")
    parser.add_argument('--simple_lr', help="Run Simple Linear Regression", action='store_true')
    parser.add_argument('--multi_lr', help="Run Multiple Linear Regression", action='store_true')
    parser.add_argument('--feature_selection', help="Perform Feature Selection", action='store_true')
    parser.add_argument('--correlation', help="Calculate Correlations", action='store_true')
    parser.add_argument('--visualize', help="Visualize Results. Use with --multi_lr or --simple_lr to generate visualizations.", action='store_true')
    parser.add_argument('--interface', help="Launch Gradio Interface", action='store_true')

    args = parser.parse_args()
    main(args)
