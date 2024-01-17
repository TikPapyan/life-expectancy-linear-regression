import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def draw_map(data, X_test, y_test, trained_model):
    y_pred = trained_model.predict(X_test)

    result_df = pd.DataFrame({
        'Country': data.loc[X_test.index, 'Country'],
        'Actual Life Expectancy': y_test.values,
        'Predicted Life Expectancy': y_pred
    })

    result_df.to_csv('data/actual_vs_predicted.csv', index=False)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    merged = world.set_index('name').join(result_df.set_index('Country'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    merged.plot(column='Actual Life Expectancy', ax=ax1, legend=True,
                legend_kwds={'label': "Actual Life Expectancy by Country",
                             'orientation': "horizontal"}, cmap='coolwarm', vmin=0, vmax=100)
    ax1.set_title('Actual Life Expectancy')

    merged.plot(column='Predicted Life Expectancy', ax=ax2, legend=True,
                legend_kwds={'label': "Predicted Life Expectancy by Country",
                             'orientation': "horizontal"}, cmap='coolwarm', vmin=0, vmax=100)
    ax2.set_title('Predicted Life Expectancy')

    plt.tight_layout()
    plt.show()
