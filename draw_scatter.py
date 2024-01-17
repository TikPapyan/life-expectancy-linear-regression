import pandas as pd

from bokeh.models import HoverTool
from bokeh.plotting import figure, show, output_file


def draw_scatter(data, X_test, y_test, trained_model):
    y_pred = trained_model.predict(X_test)

    result_df = pd.DataFrame({
        'Country': data.loc[X_test.index, 'Country'],
        'Actual Life Expectancy': y_test.values,
        'Predicted Life Expectancy': y_pred
    })

    result_df.to_csv('data/actual_vs_predicted.csv', index=False)

    output_file("data/life_expectancy.html")

    p = figure(title="Actual vs Predicted Life Expectancy", x_axis_label='Actual Life Expectancy', y_axis_label='Predicted Life Expectancy')
    p.circle('Actual Life Expectancy', 'Predicted Life Expectancy', source=result_df, legend_label="Countries", line_width=2)
    p.legend.location = "top_left"
    
    hover = HoverTool()
    hover.tooltips=[
        ('Country', '@Country'),
        ('Actual Life Expectancy', '@{Actual Life Expectancy}'),
        ('Predicted Life Expectancy', '@{Predicted Life Expectancy}')
    ]

    p.add_tools(hover)

    show(p)
