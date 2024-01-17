import gradio as gr

from prediction import calculate_predicted_expectancy


def predict_life_expectancy(adult_mortality, hiv_aids, thinness_1_19, thinness_5_9, schooling, income, bmi, polio, gdp):
    result = calculate_predicted_expectancy(adult_mortality, hiv_aids, thinness_1_19, thinness_5_9, schooling, income, bmi, polio, gdp)
    return result


iface = gr.Interface(
    fn=predict_life_expectancy,
    inputs=[
        gr.Number(label="Adult Mortality Rate"),
        gr.Number(label="HIV/AIDS"),
        gr.Number(label="Thinness 1-19 age"),
        gr.Number(label="Thinness 5-9 age"),
        gr.Number(label="Schooling Rate"),
        gr.Number(label="Income"),
        gr.Number(label="BMI"),
        gr.Number(label="Polio"),
        gr.Number(label="GDP")
    ],
    outputs=gr.Textbox(label="Predicted Expectancy"),
    title="Life Expectancy Prediction",
    description="Enter the details of the quality of life to predict its expectancy.",
    
)

if __name__ == "__main__":
    iface.launch()
