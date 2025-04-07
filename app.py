import gradio as gr
import pandas as pd
import joblib
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import datetime

# Load Model 1 (Ridge)
ridge_model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load Model 2 (Prophet Forecast)
forecast_df = pd.read_csv("forecast_gold_22k.csv")
forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

# --- Function for Model 1 ---
def predict_model1(usd_inr, gold_usd):
    print(f"[LOG] User Input - USD/INR: {usd_inr}, Gold USD/oz: {gold_usd}")
    input_df = pd.DataFrame([[usd_inr, gold_usd]], columns=["USD_INR", "Gold_USD_per_ounce"])
    scaled = scaler.transform(input_df)
    prediction = ridge_model.predict(scaled)[0]
    
   
    log_entry = pd.DataFrame([[datetime.datetime.now(), usd_inr, gold_usd, prediction]],
                             columns=["Timestamp", "USD_INR", "Gold_USD", "Predicted_22K_INR"])
    log_entry.to_csv("user_inputs_log.csv", mode='a', header=False, index=False)

    return f"💰 Predicted 22K Gold Rate (1 gram): ₹{round(prediction, 2)}"

# --- Function for Model 2 ---
def show_forecast(days=180):
    forecast_plot = forecast_df.copy().tail(days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_plot['ds'], y=forecast_plot['yhat'],
                             mode='lines', name='Predicted'))
    fig.add_trace(go.Scatter(x=forecast_plot['ds'], y=forecast_plot['yhat_upper'],
                             mode='lines', name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_plot['ds'], y=forecast_plot['yhat_lower'],
                             mode='lines', name='Lower Bound', line=dict(dash='dot')))
    fig.update_layout(title="📈 Gold Price Forecast (22K INR/Gram)",
                      xaxis_title="Date", yaxis_title="Price (INR)")
    return fig

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Gold Price Prediction Dashboard (India 🇮🇳)")
    
    with gr.Tab("🔢 Model 1: What-If Simulation"):
        gr.Markdown("Enter real-time values to simulate the 22K Gold Price (1 gram)")
        usd_inr_input = gr.Number(label="USD to INR (e.g., 85.53)", value=85.5)
        gold_usd_input = gr.Number(label="Gold USD per ounce (e.g., 2321.70)", value=2320.0)
        output1 = gr.Textbox(label="Predicted Gold Price")
        predict_btn = gr.Button("Predict")
        predict_btn.click(fn=predict_model1, inputs=[usd_inr_input, gold_usd_input], outputs=output1)
    
    with gr.Tab("📆 Model 2: Forecast (Next 6 Months)"):
        gr.Markdown("This chart shows future predictions using Facebook Prophet model.")
        days_slider = gr.Slider(minimum=30, maximum=180, step=10, label="Show next N days", value=180)
        output2 = gr.Plot()
        forecast_btn = gr.Button("Show Forecast")
        forecast_btn.click(fn=show_forecast, inputs=[days_slider], outputs=output2)
    
    gr.Markdown("---")
    gr.Markdown("⚠️ **Disclaimer:** This tool provides educational forecasts based on historical data and should not be used for financial decisions.")

demo.launch()
