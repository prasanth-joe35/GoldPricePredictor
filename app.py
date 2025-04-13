import gradio as gr
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load Model 1
ridge_model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load Forecast from Prophet
forecast_df = pd.read_csv("forecast_gold_22k.csv")
forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

# --- Model 1: What-if prediction ---
def predict_model1(usd_inr, gold_usd):
    input_df = pd.DataFrame([[usd_inr, gold_usd]], columns=["USD_INR", "Gold_USD_per_ounce"])
    scaled = scaler.transform(input_df)
    prediction = ridge_model.predict(scaled)[0]
    return f"💰 Predicted 22K Gold Rate: ₹{round(prediction, 2)}"

# --- Model 2: Forecast Chart from Prophet ---
def show_forecast(days=365):
    # Show last 30 days + next N forecast days
    full = forecast_df.copy()
    latest_date = full['ds'].min() + pd.Timedelta(days=-30)
    forecast_plot = full[full['ds'] >= latest_date].tail(days + 30)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_plot['ds'], y=forecast_plot['yhat'], mode='lines+markers', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast_plot['ds'], y=forecast_plot['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_plot['ds'], y=forecast_plot['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))
    fig.update_layout(title="📈 Gold Price Forecast (22K INR/Gram)", xaxis_title="Date", yaxis_title="Price (INR)")
    return fig

# --- Model 2: Get price by specific date ---
def gold_price_on_date(date_str):
    try:
        date = pd.to_datetime(date_str)
        row = forecast_df[forecast_df['ds'] == date]
        if not row.empty:
            price = row['yhat'].values[0]
            return f"📅 Gold Price on {date.date()}: ₹{round(price, 2)}"
        else:
            return "❌ Date not available in forecast range."
    except:
        return "❌ Invalid date format. Use YYYY-MM-DD."

# --- Final UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Gold Price Prediction Dashboard")

    with gr.Tab("🔢 Model 1: What-If Simulation"):
        usd_inr_input = gr.Number(label="USD to INR", value=86.10)
        gold_usd_input = gr.Number(label="Gold USD per ounce", value=3237.59)
        output1 = gr.Textbox(label="Predicted Gold Price")
        gr.Button("Predict").click(predict_model1, inputs=[usd_inr_input, gold_usd_input], outputs=output1)

    with gr.Tab("📈 Model 2: Forecast Chart"):
        days_slider = gr.Slider(minimum=30, maximum=365, step=10, label="Show forecast for N days", value=180)
        output2 = gr.Plot()
        gr.Button("Show Forecast").click(fn=show_forecast, inputs=[days_slider], outputs=output2)

    with gr.Tab("📅 Price on Specific Date"):
        date_input = gr.Textbox(label="Enter Date (YYYY-MM-DD)")
        date_output = gr.Textbox(label="Forecasted Price")
        gr.Button("Get Price").click(fn=gold_price_on_date, inputs=[date_input], outputs=[date_output])

demo.launch()
