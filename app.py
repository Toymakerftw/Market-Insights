import gradio as gr
from yfinance import Ticker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from datetime import datetime, timedelta

def fetch_stock_data(symbol):
    """Fetch historical data for the given stock symbol."""
    try:
        ticker = Ticker(symbol)
        historical_data = ticker.history(period="1mo")
        return historical_data
    except Exception as e:
        return None

def calculate_indicators(data):
    """Calculate technical indicators like RSI, MACD, etc."""
    if data is None:
        return None
    
    # Calculate RSI
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    
    # Calculate MACD
    macd, macd_signal, macd_hist = talib.MACD(data['Close'])
    data['MACD'] = macd
    data['MACD_Signal'] = macd_signal
    
    return data

def generate_recommendation(data):
    """Generate buy/sell recommendations based on indicators."""
    if data is None:
        return "Insufficient data"
    
    rsi = data['RSI'].iloc[-1]
    macd = data['MACD'].iloc[-1]
    macd_signal = data['MACD_Signal'].iloc[-1]
    
    if rsi < 30 and macd > macd_signal:
        return " BUY RECOMMENDATION: Stock is oversold and trending upward."
    elif rsi > 70 and macd < macd_signal:
        return " SELL RECOMMENDATION: Stock is overbought and trending downward."
    else:
        return "HOLD: No strong signal at this time."

def plot_stock_data(data):
    """Plot stock price and indicators."""
    plt.figure(figsize=(10, 6))
    
    # Plot closing prices
    plt.plot(data['Close'], label='Close Price')
    
    # Plot RSI
    ax = plt.twinx()
    ax.plot(data['RSI'], color='red', label='RSI')
    ax.axhline(30, color='r', linestyle='--')
    ax.axhline(70, color='r', linestyle='--')
    
    # Add MACD
    plt.figure(figsize=(10, 6))
    plt.plot(data['MACD'], label='MACD')
    plt.plot(data['MACD_Signal'], label='MACD Signal')
    plt.fill_between(data.index, data['MACD_hist'], color='gray', alpha=0.2, label='MACD Histogram')
    plt.legend()
    
    plt.title(f"Stock Analysis for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    
    return plt

def news_updates(symbol):
    """Fetch recent news articles for the stock."""
    try:
        news = Ticker(symbol).news
        articles = []
        for article in news:
            articles.append({
                'Title': article['title'],
                'Publish Date': article['publishDate'],
                'URL': article['url']
            })
        return pd.DataFrame(articles)
    except Exception as e:
        return pd.DataFrame(columns=['Title', 'Publish Date', 'URL'])

def main(symbol):
    """Main function to process stock symbol and generate output."""
    # Fetch real-time price and historical data
    real_time = Ticker(symbol).info
    price = real_time['regularMarketPrice']
    change = real_time['regularMarketChangePercent']
    
    # Fetch and process historical data
    historical_data = fetch_stock_data(symbol)
    if historical_data is None:
        return "Error: Could not fetch data for the symbol."
    
    data = calculate_indicators(historical_data)
    
    # Generate analysis
    analysis = generate_recommendation(data)
    
    # Plot visualization
    fig = plot_stock_data(data)
    
    # Fetch news
    news_df = news_updates(symbol)
    
    return {
        'Price Analysis': pd.DataFrame([{
            'Current Price': f'${price:.2f}',
            '24h Change': f'{change:.2f}%',
            'Recommendation': analysis
        }]),
        'Historical Data': data[['Close', 'RSI', 'MACD', 'MACD_Signal']],
        'Visualizations': fig,
        'Latest News': news_df
    }

def interface(symbol):
    """Gradio interface implementation."""
    try:
        analysis = main(symbol.strip())
        
        with gr.Blocks():
            gr.Markdown(f"### Stock Analysis for {symbol}")
            gr.Dataframe(analysis['Price Analysis'])
            gr.Plot(analysis['Visualizations'])
            if not analysis['Latest News'].empty:
                gr.Markdown("#### Latest News Updates")
                gr.Dataframe(analysis['Latest News'], max_rows=5)
    except Exception as e:
        return f"Error: {e}"

# Gradio Interface Setup
with gr.Blocks() as iface:
    gr.Markdown("# Stock Analysis Tool")
    gr.Markdown("Enter a stock symbol (e.g., TSLA, AAPL, etc.)")
    
    with gr.Row():
        stock_input = gr.Textbox(
            label="Stock Symbol",
            lines=1,
            placeholder="Enter stock symbol..."
        )
        analyze_button = gr.Button("Analyze", variant="primary")
    
    with gr.Row():
        gr.Examples(
            examples=[
                "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN"
            ],
            inputs=stock_input
        )
    
    analysis_output = gr.Textbox(label="Analysis")
    
# Connect the button to the interface function
    analyze_button.click(
        fn=interface,
        inputs=[stock_input],
        outputs=[analysis_output]
    )

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch()