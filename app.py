import logging
import gradio as gr
import pandas as pd
import torch
import numpy as np
from GoogleNews import GoogleNews
from transformers import pipeline
import yfinance as yf
import requests
from fuzzywuzzy import process
import statistics
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="fuzzywuzzy")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SENTIMENT_ANALYSIS_MODEL = (
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

logging.info("Initializing sentiment analysis model...")
sentiment_analyzer = pipeline(
    "sentiment-analysis", model=SENTIMENT_ANALYSIS_MODEL, device=DEVICE
)
logging.info("Model initialized successfully")

# Technical Analysis Parameters
TA_CONFIG = {
    'rsi_window': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_window': 20,
    'sma_windows': [20, 50, 200],
    'ema_windows': [12, 26],
    'volatility_window': 30
}

EXCHANGE_SUFFIXES = {
    "NSE": ".NS",
    "BSE": ".BO",
    "NYSE": "",
    "NASDAQ": "",
}

def calculate_technical_indicators(history):
    """Calculate various technical indicators from historical price data"""
    ta_results = {}
    
    # RSI
    delta = history['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(TA_CONFIG['rsi_window']).mean()
    avg_loss = loss.rolling(TA_CONFIG['rsi_window']).mean()
    rs = avg_gain / avg_loss
    ta_results['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
    
    # MACD
    ema_fast = history['Close'].ewm(span=TA_CONFIG['macd_fast'], adjust=False).mean()
    ema_slow = history['Close'].ewm(span=TA_CONFIG['macd_slow'], adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=TA_CONFIG['macd_signal'], adjust=False).mean()
    ta_results['macd'] = macd.iloc[-1]
    ta_results['macd_signal'] = signal.iloc[-1]
    
    # Bollinger Bands
    sma = history['Close'].rolling(TA_CONFIG['bollinger_window']).mean()
    std = history['Close'].rolling(TA_CONFIG['bollinger_window']).std()
    ta_results['bollinger_upper'] = (sma + 2 * std).iloc[-1]
    ta_results['bollinger_lower'] = (sma - 2 * std).iloc[-1]
    
    # Moving Averages
    for window in TA_CONFIG['sma_windows']:
        ta_results[f'sma_{window}'] = history['Close'].rolling(window).mean().iloc[-1]
    for window in TA_CONFIG['ema_windows']:
        ta_results[f'ema_{window}'] = history['Close'].ewm(span=window, adjust=False).mean().iloc[-1]
    
    # Volatility
    returns = history['Close'].pct_change().dropna()
    ta_results['volatility_30d'] = returns.rolling(TA_CONFIG['volatility_window']).std().iloc[-1] * np.sqrt(252)
    
    return ta_results

def generate_price_chart(history):
    """Generate interactive price chart with technical indicators"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Price and Moving Averages
    history['Close'].plot(ax=ax1, label='Price')
    for window in TA_CONFIG['sma_windows']:
        history['Close'].rolling(window).mean().plot(ax=ax1, label=f'SMA {window}')
    ax1.set_title('Price and Moving Averages')
    ax1.legend()
    
    # RSI
    delta = history['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(TA_CONFIG['rsi_window']).mean()
    avg_loss = loss.rolling(TA_CONFIG['rsi_window']).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi.plot(ax=ax2, label='RSI')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='green', linestyle='--')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def resolve_ticker_symbol(query: str, exchange: str = "NSE") -> str:
    """
    Convert company names/partial symbols to valid Yahoo Finance tickers.
    Example: "Kalyan Jewellers" → "KALYANKJIL.NS"
    """
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    headers = {"User-Agent": "Mozilla/5.0"}  # Avoid blocking
    params = {"q": query, "quotesCount": 5, "country": "India"}  # Adjust for regional markets
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    if data.get("quotes"):
        # Extract symbols and names
        tickers = [quote["symbol"] for quote in data["quotes"]]
        names = [quote["longname"] or quote["shortname"] for quote in data["quotes"]]
        
        # Fuzzy match the query with company names
        best_match = process.extractOne(query, names)
        if best_match:
            index = names.index(best_match[0])
            resolved_ticker = tickers[index]
            
            # Ensure the exchange suffix is only added if not already present
            if not resolved_ticker.endswith(EXCHANGE_SUFFIXES.get(exchange, "")):
                resolved_ticker += EXCHANGE_SUFFIXES.get(exchange, "")
            return resolved_ticker
        else:
            # Default to first result
            resolved_ticker = tickers[0]
            if not resolved_ticker.endswith(EXCHANGE_SUFFIXES.get(exchange, "")):
                resolved_ticker += EXCHANGE_SUFFIXES.get(exchange, "")
            return resolved_ticker
    else:
        raise ValueError(f"No ticker found for: {query}")

def fetch_articles(query):
    try:
        logging.info(f"Fetching articles for query: '{query}'")
        googlenews = GoogleNews(lang="en")
        googlenews.search(query)
        articles = googlenews.result()
        logging.info(f"Fetched {len(articles)} articles")
        return articles
    except Exception as e:
        logging.error(
            f"Error while searching articles for query: '{query}'. Error: {e}"
        )
        raise gr.Error(
            f"Unable to search articles for query: '{query}'. Try again later...",
            duration=5,
        )

def analyze_article_sentiment(article):
    logging.info(f"Analyzing sentiment for article: {article['title']}")
    sentiment = sentiment_analyzer(article["desc"])[0]
    article["sentiment"] = sentiment
    return article

def fetch_yfinance_data(ticker):
    """Enhanced Yahoo Finance data fetching with technical analysis"""
    try:
        logging.info(f"Fetching Yahoo Finance data for: {ticker}")
        stock = yf.Ticker(ticker)
        
        # Get historical data for technical analysis
        history = stock.history(period="1y", interval="1d")
        
        # Calculate technical indicators
        ta_data = calculate_technical_indicators(history) if not history.empty else {}
        
        # Current price data
        current_price = history['Close'].iloc[-1] if not history.empty else 0
        prev_close = history['Close'].iloc[-2] if len(history) > 1 else 0
        price_change = current_price - prev_close
        percent_change = (price_change / prev_close) * 100 if prev_close != 0 else 0

        # Generate price chart
        chart = generate_price_chart(history[-120:])  # Last 120 days
        
        return {
            'current_price': current_price,
            'price_change': price_change,
            'percent_change': percent_change,
            'chart': chart,
            'technical_indicators': ta_data,
            'fundamentals': stock.info
        }
        
    except Exception as e:
        logging.error(f"Error fetching Yahoo Finance data: {str(e)}")
        return {"error": str(e)}

def time_weighted_sentiment(articles):
    """Apply time-based weighting to sentiment scores"""
    now = datetime.now()
    weighted_scores = []
    
    for article in articles:
        try:
            article_date = datetime.strptime(article['date'], '%Y-%m-%d %H:%M:%S')
            days_old = (now - article_date).days
            weight = max(0, 1 - (days_old / 7))  # Linear decay over 7 days
        except:
            weight = 0.5  # Default weight if date parsing fails
            
        sentiment = article['sentiment']['label']
        score = 1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0
        weighted_scores.append(score * weight)
    
    return weighted_scores

def _format_number(num):
    """Helper to format large numbers with suffixes"""
    if isinstance(num, (int, float)):
        for unit in ['','K','M','B','T']:
            if abs(num) < 1000:
                return f"{num:,.2f}{unit}"
            num /= 1000
        return f"{num:,.2f}P"
    return num

def convert_to_dataframe(analyzed_articles):
    df = pd.DataFrame(analyzed_articles)
    df["Title"] = df.apply(
        lambda row: f'<a href="{row["link"]}" target="_blank">{row["title"]}</a>',
        axis=1,
    )
    df["Description"] = df["desc"]
    df["Date"] = df["date"]

    def sentiment_badge(sentiment):
        colors = {
            "negative": "red",
            "neutral": "gray",
            "positive": "green",
        }
        color = colors.get(sentiment, "grey")
        return f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 4px;">{sentiment}</span>'

    df["Sentiment"] = df["sentiment"].apply(lambda x: sentiment_badge(x["label"]))
    return df[["Sentiment", "Title", "Description", "Date"]]

def generate_stock_recommendation(articles, finance_data):
    """Enhanced recommendation system with technical analysis"""
    # Time-weighted sentiment analysis
    sentiment_scores = time_weighted_sentiment(articles)
    positive_score = sum(s for s in sentiment_scores if s > 0)
    negative_score = abs(sum(s for s in sentiment_scores if s < 0))
    total_score = positive_score - negative_score
    
    # Technical indicators
    ta = finance_data.get('technical_indicators', {})
    rec = {
        'recommendation': 'HOLD',
        'confidence': 'Medium',
        'reasons': [],
        'risk_factors': []
    }
    
    # Sentiment-based factors
    if total_score > 3:
        rec['recommendation'] = 'BUY'
        rec['reasons'].append("Strong positive sentiment trend")
    elif total_score < -3:
        rec['recommendation'] = 'SELL'
        rec['reasons'].append("Significant negative sentiment")
        
    # Technical analysis factors
    if ta.get('rsi', 50) > 70:
        rec['risk_factors'].append("RSI indicates overbought condition")
    elif ta.get('rsi', 50) < 30:
        rec['reasons'].append("RSI suggests oversold opportunity")
        
    if ta.get('macd', 0) > ta.get('macd_signal', 0):
        rec['reasons'].append("Bullish MACD crossover")
    else:
        rec['risk_factors'].append("Bearish MACD trend")
        
    # Volatility analysis
    if ta.get('volatility_30d', 0) > 0.4:
        rec['risk_factors'].append("High volatility detected")
        
    # Combine factors
    if len(rec['reasons']) > len(rec['risk_factors']):
        rec['confidence'] = 'High'
    elif len(rec['risk_factors']) > 2:
        rec['recommendation'] = 'SELL' if rec['recommendation'] == 'HOLD' else rec['recommendation']
        rec['confidence'] = 'Low'
        
    # Format output
    output = f"Recommendation: {rec['recommendation']} ({rec['confidence']} Confidence)\n\n"
    output += "Supporting Factors:\n" + "\n".join(f"- {r}" for r in rec['reasons']) + "\n\n"
    output += "Risk Factors:\n" + "\n".join(f"- {r}" for r in rec['risk_factors']) + "\n\n"
    output += f"Sentiment Score: {total_score:.2f}\n"
    output += f"30-Day Volatility: {ta.get('volatility_30d', 0):.2%}"
    
    return output

def analyze_asset_sentiment(asset_input):
    logging.info(f"Starting sentiment analysis for asset: {asset_input}")

    try:
        # Resolve ticker symbol
        ticker = resolve_ticker_symbol(asset_input)
        logging.info(f"Resolved '{asset_input}' to ticker: {ticker}")

        # Fetch and analyze articles
        articles = fetch_articles(asset_input)
        analyzed_articles = [analyze_article_sentiment(article) for article in articles]

        # Fetch financial data and technical indicators
        finance_data = fetch_yfinance_data(ticker)
        
        # Extract chart and ensure it's removed from financial data
        price_chart = finance_data.get('chart')
        if 'chart' in finance_data:
            del finance_data['chart']

        # Generate recommendation
        recommendation = generate_stock_recommendation(analyzed_articles, finance_data)

        return (
            convert_to_dataframe(analyzed_articles),  # Articles dataframe
            finance_data,                             # Financial data (without chart)
            recommendation,                           # Text recommendation
            price_chart                               # Matplotlib figure
        )

    except Exception as e:
        logging.error(f"Error in analysis: {str(e)}")
        return (
            pd.DataFrame(), 
            {"error": str(e)}, 
            "Analysis failed", 
            None
        )
        
# Update Gradio interface with new components
with gr.Blocks(theme=gr.themes.Default()) as iface:
    gr.Markdown("# Advanced Trading Analytics Suite")
    
    with gr.Row():
        input_asset = gr.Textbox(
            label="Asset Name/Ticker",
            placeholder="Enter stock name or symbol...",
            max_lines=1
        )
        analyze_btn = gr.Button("Analyze", variant="primary")
    
    with gr.Tabs():
        with gr.TabItem("Sentiment Analysis"):
            articles_output = gr.Dataframe(render=False)  # Temporary hidden
            gr.Markdown("## News Sentiment Analysis", render=False)
            articles_output.render()

        with gr.TabItem("Technical Analysis"):
            price_chart = gr.Plot(label="Price Analysis")
            ta_json = gr.JSON(label="Technical Indicators")

        with gr.TabItem("Recommendation"):
            recommendation_output = gr.Textbox(
                lines=8,
                label="Analysis Summary",
                interactive=False
            )
    
    # Move the click handler inside the Blocks context
    analyze_btn.click(
        analyze_asset_sentiment,
        inputs=[input_asset],
        outputs=[articles_output, ta_json, recommendation_output, price_chart]
    )

logging.info("Launching enhanced Gradio interface")
iface.queue().launch()