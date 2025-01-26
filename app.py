import logging
import gradio as gr
import pandas as pd
import torch
from GoogleNews import GoogleNews
from transformers import pipeline
import yfinance as yf
import requests
from fuzzywuzzy import process

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

# Exchange suffixes for Yahoo Finance
EXCHANGE_SUFFIXES = {
    "NSE": ".NS",
    "BSE": ".BO",
    "NYSE": "",
    "NASDAQ": "",
    # Add more exchanges as needed
}

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
    """Enhanced Yahoo Finance data fetching using official yfinance patterns"""
    try:
        logging.info(f"Fetching Yahoo Finance data for: {ticker}")
        stock = yf.Ticker(ticker)
        
        # Get market state and current session prices
        info = stock.info
        history = stock.history(period="2d", interval="1d")  # Get 2 days for change calculation
        
        # Base price information
        current_price = info.get('currentPrice', 
                              info.get('regularMarketPrice',
                              info.get('ask', 'N/A')))
        
        # Calculate price change using proper market session data
        if not history.empty and len(history) > 1:
            prev_close = history.iloc[-2]['Close']
            current_close = history.iloc[-1]['Close']
            change = current_close - prev_close
            percent_change = (change / prev_close) * 100
        else:
            change = info.get('regularMarketChange', 'N/A')
            percent_change = info.get('regularMarketChangePercent', 'N/A')

        # Format market cap with proper suffixes
        market_cap = info.get('marketCap')
        if market_cap and market_cap != 'N/A':
            market_cap = f"${_format_number(market_cap)}"

        return {
            # Core pricing
            'price': f"{current_price:.2f}" if isinstance(current_price, float) else current_price,
            'currency': info.get('currency', 'USD'),
            'previous_close': f"{prev_close:.2f}" if 'prev_close' in locals() else 'N/A',
            
            # Price movements
            'change': f"{change:.2f}" if isinstance(change, float) else change,
            'percent_change': f"{percent_change:.2f}%" if isinstance(percent_change, float) else percent_change,
            'day_range': f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}",
            
            # Market data
            'market_cap': market_cap,
            'volume': _format_number(info.get('volume', 'N/A')),
            'pe_ratio': info.get('trailingPE', info.get('forwardPE', 'N/A')),
            '52_week_range': f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
            
            # Additional fundamentals
            'dividend_yield': f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A',
            'beta': info.get('beta', 'N/A'),
            
            # Market state
            'market_state': info.get('marketState', 'CLOSED').title(),
            'exchange': info.get('exchangeName', 'N/A')
        }
        
    except Exception as e:
        logging.error(f"Error fetching Yahoo Finance data: {str(e)}")
        return {"error": f"Failed to fetch data for {ticker}: {str(e)}"}

def _format_number(num):
    """Helper to format large numbers with suffixes"""
    if isinstance(num, (int, float)):
        for unit in ['','K','M','B','T']:
            if abs(num) < 1000:
                return f"{num:,.2f}{unit}"
            num /= 1000
        return f"{num:,.2f}P"
    return num

def analyze_asset_sentiment(asset_input):
    logging.info(f"Starting sentiment analysis for asset: {asset_input}")

    try:
        # Resolve ticker symbol from user input
        ticker = resolve_ticker_symbol(asset_input)
        logging.info(f"Resolved '{asset_input}' to ticker: {ticker}")

        # Fetch articles and analyze sentiment
        logging.info("Fetching articles")
        articles = fetch_articles(asset_input)

        logging.info("Analyzing sentiment of each article")
        analyzed_articles = [analyze_article_sentiment(article) for article in articles]

        # Fetch Yahoo Finance data
        logging.info("Fetching Yahoo Finance data")
        finance_data = fetch_yfinance_data(ticker)

        logging.info("Sentiment analysis completed")
        return convert_to_dataframe(analyzed_articles), finance_data
    except ValueError as e:
        logging.error(f"Error resolving ticker: {str(e)}")
        raise gr.Error(f"Invalid input: {str(e)}")

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

with gr.Blocks() as iface:
    gr.Markdown("# Trading Asset Sentiment Analysis")
    gr.Markdown(
        "Enter the name of a trading asset, and I'll fetch recent articles and analyze their sentiment!"
    )

    with gr.Row():
        input_asset = gr.Textbox(
            label="Asset Name or Ticker Symbol",
            lines=1,
            placeholder="Enter the company name or ticker symbol (e.g., Kalyan Jewellers, AAPL, TSLA)...",
        )

    with gr.Row():
        analyze_button = gr.Button("Analyze Sentiment", size="sm")

    gr.Examples(
        examples=[
            "AAPL",
            "TSLA",
            "Kalyan Jewellers",
            "BTC-USD"
        ],
        inputs=input_asset,
    )

    with gr.Row():
        with gr.Column():
            with gr.Blocks():
                gr.Markdown("## Articles and Sentiment Analysis")
                articles_output = gr.Dataframe(
                    headers=["Sentiment", "Title", "Description", "Date"],
                    datatype=["markdown", "html", "markdown", "markdown"],
                    wrap=False,
                )

    with gr.Row():
        with gr.Column():
            with gr.Blocks():
                gr.Markdown("## Financial Data")
                finance_output = gr.JSON()

    analyze_button.click(
        analyze_asset_sentiment,
        inputs=[input_asset],
        outputs=[articles_output, finance_output],
    )

logging.info("Launching Gradio interface")
iface.queue().launch()