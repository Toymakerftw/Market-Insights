import logging
import gradio as gr
import pandas as pd
import torch
from GoogleNews import GoogleNews
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

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

def scrape_google_finance(ticker):
    """Enhanced Google Finance scraper with better selectors and error handling."""
    stock_data = {
        "price": "N/A",
        "change": "N/A",
        "percent_change": "N/A",
        "market_cap": "N/A",
        "pe_ratio": "N/A",
        "volume": "N/A",
        "year_range": "N/A"
    }

    try:
        url = f"https://www.google.com/finance/quote/{ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }

        # Use session with retry logic
        session = requests.Session()
        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")

        # Price extraction
        price_div = soup.find("div", class_="YMlKec")
        if price_div:
            stock_data["price"] = price_div.text.strip()

        # Price change and percentage
        change_container = soup.find("div", class_="gyFHrc")
        if change_container:
            change_elements = change_container.find_all("div", class_="JwB6zf")
            if len(change_elements) >= 2:
                stock_data["change"] = change_elements[0].text.strip()
                stock_data["percent_change"] = change_elements[1].text.strip()

        # Additional financial metrics
        metrics = {
            "Market cap": "market_cap",
            "P/E ratio": "pe_ratio",
            "Volume": "volume",
            "Year range": "year_range"
        }

        for row in soup.find_all("div", class_="mfs7Fc"):
            label = row.text.strip()
            value_div = row.find_next_sibling("div", class_="P6K39c")
            if value_div and label in metrics:
                stock_data[metrics[label]] = value_div.text.strip()

    except requests.RequestException as e:
        logging.error(f"Network error scraping {ticker}: {str(e)}")
    except Exception as e:
        logging.error(f"Error processing {ticker} data: {str(e)}")

    # Clean numerical values
    for key in ["price", "change", "percent_change", "market_cap", "volume"]:
        if stock_data[key] != "N/A":
            stock_data[key] = stock_data[key].replace(',', '').replace('$', '')

    return stock_data

def analyze_asset_sentiment(asset_name):
    logging.info(f"Starting sentiment analysis for asset: {asset_name}")

    logging.info("Fetching articles")
    articles = fetch_articles(asset_name)

    logging.info("Analyzing sentiment of each article")
    analyzed_articles = [analyze_article_sentiment(article) for article in articles]

    logging.info("Scraping Google Finance data")
    finance_data = scrape_google_finance(asset_name)

    logging.info("Sentiment analysis completed")

    return convert_to_dataframe(analyzed_articles), finance_data

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
            label="Asset Name",
            lines=1,
            placeholder="Enter the name of the trading asset...",
        )

    with gr.Row():
        analyze_button = gr.Button("Analyze Sentiment", size="sm")

    gr.Examples(
        examples=[
            "Bitcoin",
            "Tesla",
            "Apple",
            "Amazon",
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