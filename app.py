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
    url = f"https://www.google.com/finance/quote/{ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.134 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract relevant data
    stock_data = {}
    stock_data["price"] = soup.find("div", class_="YMlKec fxKbKc").text
    stock_data["change"] = soup.find("div", class_="JwB6zf").text
    stock_data["percent_change"] = soup.find("div", class_="Iap8Fd").text

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
