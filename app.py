import logging
import gradio as gr
import pandas as pd
import torch
import yfinance as yf
import requests
from GoogleNews import GoogleNews
from transformers import pipeline
from typing import Tuple, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class NewsAnalyzer:
    """Handles news article collection and sentiment analysis"""
    
    def __init__(self):
        self.sentiment_analyzer = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize sentiment analysis model"""
        model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing sentiment model on {device}")
        return pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device
        )
    
    def fetch_articles(self, query: str) -> List[Dict]:
        """Fetch relevant news articles"""
        try:
            logging.info(f"Searching news for: {query}")
            googlenews = GoogleNews(lang="en", region="US")
            googlenews.search(query)
            return googlenews.result()
        except Exception as e:
            logging.error(f"News search failed: {str(e)}")
            raise RuntimeError(f"Failed to fetch news: {str(e)}")

    def analyze_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """Process articles and add sentiment analysis"""
        analyzed = []
        for article in articles:
            try:
                sentiment = self.sentiment_analyzer(article["desc"])[0]
                analyzed.append({
                    **article,
                    "sentiment_label": sentiment["label"],
                    "sentiment_score": sentiment["score"]
                })
            except Exception as e:
                logging.warning(f"Failed to analyze article: {str(e)}")
        return self._format_results(analyzed)

    def _format_results(self, articles: List[Dict]) -> pd.DataFrame:
        """Convert articles to formatted DataFrame"""
        df = pd.DataFrame(articles)
        if not df.empty:
            df["Title"] = df.apply(
                lambda x: f'<a href="{x["link"]}" target="_blank">{x["title"]}</a>',
                axis=1
            )
            df["Sentiment"] = df["sentiment_label"].apply(self._sentiment_badge)
        return df[["Sentiment", "Title", "desc", "date"]] if not df.empty else pd.DataFrame()

    def _sentiment_badge(self, label: str) -> str:
        """Create styled sentiment badges"""
        colors = {"negative": "#ef4444", "neutral": "#64748b", "positive": "#22c55e"}
        return f'<span style="background-color: {colors[label]}; color: white; padding: 2px 8px; border-radius: 4px;">{label.title()}</span>'

class FinancialDataHandler:
    """Handles financial data retrieval and processing"""
    
    def __init__(self):
        self.valid_types = ["EQUITY", "ETF", "CRYPTOCURRENCY"]

    def resolve_symbol(self, query: str) -> Tuple[str, str]:
        """Resolve user input to valid ticker symbol"""
        try:
            # Try direct match first
            ticker = yf.Ticker(query)
            if self._is_valid(ticker):
                return query.upper(), ticker.info["shortName"]
        except:
            pass

        # Fallback to search API
        results = self._search_symbols(query)
        if not results:
            raise ValueError(f"No results found for '{query}'")

        valid_results = [r for r in results if r.get("quoteType") in self.valid_types]
        if not valid_results:
            raise ValueError(f"No valid instruments found for '{query}'")

        best_match = valid_results[0]
        return best_match["symbol"], best_match["shortname"]

    def _is_valid(self, ticker: yf.Ticker) -> bool:
        """Validate ticker has required data"""
        required_keys = ["symbol", "shortName", "regularMarketPrice"]
        return all(key in ticker.info for key in required_keys)

    def _search_symbols(self, query: str) -> List[Dict]:
        """Search Yahoo Finance API for symbols"""
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            return response.json().get("quotes", [])
        except Exception as e:
            logging.error(f"Symbol search failed: {str(e)}")
            return []

    def get_financials(self, symbol: str) -> Dict:
        """Retrieve comprehensive financial data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period="2d")
            
            return {
                "symbol": symbol,
                "name": info.get("shortName", "N/A"),
                "price": self._format_price(info),
                "change": self._calculate_change(history),
                "market_cap": self._format_market_cap(info),
                "volume": self._format_number(info.get("volume")),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "52_week_range": self._format_range(info),
                "currency": info.get("currency", "USD"),
                "error": None
            }
        except Exception as e:
            logging.error(f"Financial data error: {str(e)}")
            return {"error": str(e)}

    def _format_price(self, info: Dict) -> str:
        """Format price with currency"""
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        return f"{price:,.2f}" if price else "N/A"

    def _calculate_change(self, history: pd.DataFrame) -> Dict:
        """Calculate price changes from historical data"""
        if len(history) < 2:
            return {"amount": "N/A", "percent": "N/A"}
        
        prev_close = history.iloc[-2]["Close"]
        current_close = history.iloc[-1]["Close"]
        change = current_close - prev_close
        percent = (change / prev_close) * 100
        
        return {
            "amount": f"{change:+,.2f}",
            "percent": f"{percent:+,.2f}%"
        }

    def _format_market_cap(self, info: Dict) -> str:
        """Format market cap with proper suffix"""
        cap = info.get("marketCap")
        suffixes = ["", "K", "M", "B", "T"]
        for suffix in suffixes:
            if cap < 1000:
                return f"${cap:,.2f}{suffix}"
            cap /= 1000
        return f"${cap:,.2f}P"

    def _format_number(self, num: float) -> str:
        """Format large numbers with commas"""
        return f"{num:,.0f}" if num else "N/A"

    def _format_range(self, info: Dict) -> str:
        """Format 52-week range"""
        low = info.get("fiftyTwoWeekLow")
        high = info.get("fiftyTwoWeekHigh")
        return f"{low:,.2f} - {high:,.2f}" if low and high else "N/A"

class TradingAnalysisApp:
    """Main application orchestrator"""
    
    def __init__(self):
        self.news_analyzer = NewsAnalyzer()
        self.finance_handler = FinancialDataHandler()

    def analyze(self, user_input: str) -> Tuple[pd.DataFrame, Dict]:
        """Main analysis workflow"""
        try:
            # Resolve symbol and get financial data
            symbol, name = self.finance_handler.resolve_symbol(user_input)
            financials = self.finance_handler.get_financials(symbol)
            
            # Get and analyze news articles
            articles = self.news_analyzer.fetch_articles(name)
            analyzed_news = self.news_analyzer.analyze_articles(articles)
            
            return analyzed_news, financials
            
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            return pd.DataFrame(), {"error": str(e)}

    def create_interface(self):
        """Build Gradio interface"""
        with gr.Blocks(title="Trading Asset Analyzer", theme="soft") as app:
            gr.Markdown("# 🧠 Smart Trading Asset Analyzer")
            gr.Markdown("Analyze financial instruments using news sentiment and market data")
            
            with gr.Row():
                with gr.Column(scale=2):
                    input_box = gr.Textbox(
                        label="Asset Name or Symbol",
                        placeholder="Enter company name or ticker symbol..."
                    )
                    examples = gr.Examples(
                        examples=["Apple", "TSLA", "BTC-USD", "Amazon", "S&P 500"],
                        inputs=[input_box]
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary")
                
                with gr.Column(scale=3):
                    with gr.Tab("News Sentiment"):
                        news_output = gr.Dataframe(
                            label="Recent News Analysis",
                            headers=["Sentiment", "Title", "Description", "Date"],
                            datatype=["html", "html", "str", "str"]
                        )
                    
                    with gr.Tab("Financial Data"):
                        finance_output = gr.JSON(
                            label="Market Analysis",
                            show_label=False
                        )
                    
                    error_output = gr.JSON(
                        visible=False,
                        label="Analysis Errors"
                    )

            analyze_btn.click(
                self.analyze,
                inputs=[input_box],
                outputs=[news_output, finance_output],
                show_progress="full"
            )

        return app

if __name__ == "__main__":
    logging.info("Initializing application")
    analyzer = TradingAnalysisApp()
    interface = analyzer.create_interface()
    interface.queue(concurrency_count=2).launch()