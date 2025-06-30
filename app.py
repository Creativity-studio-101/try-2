import os
import logging
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///portfolio.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the app with the extension
db.init_app(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

with app.app_context():
    import models
    db.create_all()

def get_market_data():
    """Fetch live market data using yfinance"""
    try:
        # Major indices
        indices = {
            '^NSEI': 'Nifty 50',
            '^BSESN': 'Sensex',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ'
        }
        
        # Crypto symbols
        crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD']
        
        # Commodities
        commodities = {
            'GC=F': 'Gold',
            'CL=F': 'Crude Oil'
        }
        
        market_data = {
            'indices': {},
            'crypto': {},
            'commodities': {},
            'top_gainers': [],
            'top_losers': [],
            'most_active': []
        }
        
        # Fetch indices data
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.history(period="1d")
                if not info.empty:
                    current_price = info['Close'].iloc[-1]
                    prev_close = info['Open'].iloc[-1]
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    market_data['indices'][symbol] = {
                        'name': name,
                        'price': round(current_price, 2),
                        'change': round(change, 2),
                        'change_pct': round(change_pct, 2)
                    }
            except Exception as e:
                logging.error(f"Error fetching {symbol}: {e}")
        
        # Fetch crypto data
        for symbol in crypto_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.history(period="1d")
                if not info.empty:
                    current_price = info['Close'].iloc[-1]
                    prev_close = info['Open'].iloc[-1]
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    market_data['crypto'][symbol] = {
                        'name': symbol.replace('-USD', ''),
                        'price': round(current_price, 2),
                        'change': round(change, 2),
                        'change_pct': round(change_pct, 2)
                    }
            except Exception as e:
                logging.error(f"Error fetching {symbol}: {e}")
        
        # Fetch commodities data
        for symbol, name in commodities.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.history(period="1d")
                if not info.empty:
                    current_price = info['Close'].iloc[-1]
                    prev_close = info['Open'].iloc[-1]
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    market_data['commodities'][symbol] = {
                        'name': name,
                        'price': round(current_price, 2),
                        'change': round(change, 2),
                        'change_pct': round(change_pct, 2)
                    }
            except Exception as e:
                logging.error(f"Error fetching {symbol}: {e}")
        
        # Get top gainers/losers from NSE (sample symbols)
        nse_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 
                      'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS']
        
        stock_data = []
        for symbol in nse_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.history(period="1d")
                if not info.empty:
                    current_price = info['Close'].iloc[-1]
                    prev_close = info['Open'].iloc[-1]
                    change_pct = ((current_price - prev_close) / prev_close) * 100
                    
                    stock_data.append({
                        'symbol': symbol.replace('.NS', ''),
                        'price': round(current_price, 2),
                        'change_pct': round(change_pct, 2)
                    })
            except Exception as e:
                logging.error(f"Error fetching {symbol}: {e}")
        
        # Sort for top gainers and losers
        stock_data.sort(key=lambda x: x['change_pct'], reverse=True)
        market_data['top_gainers'] = stock_data[:5]
        market_data['top_losers'] = stock_data[-5:]
        market_data['most_active'] = stock_data[:5]  # Simplified for now
        
        return market_data
        
    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        return {}

def get_financial_news():
    """Fetch financial news from RSS feeds - Compatible with all hosting platforms"""
    try:
        import feedparser
        import requests
        
        # Universal RSS feeds that work on any hosting platform
        news_feeds = [
            {
                'url': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'source': 'Yahoo Finance'
            },
            {
                'url': 'https://feeds.marketwatch.com/marketwatch/topstories/',
                'source': 'MarketWatch'
            },
            {
                'url': 'https://www.investing.com/rss/news.rss',
                'source': 'Investing.com'
            }
        ]
        
        all_news = []
        
        for feed_info in news_feeds:
            try:
                # Use requests with proper headers for better compatibility
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(feed_info['url'], headers=headers, timeout=10)
                if response.status_code == 200:
                    feed = feedparser.parse(response.content)
                    
                    for entry in feed.entries[:4]:  # Get top 4 from each feed
                        title = entry.title if hasattr(entry, 'title') else 'Financial News Update'
                        link = entry.link if hasattr(entry, 'link') else '#'
                        summary = entry.get('summary', entry.get('description', 'No summary available'))
                        
                        # Clean and truncate summary
                        if len(summary) > 180:
                            summary = summary[:180] + '...'
                        
                        published = entry.get('published', entry.get('updated', 'Recent'))
                        
                        all_news.append({
                            'title': title,
                            'link': link,
                            'summary': summary,
                            'published': published,
                            'source': feed_info['source']
                        })
                        
            except Exception as e:
                logging.error(f"Error fetching from {feed_info['url']}: {e}")
                continue
        
        # Sort by most recent and return top 10
        return all_news[:10] if all_news else []
        
    except Exception as e:
        logging.error(f"Error in get_financial_news: {e}")
        return []

@app.route('/')
def home():
    """Home page with live market data"""
    market_data = get_market_data()
    return render_template('home.html', market_data=market_data)

@app.route('/news')
def news():
    """News page with financial news"""
    news_data = get_financial_news()
    return render_template('news.html', news=news_data)

@app.route('/calculator')
def calculator():
    """Portfolio calculator main page with introduction"""
    return render_template('calculator.html')

@app.route('/add_trades')
def add_trades():
    """Manual trade entry page"""
    trades = session.get('trades', [])
    return render_template('add_trades.html', trades=trades)

@app.route('/add_trade', methods=['POST'])
def add_trade():
    """Handle manual trade addition"""
    try:
        trade_data = {
            'asset_type': request.form.get('asset_type'),
            'symbol': request.form.get('symbol').upper(),
            'quantity': float(request.form.get('quantity')),
            'buy_price': float(request.form.get('buy_price')),
            'purchase_date': request.form.get('purchase_date'),
            'notes': request.form.get('notes', '')
        }
        
        # Initialize trades in session if not exists
        if 'trades' not in session:
            session['trades'] = []
        
        session['trades'].append(trade_data)
        session.modified = True
        
        flash('Trade added successfully!', 'success')
        return redirect(url_for('add_trades'))
        
    except ValueError as e:
        flash('Invalid input. Please check your data.', 'error')
        return redirect(url_for('add_trades'))
    except Exception as e:
        logging.error(f"Error adding trade: {e}")
        flash('Error adding trade. Please try again.', 'error')
        return redirect(url_for('add_trades'))

@app.route('/upload_csv')
def upload_csv():
    """CSV upload page"""
    return render_template('upload_csv.html')

@app.route('/download_template/<template_type>')
def download_template(template_type):
    """Download CSV templates"""
    try:
        file_path = f'data/sample_{template_type}.csv'
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logging.error(f"Error downloading template: {e}")
        flash('Error downloading template.', 'error')
        return redirect(url_for('upload_csv'))

@app.route('/upload_portfolio', methods=['POST'])
def upload_portfolio():
    """Handle CSV file upload"""
    try:
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(url_for('upload_csv'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('upload_csv'))
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and process CSV
            df = pd.read_csv(filepath)
            
            # Convert to trades format
            trades = []
            for _, row in df.iterrows():
                trade = {
                    'asset_type': row.get('asset_type', 'Stock'),
                    'symbol': str(row.get('symbol', '')).upper(),
                    'quantity': float(row.get('quantity', 0)),
                    'buy_price': float(row.get('buy_price', 0)),
                    'purchase_date': str(row.get('purchase_date', '')),
                    'notes': str(row.get('notes', ''))
                }
                trades.append(trade)
            
            session['trades'] = trades
            session.modified = True
            
            # Clean up uploaded file
            os.remove(filepath)
            
            flash(f'Successfully uploaded {len(trades)} trades!', 'success')
            return redirect(url_for('report'))
        else:
            flash('Please upload a CSV file.', 'error')
            return redirect(url_for('upload_csv'))
            
    except Exception as e:
        logging.error(f"Error uploading portfolio: {e}")
        flash('Error processing file. Please check the format.', 'error')
        return redirect(url_for('upload_csv'))

@app.route('/report')
def report():
    """Generate portfolio report"""
    trades = session.get('trades', [])
    
    if not trades:
        flash('No trades found. Please add trades or upload a portfolio.', 'warning')
        return redirect(url_for('calculator'))
    
    # Calculate portfolio metrics
    portfolio_data = calculate_portfolio_metrics(trades)
    
    return render_template('report.html', 
                         trades=trades, 
                         portfolio_data=portfolio_data)

@app.route('/download_portfolio_csv')
def download_portfolio_csv():
    """Download current portfolio as CSV"""
    trades = session.get('trades', [])
    
    if not trades:
        flash('No portfolio data found.', 'warning')
        return redirect(url_for('calculator'))
    
    import csv
    import io
    from flask import make_response
    
    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Asset Type', 'Symbol', 'Quantity', 'Buy Price', 'Purchase Date', 'Notes'])
    
    # Write data
    for trade in trades:
        writer.writerow([
            trade.get('asset_type', ''),
            trade.get('symbol', ''),
            trade.get('quantity', ''),
            trade.get('buy_price', ''),
            trade.get('purchase_date', ''),
            trade.get('notes', '')
        ])
    
    # Create response
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename=portfolio_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    return response

def calculate_advanced_metrics(portfolio_items, total_invested, current_value, historical_data, nifty_data):
    """Calculate advanced portfolio metrics like CAGR, Sharpe ratio, etc."""
    import numpy as np
    from datetime import datetime, timedelta
    
    metrics = {}
    
    try:
        # Calculate portfolio-level metrics
        total_return = (current_value - total_invested) / total_invested if total_invested > 0 else 0
        
        # Calculate CAGR (assuming average holding period of 1 year for simplicity)
        # In real implementation, we'd use actual purchase dates
        avg_holding_period = 1.0  # years
        metrics['cagr'] = round(((current_value / total_invested) ** (1/avg_holding_period) - 1) * 100 if total_invested > 0 and avg_holding_period > 0 else 0, 2)
        
        # Calculate portfolio volatility (weighted average of individual volatilities)
        portfolio_volatility = 0
        portfolio_beta = 0
        total_weight = 0
        
        returns_data = []
        weights = []
        
        for item in portfolio_items:
            weight = item['weight'] / 100
            volatility = item['volatility']
            beta = item['beta']
            
            portfolio_volatility += weight * volatility
            portfolio_beta += weight * beta
            
            returns_data.append(item['pnl_pct'])
            weights.append(weight)
            total_weight += weight
        
        metrics['portfolio_volatility'] = round(portfolio_volatility, 2)
        metrics['portfolio_beta'] = round(portfolio_beta, 2)
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 6% for India)
        risk_free_rate = 6.0
        if portfolio_volatility > 0:
            excess_return = (total_return * 100) - risk_free_rate
            metrics['sharpe_ratio'] = round(excess_return / portfolio_volatility, 2)
        else:
            metrics['sharpe_ratio'] = 0
        
        # Calculate Alpha (simplified CAPM)
        market_return = 12.0  # Assuming 12% market return
        expected_return = risk_free_rate + portfolio_beta * (market_return - risk_free_rate)
        metrics['alpha'] = round((total_return * 100) - expected_return, 2)
        
        # Calculate Maximum Drawdown (simplified)
        if returns_data:
            cumulative_returns = np.cumprod([1 + r/100 for r in returns_data])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = round(np.min(drawdown) * 100, 2)
        else:
            metrics['max_drawdown'] = 0
        
        # Calculate Value at Risk (95% confidence, simplified)
        if returns_data and len(returns_data) > 1:
            returns_array = np.array(returns_data)
            metrics['var_95'] = round(np.percentile(returns_array, 5), 2)
        else:
            metrics['var_95'] = 0
        
        # Diversification metrics
        if len(portfolio_items) > 1:
            # Herfindahl Index (concentration measure)
            weights_squared = sum([w**2 for w in weights])
            metrics['diversification_ratio'] = round(1 / weights_squared if weights_squared > 0 else 0, 2)
        else:
            metrics['diversification_ratio'] = 1
        
        # Risk exposure analysis
        risk_levels = []
        for item in portfolio_items:
            volatility = item['volatility']
            weight = item['weight']
            
            if volatility > 30:
                risk_level = 'High'
            elif volatility > 15:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            risk_levels.append({
                'symbol': item['symbol'],
                'risk_level': risk_level,
                'weight': weight,
                'volatility': volatility
            })
        
        metrics['risk_exposure'] = risk_levels
        
        # Portfolio health indicators
        health_indicators = []
        
        # Check for overexposure (single asset > 20%)
        for item in portfolio_items:
            if item['weight'] > 20:
                health_indicators.append({
                    'type': 'overexposure',
                    'message': f"{item['symbol']} represents {item['weight']}% of portfolio (>20%)",
                    'severity': 'warning'
                })
        
        # Check for underperforming assets
        for item in portfolio_items:
            if item['pnl_pct'] < -15:
                health_indicators.append({
                    'type': 'underperforming',
                    'message': f"{item['symbol']} is down {abs(item['pnl_pct'])}%",
                    'severity': 'danger'
                })
        
        # Check portfolio diversity
        if len(set([item['asset_type'] for item in portfolio_items])) == 1:
            health_indicators.append({
                'type': 'concentration',
                'message': 'Portfolio concentrated in single asset class',
                'severity': 'warning'
            })
        
        metrics['health_indicators'] = health_indicators
        
        # Dividend analysis (simplified)
        dividend_yield = 0
        stock_count = 0
        for item in portfolio_items:
            if item['asset_type'].lower() == 'stock':
                # Simplified dividend yield estimation
                estimated_yield = 2.5  # Average dividend yield for Indian stocks
                dividend_yield += (item['weight'] / 100) * estimated_yield
                stock_count += 1
        
        metrics['avg_dividend_yield'] = round(dividend_yield, 2)
        metrics['estimated_annual_dividend'] = round(current_value * (dividend_yield / 100), 2)
        
    except Exception as e:
        logging.error(f"Error calculating advanced metrics: {e}")
        # Return basic metrics in case of error
        metrics = {
            'cagr': 0,
            'portfolio_volatility': 0,
            'portfolio_beta': 1,
            'sharpe_ratio': 0,
            'alpha': 0,
            'max_drawdown': 0,
            'var_95': 0,
            'diversification_ratio': 1,
            'risk_exposure': [],
            'health_indicators': [],
            'avg_dividend_yield': 0,
            'estimated_annual_dividend': 0
        }
    
    return metrics

def calculate_portfolio_metrics(trades):
    """Calculate comprehensive portfolio performance metrics"""
    try:
        total_invested = 0
        current_value = 0
        portfolio_items = []
        historical_data = {}
        
        # Get market index data for beta calculation
        nifty_data = None
        try:
            nifty = yf.Ticker("^NSEI")
            nifty_data = nifty.history(period="1y")
        except:
            logging.warning("Could not fetch Nifty data for beta calculation")
        
        for trade in trades:
            symbol = trade['symbol']
            quantity = trade['quantity']
            buy_price = trade['buy_price']
            purchase_date = trade['purchase_date']
            invested = quantity * buy_price
            total_invested += invested
            
            # Get current price and historical data
            try:
                if trade['asset_type'].lower() == 'crypto':
                    ticker_symbol = f"{symbol}-USD"
                elif trade['asset_type'].lower() == 'stock':
                    ticker_symbol = f"{symbol}.NS"  # NSE for Indian stocks
                else:
                    ticker_symbol = symbol
                
                ticker = yf.Ticker(ticker_symbol)
                hist_1d = ticker.history(period="1d")
                hist_1y = ticker.history(period="1y")
                hist_6m = ticker.history(period="6mo")
                hist_1m = ticker.history(period="1mo")
                
                # Store historical data for advanced calculations
                historical_data[symbol] = {
                    '1y': hist_1y,
                    '6m': hist_6m,
                    '1m': hist_1m,
                    'purchase_date': purchase_date
                }
                
                if not hist_1d.empty:
                    current_price = hist_1d['Close'].iloc[-1]
                    current_val = quantity * current_price
                    current_value += current_val
                    
                    pnl = current_val - invested
                    pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
                    
                    # Calculate historical returns
                    returns_1y = ((current_price - hist_1y['Close'].iloc[0]) / hist_1y['Close'].iloc[0] * 100) if len(hist_1y) > 0 else 0
                    returns_6m = ((current_price - hist_6m['Close'].iloc[0]) / hist_6m['Close'].iloc[0] * 100) if len(hist_6m) > 0 else 0
                    returns_1m = ((current_price - hist_1m['Close'].iloc[0]) / hist_1m['Close'].iloc[0] * 100) if len(hist_1m) > 0 else 0
                    
                    # Calculate volatility (standard deviation of daily returns)
                    if len(hist_1y) > 1:
                        daily_returns = hist_1y['Close'].pct_change().dropna()
                        volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility
                    else:
                        volatility = 0
                    
                    # Calculate beta (correlation with market)
                    beta = 0
                    if nifty_data is not None and len(hist_1y) > 10:
                        try:
                            # Align dates and calculate correlation
                            stock_returns = hist_1y['Close'].pct_change().dropna()
                            market_returns = nifty_data['Close'].pct_change().dropna()
                            
                            # Find common dates
                            common_dates = stock_returns.index.intersection(market_returns.index)
                            if len(common_dates) > 10:
                                stock_aligned = stock_returns.loc[common_dates]
                                market_aligned = market_returns.loc[common_dates]
                                
                                covariance = stock_aligned.cov(market_aligned)
                                market_variance = market_aligned.var()
                                beta = covariance / market_variance if market_variance != 0 else 0
                        except:
                            beta = 0
                    
                    portfolio_items.append({
                        'symbol': symbol,
                        'asset_type': trade['asset_type'],
                        'quantity': quantity,
                        'buy_price': buy_price,
                        'current_price': round(current_price, 2),
                        'invested': round(invested, 2),
                        'current_value': round(current_val, 2),
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                        'weight': round((current_val / current_value * 100) if current_value > 0 else 0, 2),
                        'returns_1y': round(returns_1y, 2),
                        'returns_6m': round(returns_6m, 2),
                        'returns_1m': round(returns_1m, 2),
                        'volatility': round(volatility, 2),
                        'beta': round(beta, 2),
                        'purchase_date': purchase_date
                    })
                else:
                    # If no current price available, use buy price
                    portfolio_items.append({
                        'symbol': symbol,
                        'asset_type': trade['asset_type'],
                        'quantity': quantity,
                        'buy_price': buy_price,
                        'current_price': buy_price,
                        'invested': round(invested, 2),
                        'current_value': round(invested, 2),
                        'pnl': 0,
                        'pnl_pct': 0,
                        'weight': 0,
                        'returns_1y': 0,
                        'returns_6m': 0,
                        'returns_1m': 0,
                        'volatility': 0,
                        'beta': 0,
                        'purchase_date': purchase_date
                    })
                    current_value += invested
                    
            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {e}")
                # Fallback: use buy price as current price
                portfolio_items.append({
                    'symbol': symbol,
                    'asset_type': trade['asset_type'],
                    'quantity': quantity,
                    'buy_price': buy_price,
                    'current_price': buy_price,
                    'invested': round(invested, 2),
                    'current_value': round(invested, 2),
                    'pnl': 0,
                    'pnl_pct': 0,
                    'weight': 0,
                    'returns_1y': 0,
                    'returns_6m': 0,
                    'returns_1m': 0,
                    'volatility': 0,
                    'beta': 0,
                    'purchase_date': purchase_date
                })
                current_value += invested
        
        # Update weights after calculating total current value
        for item in portfolio_items:
            item['weight'] = round((item['current_value'] / current_value * 100) if current_value > 0 else 0, 2)
        
        total_pnl = current_value - total_invested
        total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
        
        # Calculate advanced portfolio metrics
        portfolio_metrics = calculate_advanced_metrics(portfolio_items, total_invested, current_value, historical_data, nifty_data)
        
        # Calculate allocation by asset type
        allocation = {}
        for item in portfolio_items:
            asset_type = item['asset_type']
            if asset_type not in allocation:
                allocation[asset_type] = 0
            allocation[asset_type] += item['current_value']
        
        # Convert to percentages
        allocation_pct = {}
        for asset_type, value in allocation.items():
            allocation_pct[asset_type] = (value / current_value) * 100 if current_value > 0 else 0
        
        return {
            'total_invested': round(total_invested, 2),
            'current_value': round(current_value, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'portfolio_items': portfolio_items,
            'allocation': allocation_pct,
            'advanced_metrics': portfolio_metrics
        }
        
    except Exception as e:
        logging.error(f"Error calculating portfolio metrics: {e}")
        return {
            'total_invested': 0,
            'current_value': 0,
            'total_pnl': 0,
            'total_pnl_pct': 0,
            'portfolio_items': [],
            'allocation': {}
        }

@app.route('/clear_portfolio')
def clear_portfolio():
    """Clear current portfolio session"""
    session.pop('trades', None)
    flash('Portfolio cleared successfully!', 'info')
    return redirect(url_for('calculator'))

@app.route('/api/market_data')
def api_market_data():
    """API endpoint for live market data"""
    return jsonify(get_market_data())

@app.route('/api/search_symbols')
def search_symbols():
    """API endpoint to search for stock symbols"""
    query = request.args.get('q', '').upper().strip()
    if len(query) < 2:
        return jsonify([])
    
    # Popular Indian stocks for suggestions
    popular_stocks = [
        {'symbol': 'RELIANCE', 'name': 'Reliance Industries', 'type': 'Stock'},
        {'symbol': 'TCS', 'name': 'Tata Consultancy Services', 'type': 'Stock'},
        {'symbol': 'HDFCBANK', 'name': 'HDFC Bank', 'type': 'Stock'},
        {'symbol': 'INFY', 'name': 'Infosys', 'type': 'Stock'},
        {'symbol': 'ICICIBANK', 'name': 'ICICI Bank', 'type': 'Stock'},
        {'symbol': 'HINDUNILVR', 'name': 'Hindustan Unilever', 'type': 'Stock'},
        {'symbol': 'ITC', 'name': 'ITC Limited', 'type': 'Stock'},
        {'symbol': 'SBIN', 'name': 'State Bank of India', 'type': 'Stock'},
        {'symbol': 'BHARTIARTL', 'name': 'Bharti Airtel', 'type': 'Stock'},
        {'symbol': 'KOTAKBANK', 'name': 'Kotak Mahindra Bank', 'type': 'Stock'},
        {'symbol': 'LT', 'name': 'Larsen & Toubro', 'type': 'Stock'},
        {'symbol': 'ASIANPAINT', 'name': 'Asian Paints', 'type': 'Stock'},
        {'symbol': 'MARUTI', 'name': 'Maruti Suzuki', 'type': 'Stock'},
        {'symbol': 'HCLTECH', 'name': 'HCL Technologies', 'type': 'Stock'},
        {'symbol': 'WIPRO', 'name': 'Wipro', 'type': 'Stock'},
        {'symbol': 'AXISBANK', 'name': 'Axis Bank', 'type': 'Stock'},
        {'symbol': 'NTPC', 'name': 'NTPC Limited', 'type': 'Stock'},
        {'symbol': 'POWERGRID', 'name': 'Power Grid Corporation', 'type': 'Stock'},
        {'symbol': 'NESTLEIND', 'name': 'Nestle India', 'type': 'Stock'},
        {'symbol': 'TATAMOTORS', 'name': 'Tata Motors', 'type': 'Stock'},
    ]
    
    # Popular cryptocurrencies
    popular_crypto = [
        {'symbol': 'BTC', 'name': 'Bitcoin', 'type': 'Crypto'},
        {'symbol': 'ETH', 'name': 'Ethereum', 'type': 'Crypto'},
        {'symbol': 'BNB', 'name': 'Binance Coin', 'type': 'Crypto'},
        {'symbol': 'XRP', 'name': 'Ripple', 'type': 'Crypto'},
        {'symbol': 'ADA', 'name': 'Cardano', 'type': 'Crypto'},
        {'symbol': 'SOL', 'name': 'Solana', 'type': 'Crypto'},
        {'symbol': 'DOT', 'name': 'Polkadot', 'type': 'Crypto'},
        {'symbol': 'MATIC', 'name': 'Polygon', 'type': 'Crypto'},
        {'symbol': 'LINK', 'name': 'Chainlink', 'type': 'Crypto'},
        {'symbol': 'UNI', 'name': 'Uniswap', 'type': 'Crypto'},
        {'symbol': 'AVAX', 'name': 'Avalanche', 'type': 'Crypto'},
        {'symbol': 'ATOM', 'name': 'Cosmos', 'type': 'Crypto'},
        {'symbol': 'LTC', 'name': 'Litecoin', 'type': 'Crypto'},
        {'symbol': 'BCH', 'name': 'Bitcoin Cash', 'type': 'Crypto'},
        {'symbol': 'SHIB', 'name': 'Shiba Inu', 'type': 'Crypto'},
    ]
    
    # Popular mutual funds
    popular_mf = [
        {'symbol': 'AXIS_BLUECHIP', 'name': 'Axis Bluechip Fund', 'type': 'Mutual Fund'},
        {'symbol': 'SBI_SMALL_CAP', 'name': 'SBI Small Cap Fund', 'type': 'Mutual Fund'},
        {'symbol': 'HDFC_TOP_100', 'name': 'HDFC Top 100 Fund', 'type': 'Mutual Fund'},
        {'symbol': 'ICICI_FOCUSSED_BLUECHIP', 'name': 'ICICI Prudential Focused Bluechip', 'type': 'Mutual Fund'},
        {'symbol': 'MIRAE_LARGE_CAP', 'name': 'Mirae Asset Large Cap Fund', 'type': 'Mutual Fund'},
        {'symbol': 'KOTAK_SMALL_CAP', 'name': 'Kotak Small Cap Fund', 'type': 'Mutual Fund'},
        {'symbol': 'NIPPON_GROWTH', 'name': 'Nippon India Growth Fund', 'type': 'Mutual Fund'},
        {'symbol': 'DSP_MIDCAP', 'name': 'DSP Midcap Fund', 'type': 'Mutual Fund'},
        {'symbol': 'FRANKLIN_PRIMA', 'name': 'Franklin India Prima Fund', 'type': 'Mutual Fund'},
        {'symbol': 'UTI_NIFTY_INDEX', 'name': 'UTI Nifty Index Fund', 'type': 'Mutual Fund'},
    ]
    
    all_symbols = popular_stocks + popular_crypto + popular_mf
    
    # Filter based on query
    suggestions = []
    for item in all_symbols:
        if (query in item['symbol'] or 
            query in item['name'].upper()):
            suggestions.append(item)
    
    return jsonify(suggestions[:10])

@app.route('/api/get_price/<symbol>')
def get_current_price(symbol):
    """API endpoint to get current price of a symbol"""
    try:
        symbol = symbol.upper()
        asset_type = request.args.get('type', 'Stock')
        
        # Determine the Yahoo Finance symbol format
        if asset_type.lower() == 'crypto':
            ticker_symbol = f"{symbol}-USD"
        elif asset_type.lower() == 'stock':
            ticker_symbol = f"{symbol}.NS"  # NSE format
        else:
            ticker_symbol = symbol
        
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1d")
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Open'].iloc[-1]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close > 0 else 0
            
            return jsonify({
                'success': True,
                'symbol': symbol,
                'price': round(current_price, 2),
                'change': round(change, 2),
                'change_pct': round(change_pct, 2),
                'currency': '₹' if asset_type.lower() == 'stock' else '$'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Price data not available'
            })
            
    except Exception as e:
        logging.error(f"Error fetching price for {symbol}: {e}")
        return jsonify({
            'success': False,
            'error': 'Unable to fetch price'
        })

@app.route('/sip_calculator')
def sip_calculator():
    """SIP/SWP Calculator page"""
    return render_template('sip_calculator.html')

@app.route('/portfolio_recommendations')
def portfolio_recommendations():
    """Portfolio Recommendations page"""
    return render_template('portfolio_recommendations.html')

@app.route('/stock_research')
def stock_research():
    """Stock Research page with professional trading charts and company data"""
    return render_template('stock_research_professional.html')

@app.route('/api/calculate_sip', methods=['POST'])
def calculate_sip():
    """Calculate SIP projections"""
    try:
        data = request.get_json()
        monthly_investment = float(data.get('monthly_investment', 0))
        annual_return = float(data.get('annual_return', 12)) / 100
        duration_years = int(data.get('duration_years', 10))
        
        # SIP Formula: FV = P × [((1 + r/n)^(n×t) - 1) / (r/n)] × (1 + r/n)
        r = annual_return
        n = 12  # Monthly compounding
        t = duration_years
        P = monthly_investment
        
        # Calculate future value
        fv = P * (((1 + r/n)**(n*t) - 1) / (r/n)) * (1 + r/n)
        
        total_invested = P * n * t
        wealth_gained = fv - total_invested
        
        # Generate year-wise breakdown
        yearly_data = []
        for year in range(1, duration_years + 1):
            year_fv = P * (((1 + r/n)**(n*year) - 1) / (r/n)) * (1 + r/n)
            year_invested = P * n * year
            yearly_data.append({
                'year': year,
                'invested': round(year_invested, 0),
                'value': round(year_fv, 0),
                'gain': round(year_fv - year_invested, 0)
            })
        
        return jsonify({
            'success': True,
            'total_invested': round(total_invested, 0),
            'maturity_value': round(fv, 0),
            'wealth_gained': round(wealth_gained, 0),
            'yearly_data': yearly_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/calculate_swp', methods=['POST'])
def calculate_swp():
    """Calculate SWP projections"""
    try:
        data = request.get_json()
        corpus = float(data.get('corpus', 0))
        monthly_withdrawal = float(data.get('monthly_withdrawal', 0))
        annual_return = float(data.get('annual_return', 8)) / 100
        
        # Calculate how long money will last
        r = annual_return / 12  # Monthly return
        remaining_corpus = corpus
        months_data = []
        month = 0
        
        while remaining_corpus > monthly_withdrawal and month < 600:  # Max 50 years
            month += 1
            # Add monthly return
            remaining_corpus = remaining_corpus * (1 + r)
            # Subtract withdrawal
            remaining_corpus -= monthly_withdrawal
            
            if month % 12 == 0:  # Yearly data
                months_data.append({
                    'year': month // 12,
                    'remaining_corpus': round(remaining_corpus, 0),
                    'total_withdrawn': round(monthly_withdrawal * month, 0)
                })
        
        return jsonify({
            'success': True,
            'duration_months': month,
            'duration_years': round(month / 12, 1),
            'total_withdrawal': round(monthly_withdrawal * month, 0),
            'yearly_data': months_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/stock_data/<symbol>')
def get_stock_data(symbol):
    """API endpoint to get comprehensive stock data"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Basic price data
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        change = info.get('regularMarketChange', 0)
        change_percent = info.get('regularMarketChangePercent', 0)
        
        # Company information
        company_data = {
            'symbol': symbol,
            'name': info.get('longName', 'N/A'),
            'price': round(current_price, 2) if current_price else 0,
            'change': round(change, 2) if change else 0,
            'changePercent': round(change_percent, 2) if change_percent else 0,
            'marketCap': format_market_cap(info.get('marketCap')),
            'peRatio': round(info.get('forwardPE', 0), 2) if info.get('forwardPE') else 'N/A',
            'weekHigh52': round(info.get('fiftyTwoWeekHigh', 0), 2) if info.get('fiftyTwoWeekHigh') else 0,
            'weekLow52': round(info.get('fiftyTwoWeekLow', 0), 2) if info.get('fiftyTwoWeekLow') else 0,
            'volume': format_number(info.get('volume')),
            'avgVolume': format_number(info.get('averageVolume')),
            'dayLow': round(info.get('dayLow', 0), 2) if info.get('dayLow') else 0,
            'dayHigh': round(info.get('dayHigh', 0), 2) if info.get('dayHigh') else 0,
            'bookValue': round(info.get('bookValue', 0), 2) if info.get('bookValue') else 'N/A',
            'dividendYield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 'N/A',
            'eps': round(info.get('trailingEps', 0), 2) if info.get('trailingEps') else 'N/A',
            'beta': round(info.get('beta', 0), 2) if info.get('beta') else 'N/A',
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'employees': format_number(info.get('fullTimeEmployees')),
            'website': info.get('website', '#'),
            'revenue': format_market_cap(info.get('totalRevenue')),
            'profitMargin': round(info.get('profitMargins', 0) * 100, 2) if info.get('profitMargins') else 'N/A',
            'roe': round(info.get('returnOnEquity', 0) * 100, 2) if info.get('returnOnEquity') else 'N/A',
            'roa': round(info.get('returnOnAssets', 0) * 100, 2) if info.get('returnOnAssets') else 'N/A',
            'debtToEquity': round(info.get('debtToEquity', 0), 2) if info.get('debtToEquity') else 'N/A',
            'description': info.get('longBusinessSummary', 'Company information not available.')
        }
        
        return jsonify(company_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock_chart/<symbol>')
def get_stock_chart(symbol):
    """API endpoint to get OHLC stock chart data"""
    try:
        # Auto-add .NS suffix for Indian stocks if not present
        if not symbol.endswith('.NS') and not '.' in symbol:
            symbol = symbol + '.NS'
            
        period = request.args.get('period', '5d')
        interval = request.args.get('interval', '1d')
        
        # Enhanced interval mapping for professional trading charts
        interval_map = {
            '1d': '5m',     # 5-minute intervals for intraday
            '5d': '15m',    # 15-minute intervals for 5 days
            '1mo': '1h',    # 1-hour intervals for 1 month
            '3mo': '1d',    # Daily intervals for 3 months
            '1y': '1d',     # Daily intervals for 1 year
            '2y': '1wk',    # Weekly intervals for 2 years
            '5y': '1wk',    # Weekly intervals for 5 years
            '10y': '1mo'    # Monthly intervals for 10 years
        }
        
        interval = interval_map.get(period, '1d')
        
        ticker = yf.Ticker(symbol)
        
        # Get historical data with OHLC
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            return jsonify({'error': 'No chart data available'}), 404
        
        # Convert DataFrame to list format for ApexCharts
        chart_data = []
        
        # Drop any NaN rows and reset index
        hist_clean = hist.dropna().reset_index()
        
        for idx, row in hist_clean.iterrows():
            try:
                # Get date column (first column after reset_index)
                date_col = hist_clean.columns[0]
                date_val = row[date_col]
                
                # Convert to unix timestamp
                unix_timestamp = int(pd.Timestamp(date_val).timestamp())
                
                chart_data.append({
                    'time': unix_timestamp,
                    'open': round(float(row['Open']), 2),
                    'high': round(float(row['High']), 2),
                    'low': round(float(row['Low']), 2),
                    'close': round(float(row['Close']), 2),
                    'volume': int(float(row.get('Volume', 0))) if pd.notna(row.get('Volume', 0)) else 0
                })
            except (ValueError, TypeError, KeyError):
                continue
        
        if not chart_data:
            return jsonify({'error': f'No valid chart data available for {symbol}'}), 404
        
        return jsonify(chart_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def format_market_cap(value):
    """Format large numbers for market cap display"""
    if not value:
        return 'N/A'
    
    if value >= 1e12:
        return f'₹{value/1e12:.2f}T'
    elif value >= 1e9:
        return f'₹{value/1e9:.2f}B'
    elif value >= 1e7:
        return f'₹{value/1e7:.2f}Cr'
    elif value >= 1e5:
        return f'₹{value/1e5:.2f}L'
    else:
        return f'₹{value:,.0f}'

def format_number(value):
    """Format numbers with commas"""
    if not value:
        return 'N/A'
    
    if value >= 1e7:
        return f'{value/1e7:.2f}Cr'
    elif value >= 1e5:
        return f'{value/1e5:.2f}L'
    else:
        return f'{value:,.0f}'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
