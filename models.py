from app import db
from datetime import datetime

class Portfolio(db.Model):
    """Model for storing portfolio data"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with trades
    trades = db.relationship('Trade', backref='portfolio', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Portfolio {self.name}>'

class Trade(db.Model):
    """Model for storing individual trades"""
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'), nullable=False)
    
    asset_type = db.Column(db.String(50), nullable=False)  # Stock, Crypto, Mutual Fund
    symbol = db.Column(db.String(20), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    buy_price = db.Column(db.Float, nullable=False)
    purchase_date = db.Column(db.Date, nullable=False)
    notes = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Trade {self.symbol} - {self.quantity}>'

class MarketData(db.Model):
    """Model for caching market data"""
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, unique=True)
    name = db.Column(db.String(100))
    current_price = db.Column(db.Float)
    change = db.Column(db.Float)
    change_pct = db.Column(db.Float)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<MarketData {self.symbol}>'
