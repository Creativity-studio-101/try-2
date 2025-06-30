# Portfolio Analyzer - Technical Specification

## Overview

Portfolio Analyzer is a comprehensive Flask web application designed to help users track and analyze their investment portfolios. The application provides real-time market data, portfolio performance analysis, financial news aggregation, and advanced risk management tools. Users can manage their investments across multiple asset classes including stocks, cryptocurrencies, mutual funds, and ETFs. The platform now includes SIP/SWP calculators, personalized fund recommendations, hedging strategies, and enhanced dark mode functionality.

## User Preferences

```
Preferred communication style: Simple, everyday language.
```

## System Architecture

### Backend Architecture
- **Framework**: Flask 3.1.1 with Python 3.11
- **Database**: SQLAlchemy ORM with configurable database backend (SQLite for development, PostgreSQL for production)
- **Data Processing**: Pandas for portfolio analytics and CSV processing
- **Market Data**: Yahoo Finance API (yfinance) for real-time stock and crypto prices
- **News Integration**: RSS feed parsing using feedparser for financial news
- **Web Server**: Gunicorn for production deployment with autoscaling support

### Frontend Architecture
- **Template Engine**: Jinja2 with Flask
- **CSS Framework**: Bootstrap 5.3.0 for responsive design
- **Icons**: Font Awesome 6.0.0 for consistent iconography
- **Charts**: Chart.js for interactive data visualizations
- **JavaScript**: Vanilla JavaScript for form validation and interactive features

### Deployment Strategy
- **Platform**: Replit with Nix package management
- **Configuration**: Uses .replit file for deployment settings
- **Production Server**: Gunicorn with auto-scaling capabilities
- **Static Assets**: Served directly through Flask for development

## Key Components

### Database Models (`models.py`)
1. **Portfolio Model**: Stores portfolio metadata and relationships
2. **Trade Model**: Individual investment transactions with asset type, quantity, and pricing
3. **MarketData Model**: Caches market data to reduce API calls and improve performance

### Core Application (`app.py`)
- Flask application factory pattern with SQLAlchemy integration
- Session management with configurable secret keys
- File upload handling with security constraints (16MB limit)
- Proxy fix middleware for proper deployment behind reverse proxies

### Market Data Integration
- Real-time data fetching using yfinance library
- Support for stocks, cryptocurrencies, and market indices
- Caching mechanism to prevent excessive API calls
- Error handling for network failures and invalid symbols

### Portfolio Analysis Engine
- Calculates current portfolio value using live market prices
- Computes profit/loss metrics and percentage gains
- Generates asset allocation breakdowns
- Supports multiple currency formats (primarily INR)

## Data Flow

### Portfolio Management Workflow
1. **Data Input**: Users can add trades manually or upload CSV files
2. **Validation**: Input validation for asset types, quantities, and dates
3. **Storage**: Trades stored in normalized database structure
4. **Analysis**: Real-time calculation of portfolio metrics using current market prices
5. **Reporting**: Generation of comprehensive reports with charts and insights

### Market Data Pipeline
1. **API Integration**: Fetch live prices from Yahoo Finance
2. **Data Processing**: Parse and normalize market data
3. **Caching**: Store frequently accessed data to improve performance
4. **Display**: Present data in user-friendly dashboard format

### News Aggregation
1. **Source Integration**: RSS feeds from financial news sources
2. **Content Processing**: Parse and format news articles
3. **Presentation**: Display as card-based layout with timestamps

## External Dependencies

### Core Python Packages
- **Flask Ecosystem**: flask, flask-sqlalchemy, werkzeug for web framework
- **Database**: sqlalchemy, psycopg2-binary for database operations
- **Data Processing**: pandas for analytics, yfinance for market data
- **Web Utilities**: requests for HTTP operations, feedparser for RSS feeds
- **Validation**: email-validator for input validation

### Frontend Dependencies (CDN)
- **Bootstrap 5.3.0**: Responsive UI framework
- **Font Awesome 6.0.0**: Icon library
- **Chart.js**: Data visualization library

### System Dependencies (Nix)
- **PostgreSQL**: Production database system
- **OpenSSL**: Secure communications
- **glibc Locales**: Internationalization support

## Deployment Strategy

### Development Environment
- **Runtime**: Python 3.11 with Nix package management
- **Database**: SQLite for local development
- **Server**: Flask development server with auto-reload

### Production Environment
- **Server**: Gunicorn WSGI server with multiple workers
- **Database**: PostgreSQL with connection pooling
- **Scaling**: Auto-scaling deployment target on Replit
- **Security**: ProxyFix middleware for proper header handling

### Configuration Management
- **Environment Variables**: DATABASE_URL for database connection, SESSION_SECRET for security
- **File Uploads**: Configurable upload directory with size limits
- **Logging**: Structured logging with configurable levels

### Data Management
- **CSV Templates**: Pre-built templates for different asset types
- **Sample Data**: Example datasets for stocks, crypto, and mutual funds
- **File Security**: Secure filename handling for uploads

The application follows a traditional MVC pattern with clear separation of concerns, making it maintainable and scalable. The architecture supports both development and production environments with appropriate configuration switches.

## Recent Changes (June 30, 2025)

### Professional Trading Chart Implementation
- **Complete Chart System Overhaul**: Replaced TradingView Lightweight Charts with ApexCharts for professional-grade candlestick visualization
- **Symbol Auto-Correction**: Automatically adds .NS suffix for Indian stocks (HDFCBANK → HDFCBANK.NS)
- **Enhanced Chart Sizing**: Increased chart height to 650px and container minimum height to 700px for better visibility
- **Advanced Interval Mapping**: Professional timeframe intervals - 1D (5min), 5D (15min), 1M (1hr), 3M (daily), 1Y+ (weekly/monthly)
- **Interactive Features**: Full zoom, pan, hover tooltips with complete OHLC and volume data
- **Technical Indicators**: EMA (20), SMA (50), Bollinger Bands, and synchronized volume analysis
- **Multiple Chart Types**: Candlestick, line, and area charts with seamless switching
- **Robust Data Processing**: Fixed pandas DataFrame handling for reliable OHLC data extraction
- **Professional UI**: Trading platform-like interface with popular Indian stocks quick access

### Portfolio Analyzer Simplification
- **Removed Instruction Carousel**: Eliminated slides interface in favor of direct action buttons
- **Streamlined Navigation**: Clean header with immediate access to portfolio management features
- **Risk Management Integration**: Incorporated investment guidelines within main workflow

### Chart API Enhancements
- **Improved Data Validation**: Better handling of NaN values and missing data points
- **Enhanced Error Handling**: Clear error messages for invalid symbols or missing data
- **Optimized Performance**: Cleaned data processing pipeline for faster chart rendering
- **Symbol Search Integration**: Autocomplete with popular Indian stocks and real-time suggestions

### User Experience Improvements
- **Professional Styling**: Trading platform-quality interface design
- **Dark Mode Compatibility**: Full dark mode support for all chart components
- **Responsive Design**: Charts adapt properly to different screen sizes
- **Loading States**: Professional loading animations and error handling

### Technical Architecture Updates
- **API Endpoint Optimization**: Enhanced `/api/stock_chart/<symbol>` with better interval logic
- **Data Format Standardization**: Consistent OHLC data structure for ApexCharts compatibility
- **Error Recovery**: Graceful handling of delisted or invalid stock symbols
- **Performance Optimization**: Efficient data processing and chart rendering pipeline

### Previous Features (Maintained)
- News feed compatibility with universal RSS sources
- Dark mode persistence with localStorage
- CSV download functionality for portfolio exports
- Enhanced symbol search with autocomplete
- Comprehensive stock data API endpoints
- Risk management guidelines integration
- Professional navigation and user interface design