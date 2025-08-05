# Algorithmic Trading System

A comprehensive, AI-driven algorithmic trading system specification and implementation framework.

## 🚀 Overview

This repository contains a detailed specification for building a sophisticated algorithmic trading system that combines traditional technical analysis with modern AI/ML capabilities, analyst ratings, and comprehensive risk management.

## 📋 Key Features

### Core Trading Features
- **Multi-Strategy Trading Engine**: 10+ built-in strategies with custom development framework
- **Multi-Interval Analysis**: Support for 1min to 1month intervals with cross-interval confirmation
- **Real-Time Market Data**: Live streaming via Alpaca API with 50+ technical indicators
- **Advanced Order Management**: Market, limit, stop, and stop-limit orders

### AI & Machine Learning
- **AI-Driven Algorithm Discovery**: Automatic algorithm discovery based on market conditions
- **Machine Learning Engine**: Price prediction models (Random Forest, LSTM)
- **Natural Language Interface**: LLM-powered queries and strategy explanations
- **Autonomous Trading System**: Fully autonomous trading cycles with continuous learning

### Risk Management & Loss Control
- **Comprehensive Loss Management**: Percentage-based limits (transaction, daily, lifetime)
- **Risk-Adjusted Position Sizing**: Kelly Criterion with volatility adjustments
- **Dynamic Risk Controls**: Dynamic stop loss and take profit calculation
- **Risk Profiles**: Conservative, moderate, and aggressive configurations

### Analyst Rating Integration
- **Multi-Source Ratings**: Yahoo Finance, Alpha Vantage, Finnhub, Polygon
- **Configurable Minimum Ratings**: Set minimum analyst rating for buy decisions
- **Rating Trend Analysis**: Consider rating trends in trading decisions
- **Position Size Adjustment**: Adjust positions based on analyst sentiment

### Backtesting & Performance
- **Comprehensive Backtesting**: Historical strategy validation with multiple metrics
- **Performance Analytics**: Real-time portfolio tracking and risk-adjusted returns
- **Strategy Optimization**: Parameter optimization with genetic algorithms
- **Walk-Forward Analysis**: Robust strategy validation

## 🏗️ Architecture

### Technology Stack
- **Backend**: Python 3.9+ with FastAPI
- **Trading API**: Alpaca Trading API (v2)
- **Database**: PostgreSQL (primary), Redis (caching), Chroma (vector DB)
- **Web UI**: React/Next.js with real-time dashboards
- **ML/AI**: scikit-learn, TensorFlow/PyTorch, Ollama/Llama
- **Monitoring**: Prometheus + Grafana, ELK Stack
- **Deployment**: Docker with Docker Compose

### System Components
1. Data Manager - Market data processing
2. Strategy Engine - Trading algorithms
3. Signal Generator - Multi-algorithm combination
4. Order Manager - Trade execution
5. Risk Manager - Position sizing and risk control
6. Analyst Rating Manager - External rating integration
7. ML Engine - Predictive models
8. LLM Interface - Natural language processing
9. Web UI - User interface and monitoring
10. Backtesting Engine - Strategy validation

## 📁 Repository Structure

```
algo/
├── spec.md              # Comprehensive system specification
├── ai_prompt.txt        # AI development guidelines
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- PostgreSQL
- Redis
- Node.js 18+ (for web UI)

### Configuration
The system uses YAML configuration files for:
- Trading strategies and parameters
- Risk management settings
- Analyst rating requirements
- Database connections
- API keys and external services

### Key Configuration Options

```yaml
# Risk Management
risk_management:
  max_transaction_loss_pct: 2.0    # 2% per trade
  max_daily_loss_pct: 5.0          # 5% per day
  max_lifetime_loss_pct: 15.0      # 15% lifetime
  risk_profile: "moderate"         # conservative, moderate, aggressive

# Analyst Ratings
analyst_ratings:
  enabled: true
  min_rating_for_buy: 3.0          # Minimum rating for buy decisions
  rating_weight: 0.2               # 20% weight in decisions
  rating_sources:
    yahoo_finance: true
    alpha_vantage: true
    finnhub: true

# Trading
trading:
  paper_trading: true              # Start with paper trading
  symbols: ["AAPL", "MSFT", "GOOGL"]
  max_concurrent_positions: 10
```

## 🔧 Development

### Setting Up Development Environment
1. Clone the repository
2. Install Python dependencies
3. Set up PostgreSQL and Redis
4. Configure API keys for Alpaca and external data sources
5. Run the development environment with Docker Compose

### Testing
The specification includes comprehensive testing strategies:
- Unit testing for all components
- Integration testing for system components
- API testing for web endpoints
- Performance and security testing
- Docker-based testing environment

## 📊 Features in Detail

### Loss Management System
- **Transaction Level**: 2% maximum loss per trade
- **Daily Level**: 5% maximum loss per day
- **Lifetime Level**: 15% maximum lifetime loss
- **Automatic Stops**: Trading stops when limits are exceeded

### Analyst Rating Integration
- **Minimum Rating Requirement**: Configurable threshold for buy decisions
- **Multi-Source Aggregation**: Combines ratings from multiple providers
- **Trend Analysis**: Considers rating changes over time
- **Position Adjustment**: Adjusts position sizes based on ratings

### AI-Driven Features
- **Algorithm Discovery**: Automatically finds optimal strategies for each stock
- **Market Classification**: Analyzes market types and stock characteristics
- **Autonomous Trading**: Fully automated trading cycles
- **Continuous Learning**: Learns from trading results and optimizes performance

## 🔒 Security & Compliance

- **Data Security**: Encrypted storage and transmission
- **API Security**: Rate limiting and authentication
- **Trading Compliance**: Regulatory compliance monitoring
- **Audit Trail**: Comprehensive logging and tracking

## 📈 Performance Monitoring

- **Real-Time Dashboards**: Live portfolio and performance monitoring
- **Risk Metrics**: Sharpe ratio, max drawdown, win rate tracking
- **System Health**: Prometheus metrics and Grafana visualizations
- **Logging**: Structured logging with ELK stack

## 🤝 Contributing

This is a specification repository. To contribute:
1. Review the `spec.md` file for the complete system specification
2. Follow the guidelines in `ai_prompt.txt` for AI-assisted development
3. Implement components according to the specification
4. Add tests and documentation

## 🚀 GitHub Repository Setup Guide

### Setting Up a New Repository with Correct GitHub Account

This guide helps you set up a new repository using the `dsdjung` GitHub account instead of the default `pulzzedavid` account.

#### Prerequisites
- Git installed on your system
- GitHub account (`dsdjung`)
- GitHub Personal Access Token (for HTTPS authentication)

#### Step-by-Step Process

##### 1. Initialize Git Repository
```bash
# Navigate to your project directory
cd /path/to/your/project

# Initialize git repository
git init

# Set the default branch to main
git branch -m main
```

##### 2. Configure Git User (Local Repository)
```bash
# Set local git user configuration (overrides global)
git config --local user.name "dsdjung"
git config --local user.email "david@interactor.com"

# Verify the configuration
git config --local --list | grep user
```

##### 3. Add Files and Make Initial Commit
```bash
# Add all files to staging
git add .

# Make initial commit
git commit -m "Initial commit: [Your project description]"
```

##### 4. Create Remote Repository on GitHub
- Go to [GitHub.com](https://github.com) and log in with your `dsdjung` account
- Click the "+" icon → "New repository"
- Name your repository (e.g., `your-project-name`)
- Make it public or private as preferred
- **Don't** initialize with README (you already have files)
- Click "Create repository"

##### 5. Fix Authentication Issues (If Using SSH)
If you encounter authentication errors like:
```
ERROR: Permission to dsdjung/repo.git denied to pulzzedavid.
```

**Solution: Use HTTPS instead of SSH**
```bash
# Check current remote (if already added)
git remote -v

# Remove existing remote if it uses SSH
git remote remove origin

# Add remote using HTTPS
git remote add origin https://github.com/dsdjung/your-repo-name.git

# Verify remote URL
git remote -v
```

##### 6. Push to GitHub
```bash
# Push to GitHub (will prompt for username and token)
git push -u origin main
```

When prompted:
- **Username**: `dsdjung`
- **Password**: Use your GitHub Personal Access Token (not your GitHub password)

#### Alternative: GitHub CLI Setup
```bash
# Install GitHub CLI
brew install gh

# Login with dsdjung account
gh auth login

# Create repository and push in one command
gh repo create your-repo-name --public --source=. --remote=origin --push
```

#### Troubleshooting Common Issues

##### Issue: SSH Key Associated with Wrong Account
```bash
# Check which account your SSH key is associated with
ssh -T git@github.com

# If it shows "Hi pulzzedavid!" instead of "Hi dsdjung!"
# Use HTTPS authentication instead of SSH
```

##### Issue: Global Git Configuration Override
```bash
# Check global configuration
git config --global --list | grep user

# If global config shows pulzzedavid, use local config
git config --local user.name "dsdjung"
git config --local user.email "david@interactor.com"
```

##### Issue: URL Conversion from HTTPS to SSH
```bash
# Check if there's a URL conversion rule
git config --list | grep url

# Remove any URL conversion rules
git config --global --unset url."git@github.com:".insteadof
git config --global --unset url."https://github.com/".insteadOf
```

#### Verification Steps
```bash
# Verify local git configuration
git config --local --list | grep user

# Verify remote URL
git remote -v

# Test authentication (for HTTPS)
git ls-remote origin

# Check commit author
git log --oneline --decorate
```

#### Best Practices
1. **Always use local git config** for project-specific user settings
2. **Use HTTPS authentication** to avoid SSH key conflicts
3. **Create Personal Access Token** for secure authentication
4. **Verify configuration** before pushing
5. **Test authentication** with `git ls-remote origin`

#### Security Notes
- Store your GitHub Personal Access Token securely
- Use environment variables for sensitive data
- Regularly rotate your access tokens
- Never commit API keys or sensitive credentials

## 📄 License

This project is for educational and development purposes. Please ensure compliance with all applicable trading regulations and broker terms of service.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## 📞 Support

For questions about the specification or implementation:
- Review the comprehensive `spec.md` file
- Check the configuration examples
- Refer to the testing and deployment sections

---

**Note**: This repository contains a detailed specification for an algorithmic trading system. Implementation should be done carefully with proper testing and risk management. 