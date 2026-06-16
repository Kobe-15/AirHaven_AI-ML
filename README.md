# AirHaven — Air Quality Monitoring System

## Overview
AirHaven AI is a machine learning system that processes real-time air quality sensor data to deliver hourly and daily AQI forecasts. Built with Python and scikit-learn, it performs automated data cleaning, feature engineering, and dual-model prediction — integrating seamlessly with Firebase to power the AirHaven mobile application.

## Tech Stack
- **Language**: Python 3.13.7
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, NumPy
- **Serialization**: joblib
- **Database**: Firebase Realtime Database
- **Data Format**: CSV, JSON

## Features
- **Dual-Prediction Models**: Separate hourly and daily air quality forecasting models
- **Real-time Data Processing**: Continuous data cleaning and normalization
- **Feature Engineering**: Advanced feature extraction for improved predictions
- **Model Persistence**: Trained models saved with scalers for production deployment
- **Data Management**: Automated data loading from Firebase and CSV sources
- **Model Metadata**: Comprehensive tracking of model configurations and performance metrics

## Project Structure
```
AI/
├── main.py                    # Entry point for the application
├── model.py                   # ML model definitions and logic
├── forecast.py                # Forecasting engine
├── features.py                # Feature engineering utilities
├── data_loader.py             # Firebase and data loading utilities
├── data_cleaner.py            # Data preprocessing and cleaning
├── config.py                  # Configuration management
├── models/                    # Trained model artifacts
│   ├── daily_model.joblib
│   ├── daily_scaler.joblib
│   ├── hourly_model.joblib
│   ├── hourly_scaler.joblib
│   ├── daily_meta.json
│   └── hourly_meta.json
├── Data/                      # Historical data
│   └── area_history.csv
└── README.md
```

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- Virtual environment (recommended)

### Setup Instructions
1. Clone the repository
   ```bash
   git clone <repository-url>
   cd airhaven-AI/AI
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your Firebase credentials and configuration
   ```

5. Run the application
   ```bash
   python main.py
   ```

## Key Scripts

### Core Application Scripts (Production)
- **main.py**: Entry point for the application
- **forecast.py**: Generate air quality predictions
- **features.py**: Feature engineering utilities
- **data_loader.py**: Fetch and manage data from Firebase
- **data_cleaner.py**: Data preprocessing and cleaning
- **config.py**: Configuration management
- **model.py**: ML model definitions and logic

### Paper Analysis Scripts (For Thesis Documentation)
The following scripts were created specifically for paper analysis and are not required for running the production application:
- **build_history.py**: Construct historical data archives for analysis
- **check_ranges.py**: Validate data ranges and anomalies for research
- **run_training.py**: Train or retrain the ML models for experimentation
- **visualize.py**: Generate analytics and visualization plots for the thesis paper

## Model Information
The system uses two separate machine learning models:
- **Daily Model**: Forecasts daily average air quality levels
- **Hourly Model**: Predicts hourly air quality measurements

Each model includes:
- Trained classifier/regressor
- Feature scaler for normalization
- Metadata with training configuration and performance metrics

## License

This project is part of an academic thesis. All rights reserved.
