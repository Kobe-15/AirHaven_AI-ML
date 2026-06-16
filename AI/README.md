# AirHaven — Air Quality Monitoring System
A thesis project developed at Pamantasan ng Lungsod ng Maynila (2025–2026)

## Overview
AirHaven is an intelligent air quality monitoring and forecasting system developed as a thesis project for the BS Computer Engineering program. The system utilizes machine learning models to predict air quality levels and provide real-time monitoring across different areas. This repository contains the AI/ML backbone of the AirHaven application, featuring data processing, model training, and prediction capabilities.

## Tech Stack
- **Language**: Python 3.x
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

## Thesis Information
- **Institution**: Pamantasan ng Lungsod ng Maynila
- **Program**: BS Computer Engineering
- **Year**: 2025–2026
- **Project Title**: AirHaven — Real-time Air Quality Monitoring and Forecasting System

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

## Contributing
This is a thesis project for educational purposes at PLM. 
External contributions are not expected but feel free to fork and explore.