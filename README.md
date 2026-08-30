# 🏠 Lahore Real Estate AI Analyst

An intelligent, full-stack web application that predicts property prices in Lahore, Pakistan, and answers complex real estate questions using **Machine Learning** and **Large Language Models (LLMs)**.

This project showcases a complete data pipeline: from raw data cleaning and feature engineering, to training a custom ML model, to deploying a production-ready AI chatbot powered by the Groq API.

## ✨ Key Features

- **📊 Interactive Market Dashboard:** Visualizes key metrics (Total Listings, Average Prices, Top Locations) using Streamlit.
- **🔮 Price Prediction Engine:** A trained **Random Forest Regressor** (using Scikit-learn) that estimates the price of a property based on Location, Area, Bedrooms, and Bathrooms.
- **🤖 AI-Powered Chatbot:** Uses **Groq's LLM** (GPT-OSS-120b) to answer user questions intelligently based on the specific dataset and local Lahore market knowledge.
- **📈 Model Performance:** Displays R² Score and Mean Absolute Error (MAE) to validate the accuracy of the ML model directly in the app.


## 📂 Project Structure

```text
Lahore-Real-Estate-AI/
├── real_estate_app.py       # Main Streamlit Application
├── requirements.txt         # Python dependencies
├── lahore_data_CLEANED.csv  # The cleaned dataset used for training and analytics
├── .env                     # Your secret API key (DO NOT UPLOAD THIS TO GITHUB)
└── README.md                # Project documentation
