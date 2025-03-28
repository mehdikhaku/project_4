S&P500 PRICE PREDICTION 

Project Overview
This project focuses on predicting the S&P 500 index using a Linear Regression model, LSTM, Linear_ Spark. Utilizing a hybrid model incorporating LSTM neural networks, financial sentiment analysis, and volatility measures. It leverages historical stock market data to identify patterns and trends that influence market movements. The goal is to build a predictive model that can estimate future S&P 500 values based on relevant financial indicators.
Objective
To build a predictive model using Python and machine learning techniques to analyze historical S&P 500 data and forecast future index values.

Technologies used
1. Data Collection & Storage
•	Google Cloud – Used for storing large datasets, cloud computing, and running scalable machine learning models using services like BigQuery, Cloud Storage, and Vertex AI.
2. Data Processing & Analysis
•	Python – Core programming language for data wrangling, analysis, and model building.
•	Jupyter Notebook – Interactive environment for writing and running Python code, mainly for data exploration and prototyping.
•	Pandas & NumPy (implicitly used) – Essential for data manipulation and numerical operations.
3. Data Visualization
•	Tableau – Creates interactive dashboards and visual analytics for business intelligence and reporting.
•	Matplotlib & Seaborn – Used for static visualizations, such as line charts, histograms, heatmaps, and scatter plots within Python.
4. Machine Learning & Modeling
•	Scikit-learn – Provides machine learning models for classification, regression, clustering, and preprocessing.
•	LSTM (via TensorFlow/PyTorch, not listed but likely used) – For time-series forecasting, particularly for your S&P 500 prediction project.
5. Deployment & Sharing
•	Gradio – Creates interactive UIs for machine learning models, making them accessible via a web interface.
•	ngrok – Exposes local Gradio apps or Jupyter Notebooks to the internet for external access and testing.

Team Members
•	[Mohamedmehdi Khaku, Daniela Molina, Israa Adam, Jennifer Giraldo, Marie Louies Iraba and Pratiksha]

Project Requirements & Deliverables
 Data Model Implementation
•	Python script initializes, trains, and evaluates a model
•	Data cleaning, normalization, and standardization steps
•	Model uses data retrieved from relevant financial datasets
•	The model utilizes data retrieved from Spark 
•	The model demonstrates meaningful predictive power at least 75% classification accuracy or 0.80 R-squared

 Outlines & Visualizations
S&P 500 Prediction for the last 3 years using linear regression
This project focuses on analyzing and predicting the S&P 500 index using historical data. The approach combines data analysis, visualization, and machine learning techniques to identify market trends and patterns.
Features
•	Data Loading & Cleaning: Reads and preprocesses historical S&P 500 data from SP500_3yr.csv.
•	Exploratory Data Analysis (EDA): Visualizes trends using Matplotlib and Seaborn.
•	Predictive Modeling: Implements machine learning models (e.g., LSTMs, linear regression) to forecast market movements.
•	Volatility & Sentiment Analysis: Incorporates financial sentiment and volatility measures for enhanced predictions
Results
Visualization of S&P 500 trends, Performance evaluation of predictive models and insights into market behavior based on historical data.

Comparison of Closing Price data from 2015-2025
 
 Model Architecture & Training
ML model: SARIMA (Seasonal Autoregressive Integrated Moving Average)
Why: It is a time series forecasting model designed to handle data with seasonal patterns. captures both short-term and long-term dependencies within the data and combines the concepts of autoregressive (AR), integrated (I), and moving average (MA) models with seasonal components to make forecasts.
Hyperparameters used:
 (p): AR component of order p: 1. MA(q): MA component of order q: 1. (d): Integrated component of order d:1. (P): Seasonal AR component of order P:1. (Q): Seasonal MA component of order Q:1
(D): Seasonal I component of order D:1 (S): Seasonal period:18

Key Findings:
The model demonstrates an R² score of ≥ 0.80, indicating a strong correlation between the features and the target variable. The model predicted the highest performing month as November. Limitations & Future Improvements:
Incorporate additional market indicators Experiment with advanced models (Prophet, Convolutional Neural Networks) for better predictive performance.

Result:
The SARIMAX model predicts future index values based on historical stock closing data. The dataset includes S&P 500 prices from 2015 to 2025. 

 S&P 500 Prediction Using Linear Regression & Spark
This project predicts the next-day closing price of the S&P 500 using Apache Spark for data processing and Scikit-learn for modeling. It incorporates stock prices, the VIX index, and news sentiment analysis.
 Key Features
- **Data Processing:** Merge S&P 500, VIX, and sentiment data with PySpark  
- **Feature Engineering:** Calculate **RSI (Relative Strength Index)**  
- **Modeling:** Train a **Linear Regression** model and evaluate performance  
- **Prediction:** Forecast the next day's closing price  
 Performance
- **MAE:** 13.08 | **MSE:** 253.15 | **R²:** 0.9973  
- **Example Prediction:** Next-day close = **$5135.99** (-0.82%)  
How to Run
1. Install dependencies:  
   ```bash
   pip install pyspark pandas numpy matplotlib seaborn scikit-learn
Result
**MAE:** 13.08 | **MSE:** 253.15 | **R²:** 0.9973  
- **Example Prediction:** Next-day close = **$5135.99** (-0.82%)
After building the models, we used Gradio to build the interactive UI for our machine learning models due to its simplicity and ease of public sharing. Although Gradio automatically generates public links (gradio.live), we opted to use ngrok for a more stable and secure connection when running the server locally. If you're running this project in Google Colab, Gradio will default to its built-in sharing mechanism and bypass ngrok.
 
 S&P 500 Prediction Using LSTM & Sentiment Analysis
This project predicts the next-day closing price of the **S&P 500 Index** using a **Long Short-Term Memory (LSTM)** neural network. It incorporates historical stock prices, **Relative Strength Index (RSI)**, **VIX Index**, and **news sentiment scores** to improve forecast accuracy.
 Features  
- **Data Preprocessing:**  
  - Merge S&P 500, VIX, and sentiment data  
  - Compute **RSI (Relative Strength Index)**  
  - Normalize data using **MinMaxScaler**  
- **LSTM Neural Network:**  
  - Uses a **30-day lookback window** for sequence modeling  
  - **Dropout** and **L2 regularization** to prevent overfitting  
  - **Leaky ReLU** activation for better gradient flow  
  - **Early stopping** to optimize training  
- **Prediction & Evaluation:**  
  - Inverse transform predictions to real prices  
  - Evaluate with **MAE, MSE, and R² score**  
 Model Performance  
- **MAE:** _TBD_  
- **MSE:** _TBD_  
- **R² Score:** _TBD_  
 How to Run  
1. Install dependencies:  
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib joblib 
Result
Mean Absolute Error (MAE): 54.37 Mean Squared Error (MSE): 4681.00 R^2 Score: 0.83
Tableau:
This Tableau project was created to visually explore the relationship between the S&P 500 closing price, market volatility (VIX), and news sentiment over time. The goal was to better understand how emotional factors and volatility may align with or help predict shifts in the market.
The dashboard features three interactive line and area charts that display:
•	Daily or monthly S&P 500 prices
•	VIX closing values, representing expected volatility
•	News sentiment scores, normalized from 0 to 100 for comparison
We included interactive filters for Year and Month, allowing users to narrow in on specific time periods and analyze how these indicators behaved during notable market events. Hover-enabled tooltips display key statistics for each time point, such as average closing price, sentiment score, and volatility level.
The visualizations allow users to explore short-term and long-term patterns — such as how sentiment changes around price drops or how VIX spikes may align with major market swings. This project helps make complex financial data more accessible, while inviting further analysis into how public emotion and risk perception influence market behavior.


Summary:
The S&P 500 Prediction model utilizes Linear Regression to estimate future index values based on historical stock data. The dataset includes S&P 500 prices, annual percentage changes, dividends, total returns, and the VIX (Volatility Index).
Key Findings:
•	The model demonstrates an R² score of ≥ 0.80, indicating a strong correlation between the features and the target variable.
•	Feature scaling and data preprocessing improved prediction accuracy.
•	The VIX (Volatility Index) played a significant role in capturing market uncertainty.
•	The model effectively captures long-term trends but may struggle with short-term market volatility.
Limitations & Future Improvements:
•	Incorporate additional market indicators like interest rates, inflation, and macroeconomic trends.
•	Experiment with advanced models (Random Forest, Neural Networks) for better predictive performance.
•	Implement time-series forecasting techniques to capture sequential market behaviors.

Next steps for improvement:
·Trying other models like Prophet by Facebook or RandomForest
·Incorporating additional indicators like MACD and Bollinger Bands could further improve accuracy.
·Experiment with longer look-back periods.
·Tune hyperparameters using grid search.
·Develop a deployment-ready prediction interface.
