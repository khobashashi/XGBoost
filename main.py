import time
import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb

# Alpaca Imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Securely load Alpaca Keys
API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

if API_KEY and SECRET_KEY:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
else:
    print("⚠️ Alpaca Keys missing. Running in simulation mode.")
    trading_client = None

def prepare_data(ticker_symbol):
    """Fetches data, creates Features (clues), and creates the Target (answers)"""
    # Pull 60 days of 15-minute data to give the AI enough history to learn from
    df = yf.download(ticker_symbol, period="60d", interval="15m", progress=False)
    
    if df.empty:
        return None
        
    # --- 1. FEATURE ENGINEERING (The Clues) ---
    # We give the AI moving averages, volatility, and momentum
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['Price_to_SMA'] = df['Close'] / df['SMA_10']
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['Returns'] = df['Close'].pct_change()
    
    # --- 2. TARGET CREATION (The Answer Key) ---
    # If the NEXT 15-minute candle's close is higher than the CURRENT close, Target = 1 (UP)
    # Otherwise, Target = 0 (DOWN)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop rows with NaN values created by moving averages and shifting
    df = df.dropna()
    return df

def run_ml_bot(ticker_symbol):
    print(f"Initializing XGBoost Machine Learning Model for {ticker_symbol}...")
    
    current_position = 0
    last_processed_timestamp = None
    
    print(f"Live AI Tracking Started. Checking and predicting every 15 minutes...")
    print(f"{'Time (IST)':<10} | {'Live Price':<10} | {'AI Confidence':<14} | {'Signal':<8} | {'Action'}")
    print("-" * 75)

    while True:
        try:
            # 1. Fetch data and build features
            df = prepare_data(ticker_symbol)
            
            if df is not None and not df.empty:
                latest_timestamp = df.index[-1]
                
                # Ensure we only trade once per 15-minute candle
                if latest_timestamp != last_processed_timestamp:
                    last_processed_timestamp = latest_timestamp
                    
                    # 2. SEPARATE PAST DATA (TRAINING) FROM PRESENT DATA (PREDICTING)
                    # We train the model on everything EXCEPT the very last row
                    train_df = df.iloc[:-1] 
                    # The very last row is right now. We don't know the future yet!
                    live_candle = df.iloc[[-1]] 
                    
                    # Define our feature columns
                    features = ['SMA_10', 'SMA_30', 'Price_to_SMA', 'Volatility', 'Returns']
                    
                    X_train = train_df[features]
                    y_train = train_df['Target']
                    X_live = live_candle[features]
                    
                    # 3. TRAIN THE XGBOOST MODEL
                    # This builds hundreds of decision trees instantly
                    model = xgb.XGBClassifier(
                        n_estimators=100,     # Build 100 trees
                        learning_rate=0.1,    # The step size for correcting errors
                        max_depth=4,          # How deep each tree can go
                        random_state=42,
                        eval_metric='logloss'
                    )
                    model.fit(X_train, y_train)
                    
                    # 4. PREDICT THE LIVE MARKET
                    live_price = float(live_candle['Close'].iloc[0])
                    # Predicts probability [Chance of going DOWN, Chance of going UP]
                    prediction_prob = model.predict_proba(X_live)[0]
                    up_probability = prediction_prob[1]
                    
                    signal = "HOLD  ⚪"
                    action_taken = "None"
                    trade_symbol = ticker_symbol.replace(".NS", "")
                    
                    # 5. EXECUTION LOGIC
                    # If the AI is over 65% confident the stock is going up -> BUY
                    if up_probability > 0.65:
                        signal = "LONG  🟢"
                        if current_position <= 0:
                            if trading_client:
                                order = MarketOrderRequest(
                                    symbol=trade_symbol,
                                    qty=1,
                                    side=OrderSide.BUY,
                                    time_in_force=TimeInForce.GTC
                                )
                                trading_client.submit_order(order)
                                action_taken = "EXECUTED BUY"
                            else:
                                action_taken = "SIMULATED BUY"
                            current_position = 1
                            
                    # If the AI is over 65% confident the stock is going down -> SELL
                    elif up_probability < 0.35:
                        signal = "SELL 🔴"
                        if current_position == 1:
                            if trading_client:
                                trading_client.close_all_positions(cancel_orders=True)
                                action_taken = "CLOSED LONG"
                            else:
                                action_taken = "SIMULATED SELL"
                            current_position = 0
                            
                    time_str = datetime.now().strftime('%H:%M:%S')
                    # Format probability to show as a percentage (e.g., 68.5%)
                    conf_str = f"{up_probability * 100:.1f}% UP" 
                    print(f"{time_str:<10} | {live_price:<10.2f} | {conf_str:<14} | {signal:<8} | {action_taken}")
                    
        except Exception as e:
            print(f"Error fetching data or training model: {e}. Retrying...")
            
        # Sleep for 5 minutes before pulling data and retraining the model
        time.sleep(300)

if __name__ == "__main__":
    # Tracking Reliance Industries. (Remove .NS if trading US stocks)
    run_ml_bot("RELIANCE.NS")
