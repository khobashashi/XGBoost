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
    """Fetches data, creates Advanced Features (clues), and creates the Target"""
    df = yf.download(ticker_symbol, period="60d", interval="15m", progress=False)
    
    if df.empty:
        return None
        
    # --- 1. RSI (Momentum / Overbought & Oversold) ---
    delta = df['Close'].diff()
    # Using Wilder's Smoothing for true RSI calculation
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- 2. MACD (Trend Direction & Strength) ---
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # --- 3. BOLLINGER BANDS (Volatility & Snap-back points) ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['Std_Dev_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Std_Dev_20'] * 2)
    
    # We feed the AI the "Width" of the bands and how close the price is to the bottom
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['Price_to_BB_Lower'] = df['Close'] - df['BB_Lower']
    
    # --- 4. BASIC PRICE ACTION ---
    df['Returns'] = df['Close'].pct_change()
    
    # --- 5. TARGET CREATION (The Answer Key) ---
    # If the NEXT candle closes higher, Target = 1. Else, Target = 0.
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop all rows that have blank data from the rolling calculations
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
                    features = [
                        'RSI', 
                        'MACD', 
                        'MACD_Histogram', 
                        'BB_Width', 
                        'Price_to_BB_Lower', 
                        'Returns'
                    ]
                    
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
