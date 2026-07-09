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
    """Fetches data, creates Advanced Features, and creates the Target"""
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="60d", interval="5m")
    
    if df.empty:
        return None
        
    # --- 1. RSI (Momentum) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- 2. MACD (Trend Momentum) ---
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # --- 3. BOLLINGER BANDS (Volatility Bounds) ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['Std_Dev_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Std_Dev_20'] * 2)
    
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['Price_to_BB_Lower'] = df['Close'] - df['BB_Lower']
    
    # --- 4. RETURNS ---
    df['Returns'] = df['Close'].pct_change()
    
    # --- 5. TARGET ---
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df = df.dropna()
    return df

def run_ml_bot(ticker_symbol):
    print(f"Initializing Upgraded XGBoost AI Engine for {ticker_symbol}...")
    
    # RISK MANAGEMENT CONFIGURATION
    stop_loss_pct = 0.005    # Cut losses immediately at 0.5%
    take_profit_pct = 0.015  # Lock in profits at 1.5%
    
    current_position = 0     # 0 = Flat, 1 = Long
    entry_price = 0.0
    last_processed_timestamp = None
    
    print(f"Live AI Tracking Active. Running checks every 5 minutes...")
    print(f"{'Time (IST)':<10} | {'Live Price':<10} | {'AI Confidence':<14} | {'Signal':<13} | {'Action'}")
    print("-" * 85)

    while True:
        try:
            df = prepare_data(ticker_symbol)
            
            if df is not None and not df.empty:
                latest_row = df.iloc[-1]
                latest_timestamp = df.index[-1]
                live_price = float(latest_row['Close'])
                
                # --- STEP 1: ACTIVE RISK MANAGEMENT CHECK ---
                if current_position == 1:
                    # Check if we hit Stop Loss
                    if live_price <= entry_price * (1 - stop_loss_pct):
                        if trading_client:
                            trading_client.close_all_positions(cancel_orders=True)
                        print(f"{datetime.now().strftime('%H:%M:%S'):<10} | {live_price:<10.2f} | N/A            | STOP LOSS 🚨  | CLOSED LONG AT LOSS")
                        current_position = 0
                        entry_price = 0.0
                        
                    # Check if we hit Take Profit
                    elif live_price >= entry_price * (1 + take_profit_pct):
                        if trading_client:
                            trading_client.close_all_positions(cancel_orders=True)
                        print(f"{datetime.now().strftime('%H:%M:%S'):<10} | {live_price:<10.2f} | N/A            | TAKE PROFIT 💰| CLOSED LONG WITH PROFIT")
                        current_position = 0
                        entry_price = 0.0

                # --- STEP 2: NEW CANDLE PREDICTION LOGIC ---
                if latest_timestamp != last_processed_timestamp:
                    last_processed_timestamp = latest_timestamp
                    
                    train_df = df.iloc[:-1] 
                    live_candle = df.iloc[[-1]] 
                    
                    features = ['RSI', 'MACD', 'MACD_Histogram', 'BB_Width', 'Price_to_BB_Lower', 'Returns']
                    
                    X_train = train_df[features]
                    y_train = train_df['Target']
                    X_live = live_candle[features]
                    
                    # Train Model
                    model = xgb.XGBClassifier(
                        n_estimators=100,     
                        learning_rate=0.1,    
                        max_depth=4,          
                        random_state=42,
                        eval_metric='logloss'
                    )
                    model.fit(X_train, y_train)
                    
                    # Predict
                    prediction_prob = model.predict_proba(X_live)[0]
                    up_probability = prediction_prob[1]
                    
                    signal = "HOLD  ⚪"
                    action_taken = "None"
                    trade_symbol = ticker_symbol.replace(".NS", "")
                    
                    # AI Core Signals
                    if up_probability > 0.65 and current_position == 0:
                        signal = "LONG  🟢"
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
                        entry_price = live_price
                            
                    elif up_probability < 0.35 and current_position == 1:
                        signal = "SELL 🔴"
                        if trading_client:
                            trading_client.close_all_positions(cancel_orders=True)
                            action_taken = "CLOSED LONG"
                        else:
                            action_taken = "SIMULATED SELL"
                        
                        current_position = 0
                        entry_price = 0.0
                            
                    time_str = datetime.now().strftime('%H:%M:%S')
                    conf_str = f"{up_probability * 100:.1f}% UP" 
                    print(f"{time_str:<10} | {live_price:<10.2f} | {conf_str:<14} | {signal:<13} | {action_taken}")
                    
        except Exception as e:
            print(f"Error executing AI loop: {e}. Retrying...")
            
        time.sleep(300)

if __name__ == "__main__":
    run_ml_bot("HDFCBANK.NS")
