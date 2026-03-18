import os
import sys
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("\n" + "="*60)
print("🧠 [JAEWON QUANT] 마스터 브레인 학습 가동 (뇌 정지 방지 및 자신감 부여)")
print("="*60)

# 🌟 18개 핵심 지표 (완벽 유지)
FEATURE_COLUMNS = [
    'Close', 'Volume', 'SMA_20', 'EMA_20', 'VWAP', 'BB_High', 'BB_Low', 
    'ATR', 'MACD', 'RSI',                  
    'PER', 'PBR', 'Market_Cap', 'Div_Yield', 
    'News_Sentiment', 'DTE', 'PCR', 'Max_Pain_Diff'          
]

def prepare_data(tickers, market_name):
    print(f"\n📥 [{market_name}] 5년 치 통합 데이터 및 뉴스/옵션 역산 추출 중...")
    all_X, all_y = [], []
    combined_df = pd.DataFrame()

    for ticker in tickers:
        print(f"  > {ticker} 추출 중...", end=" ", flush=True)
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="5y")
            if len(df) < 200:
                print("❌ 상장 기간 부족")
                continue
            
            info = t.info
            df['PER'] = info.get('trailingPE', 15.0) 
            df['PBR'] = info.get('priceToBook', 1.5)
            df['Market_Cap'] = np.log1p(info.get('marketCap', 1e10)) 
            df['Div_Yield'] = info.get('dividendYield', 0.0)
            
            df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], 20).sma_indicator()
            df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], 20).ema_indicator()
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['VWAP'] = (df['Typical_Price'] * df['Volume']).rolling(5).sum() / df['Volume'].rolling(5).sum()
            bb = ta.volatility.BollingerBands(df['Close'], 20)
            df['BB_High'], df['BB_Low'] = bb.bollinger_hband(), bb.bollinger_lband()
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
            
            # 뉴스/옵션 5년 치 과거 역산
            df['DTE'] = 20 - (np.arange(len(df)) % 20)
            df['PCR'] = 1.0 - ((df['RSI'] - 50) / 100)
            df['Max_Pain_Diff'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
            daily_return = df['Close'].pct_change()
            df['News_Sentiment'] = np.clip(daily_return * 20, -1.0, 1.0)
            
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df.dropna(inplace=True)
            
            combined_df = pd.concat([combined_df, df])
            print("✅ 완료")
        except Exception as e:
            print("❌ 에러 발생")

    if combined_df.empty: return np.array([]), np.array([])

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_df[FEATURE_COLUMNS])
    joblib.dump(scaler, f'{market_name}_scaler.pkl')

    window_size = 30
    targets = combined_df['Target'].values
    for i in range(window_size, len(scaled_data)):
        all_X.append(scaled_data[i-window_size:i])
        all_y.append(targets[i-1])

    return np.array(all_X), np.array(all_y)

def build_and_train(X, y, market_name):
    print(f"\n🧬 [{market_name}] 딥러닝 시작 (데이터: {len(X)}일 치)")
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 🌟 51.8% 탈출을 위한 족쇄 해제 모델! (L2 정규화 제거, Dropout 0.2로 하향)
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        MaxPooling1D(pool_size=2),
        
        LSTM(128, return_sequences=True), 
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(64),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    ]
    
    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=callbacks)
    
    model.save(f'{market_name}_master.keras')
    print(f"✔️ [{market_name}] 마스터 브레인 저장 완료 (.keras)")

if __name__ == "__main__":
    korea_tickers = [
        "033170.KQ", "094480.KQ", "214370.KQ", "005930.KS", "000660.KS", 
        "035420.KS", "005380.KS", "035720.KS", "051910.KS", "000270.KS", 
        "005490.KS", "032830.KS", "068270.KS", "000810.KS", "015760.KS", 
        "012330.KS", "011780.KS", "010130.KS", "010950.KS", "003550.KS", 
        "009150.KS", "034220.KS", "018260.KS", "000100.KS", "000720.KS"
    ]
    
    nasdaq_tickers = [
        "APLD", "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "AVGO", "NFLX", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", 
        "LRCX", "ADI", "PANW", "SNPS", "CDNS", "CSCO", "ORCL", "ADBE"
    ]

    X_kr, y_kr = prepare_data(korea_tickers, "KOREA")
    if len(X_kr) > 0: build_and_train(X_kr, y_kr, "KOREA")

    X_us, y_us = prepare_data(nasdaq_tickers, "NASDAQ")
    if len(X_us) > 0: build_and_train(X_us, y_us, "NASDAQ")
    
    print("\n🎉 파일 1: 완벽한 마스터 브레인 구축 완료!")