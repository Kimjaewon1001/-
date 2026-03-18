import os
import sys
import warnings
import time
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import feedparser
import joblib
from tensorflow.keras.models import load_model
from transformers import pipeline

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("\n" + "="*60)
print("🚀 [JAEWON QUANT] 데일리 스캐너 가동 (기억상실증 방지 버전)")
print("="*60)

FEATURE_COLUMNS = [
    'Close', 'Volume', 'SMA_20', 'EMA_20', 'VWAP', 'BB_High', 'BB_Low', 
    'ATR', 'MACD', 'RSI',                  
    'PER', 'PBR', 'Market_Cap', 'Div_Yield', 
    'News_Sentiment', 'DTE', 'PCR', 'Max_Pain_Diff'          
]

try:
    print("📰 실시간 뉴스 분석 AI 시동 중...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
except Exception as e:
    print(f"❌ 뉴스 엔진 로드 실패: {e}")
    sys.exit()

def get_real_news_sentiment(ticker_symbol):
    titles = []
    try:
        t = yf.Ticker(ticker_symbol)
        if t.news: titles = [n.get('title', '') for n in t.news[:15]]
        
        search_name = ticker_symbol.split('.')[0]
        rss_url = f"https://news.google.com/rss/search?q={search_name}+stock+news&hl=en-US"
        feed = feedparser.parse(rss_url)
        titles += [entry.title for entry in feed.entries[:15]]

        if not titles: return 0.0

        total_score = 0
        for title in set(titles):
            if not title: continue
            res = sentiment_analyzer(title[:512])[0]
            prob = res['score']
            if res['label'] == 'positive': total_score += prob
            elif res['label'] == 'negative': total_score -= prob
            else: total_score += (prob * 0.01)

        return round(total_score / len(titles), 4)
    except: return 0.0

def get_real_options_data(ticker_symbol, current_price, sma_20):
    dte, pcr, max_pain_diff = 15.0, 1.0, 0.0
    try:
        t = yf.Ticker(ticker_symbol)
        opt_dates = t.options
        if opt_dates:
            from datetime import datetime
            exp_date = datetime.strptime(opt_dates[0], '%Y-%m-%d')
            dte = float((exp_date - datetime.now()).days)
            if dte < 0: dte = 0.0
            
            opt_chain = t.option_chain(opt_dates[0])
            puts, calls = opt_chain.puts, opt_chain.calls
            
            put_vol = puts['volume'].sum() if 'volume' in puts else 0
            call_vol = calls['volume'].sum() if 'volume' in calls else 0
            if call_vol > 0: pcr = float(put_vol / call_vol)
            
            if not calls.empty and not puts.empty:
                max_pain_strike = calls.loc[calls['volume'].idxmax()]['strike']
                max_pain_diff = float((current_price - max_pain_strike) / max_pain_strike * 100)
    except:
        if current_price > 0 and sma_20 > 0:
            max_pain_diff = float((current_price - sma_20) / sma_20 * 100)
    return dte, pcr, max_pain_diff

def run_daily_scanner():
    if not os.path.exists('KOREA_master.keras'):
        print("❌ 'KOREA_master.keras'가 없습니다. 파일 1을 먼저 돌려주세요.")
        return

    print("🧠 마스터 브레인 및 스케일러 로드 중...")
    k_model = load_model('KOREA_master.keras', compile=False)
    n_model = load_model('NASDAQ_master.keras', compile=False)
    k_scaler = joblib.load('KOREA_scaler.pkl')
    n_scaler = joblib.load('NASDAQ_scaler.pkl')
    
    k_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    n_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 🌟 재원 님 픽: APLD, ALAB, 제룡전기 등 최우선 배치
    scan_list = {
        "KOREA": [
            "033170.KQ", "214370.KQ", "094820.KQ", 
            "005930.KS", "000660.KS", "042700.KS", "259960.KS", "010140.KS", 
            "247540.KQ", "086520.KQ", "034020.KS", "011780.KS", "010950.KS",
            "035420.KS", "035720.KS", "005380.KS", "000270.KS", "051910.KS",
            "207940.KS", "068270.KS", "000100.KS", "105560.KS", "055550.KS"
        ],
        "NASDAQ": [
            "APLD", "ALAB", 
            "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
            "AVGO", "AMD", "ARM", "SMCI", "PLTR", "MU", "INTC", "QCOM",
            "ADBE", "CRM", "NOW", "SNPS", "CDNS", "PANW", "CRWD", "PLD"
        ]
    }
    
    final_picks = []

    for market, tickers in scan_list.items():
        print(f"\n📊 [{market}] 실시간 스캔 중...")
        model = n_model if market == "NASDAQ" else k_model
        scaler = n_scaler if market == "NASDAQ" else k_scaler

        for ticker in tickers:
            print(f"  > {ticker} 스캔 중...", end=" ", flush=True)
            try:
                t = yf.Ticker(ticker)
                df = t.history(period="60d") 
                if len(df) < 35:
                    print("❌ 데이터 부족")
                    continue
                    
                df.index = df.index.tz_localize(None)
                current_price = df['Close'].iloc[-1]
                
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
                df.dropna(inplace=True)

                news_score = get_real_news_sentiment(ticker)
                dte, pcr, mp_diff = get_real_options_data(ticker, current_price, df['SMA_20'].iloc[-1])
                
                df['News_Sentiment'] = news_score
                df['DTE'] = dte
                df['PCR'] = pcr
                df['Max_Pain_Diff'] = mp_diff

                scaled_data = scaler.transform(df[FEATURE_COLUMNS])

                # 🌟 [수정 포인트] AI가 뇌 정지를 겪지 않도록 실시간 학습(model.fit)을 주석 처리하여 안전하게 보호!
                window_size = 30
                # yesterday_input = np.expand_dims(scaled_data[-(window_size+1):-1], axis=0)
                # today_actual = 1 if df['Close'].iloc[-1] > df['Close'].iloc[-2] else 0
                # model.fit(yesterday_input, np.array([[today_actual]]), epochs=1, verbose=0) 

                # 🎯 내일 예측
                today_input = np.expand_dims(scaled_data[-window_size:], axis=0)
                prob = model.predict(today_input, verbose=0)[0][0]
                
                final_picks.append({'Ticker': ticker, 'Prob': prob, 'Sentiment': news_score, 'PCR': pcr, 'Market': market})
                print(f"✅ (뉴스: {news_score:.2f} | 확률: {prob*100:.1f}%)")
            except Exception as e:
                print(f"❌ 실패")

    # 🌟 [수정 포인트] 뇌가 망가지는 것을 막았으므로, 덮어쓰기 저장도 꺼둡니다.
    # k_model.save('KOREA_master.keras')
    # n_model.save('NASDAQ_master.keras')

    report_df = pd.DataFrame(final_picks).sort_values(by='Prob', ascending=False)
    report_df.to_csv('ai_report.csv', index=False)
    
    print("\n" + "="*60 + "\n🏆 오늘의 실시간 퀀트 랭킹 (ai_report.csv 저장 완료)\n" + "="*60)
    for i, row in report_df.head(15).iterrows():
        print(f"[{row['Market']}] {row['Ticker']:<10} | 내일 상승확률: {row['Prob']*100:.1f}% | 뉴스 점수: {row['Sentiment']:.3f} | PCR: {row['PCR']:.2f}")

if __name__ == "__main__":
    run_daily_scanner()