import streamlit as st
import pandas as pd

# 🌟 1. 페이지 설정 및 자동 번역 방지 로직
st.set_page_config(page_title="JAEWON QUANT", page_icon="🚀", layout="wide")

# 브라우저 자동 번역으로 인한 removeChild 에러 강제 방지
st.markdown("""
    <script>
        var meta = document.createElement('meta');
        meta.name = 'google';
        meta.content = 'notranslate';
        document.getElementsByTagName('head')[0].appendChild(meta);
    </script>
""", unsafe_allow_html=True)

# 🌟 2. 종목명 번역 사전 (재원 님 픽 포함)
TICKER_MAP = {
    "033170.KQ": "제룡전기", "214370.KQ": "케어젠", "094820.KQ": "일진파워",
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "042700.KS": "한미반도체",
    "APLD": "어플라이드 디지털", "ALAB": "아스테라랩스", "NVDA": "엔비디아",
    "TSLA": "테슬라", "AAPL": "애플", "MSFT": "마이크로소프트", "GOOGL": "구글",
    "AMZN": "아마존", "META": "메타", "AVGO": "브로드컴", "AMD": "AMD",
    "PLTR": "팔란티어", "MU": "마이크론", "INTC": "인텔", "QCOM": "퀄컴"
}

st.title("🚀 JAEWON QUANT AI RANKING")
st.write("실시간 뉴스와 차트를 분석한 내일의 상승 확률 리포트입니다.")

# 🌟 3. 데이터 로드
try:
    df = pd.read_csv('ai_report.csv')
    df['Prob'] = (df['Prob'] * 100).round(2)
    df['종목명'] = df['Ticker'].map(TICKER_MAP).fillna(df['Ticker'])
    df = df[['Market', '종목명', 'Ticker', 'Prob', 'Sentiment', 'PCR']]
except Exception:
    st.error("데이터 파일(ai_report.csv)을 찾을 수 없습니다.")
    st.stop()

# 🌟 4. 핵심 타겟 종목 카드
st.subheader("🔥 핵심 타겟 종목")
targets = ["어플라이드 디지털", "아스테라랩스", "제룡전기", "케어젠", "일진파워"]
target_df = df[df['종목명'].isin(targets)]

if not target_df.empty:
    cols = st.columns(len(targets))
    for i, name in enumerate(targets):
        with cols[i % len(cols)]:
            data = target_df[target_df['종목명'] == name]
            if not data.empty:
                st.metric(label=name, value=f"{data['Prob'].values[0]}%", delta=f"뉴스:{data['Sentiment'].values[0]}")

st.markdown("---")

# 🌟 5. 전체 랭킹 표
st.subheader("📊 전체 AI 예측 랭킹")
styled_df = df.style.background_gradient(cmap='RdYlGn', subset=['Prob']) \
                    .background_gradient(cmap='coolwarm', subset=['Sentiment'])
st.dataframe(styled_df, width=1500, height=600)