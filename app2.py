import streamlit as st
import pandas as pd

# 🌟 1. 티커 -> 한국어 종목명 자동 번역 사전
TICKER_MAP = {
    # 한국 주식
    "033170.KQ": "제룡전기", "214370.KQ": "케어젠", "094820.KQ": "일진파워",
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "042700.KS": "한미반도체",
    "259960.KS": "크래프톤", "010140.KS": "삼성중공업", "247540.KQ": "에코프로비엠",
    "086520.KQ": "에코프로", "034020.KS": "두산에너빌리티", "011780.KS": "금호석유",
    "010950.KS": "S-Oil", "035420.KS": "NAVER", "035720.KS": "카카오",
    "005380.KS": "현대차", "000270.KS": "기아", "051910.KS": "LG화학",
    "207940.KS": "삼성바이오로직스", "068270.KS": "셀트리온", "000100.KS": "유한양행",
    "105560.KS": "KB금융", "055550.KS": "신한지주",
    # 미국 주식
    "APLD": "어플라이드 디지털", "ALAB": "아스테라랩스", "NVDA": "엔비디아",
    "TSLA": "테슬라", "AAPL": "애플", "MSFT": "마이크로소프트", "GOOGL": "구글",
    "AMZN": "아마존", "META": "메타", "AVGO": "브로드컴", "AMD": "AMD",
    "ARM": "ARM", "SMCI": "슈퍼마이크로", "PLTR": "팔란티어", "MU": "마이크론",
    "INTC": "인텔", "QCOM": "퀄컴", "ADBE": "어도비", "CRM": "세일즈포스",
    "NOW": "서비스나우", "SNPS": "시놉시스", "CDNS": "케이던스", "PANW": "팔로알토",
    "CRWD": "크라우드", "PLD": "프로로지스"
}

st.set_page_config(page_title="JAEWON QUANT", page_icon="🚀", layout="wide")

st.title("🚀 JAEWON QUANT AI SCANNED RANKING")
st.markdown("딥러닝 마스터 브레인이 실시간 뉴스와 차트를 분석한 내일의 상승 확률 리포트입니다.")
st.markdown("---")

try:
    df = pd.read_csv('ai_report.csv')
    df['Prob'] = (df['Prob'] * 100).round(2)
    
    # 🌟 2. 번역 사전을 이용해 '종목명' 컬럼 새로 만들기
    df['종목명'] = df['Ticker'].map(TICKER_MAP).fillna(df['Ticker'])
    
    # 보기 좋게 컬럼 순서 재배치
    df = df[['Market', '종목명', 'Ticker', 'Prob', 'Sentiment', 'PCR']]
except FileNotFoundError:
    st.error("❌ 'ai_report.csv' 파일이 없습니다. 데일리 스캐너(파일 2)를 먼저 실행해 주세요.")
    st.stop()

# 🌟 3. 재원 님 핵심 타겟 종목 전용 대시보드 (한국어로 표시)
st.subheader("🔥 재원's 핵심 타겟 종목 현황")
target_names = ["어플라이드 디지털", "아스테라랩스", "제룡전기", "케어젠", "일진파워"]
target_df = df[df['종목명'].isin(target_names)]

if not target_df.empty:
    cols = st.columns(len(target_names))
    for i, name in enumerate(target_names):
        with cols[i % len(cols)]:
            stock_data = target_df[target_df['종목명'] == name]
            if not stock_data.empty:
                prob = stock_data['Prob'].values[0]
                sentiment = stock_data['Sentiment'].values[0]
                delta_color = "normal" if prob >= 50 else "inverse"
                st.metric(label=f"🎯 {name}", value=f"{prob}%", delta=f"뉴스 점수: {sentiment}", delta_color=delta_color)
            else:
                st.metric(label=f"🎯 {name}", value="분석 대기", delta="-")
else:
    st.info("타겟 종목 데이터가 아직 생성되지 않았습니다.")

st.markdown("---")

# 🌟 4. 전체 시장 AI 예측 랭킹
st.subheader("📊 전체 시장 AI 예측 랭킹 (클릭해서 정렬 가능)")

styled_df = df.style.background_gradient(cmap='RdYlGn', subset=['Prob']) \
                    .background_gradient(cmap='coolwarm', subset=['Sentiment']) \
                    .format({'Prob': '{:.2f}%', 'Sentiment': '{:.3f}', 'PCR': '{:.2f}'})

st.dataframe(styled_df, width='stretch', height=600)