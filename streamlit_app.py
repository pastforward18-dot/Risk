
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px

TRADING_DAYS = 252

st.set_page_config(
    page_title="Portfolio Risk Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {
    max-width: 1380px;
    padding-top: .65rem;
    padding-bottom: 2rem;
}
h1 { font-size: 2.0rem !important; }
div[data-testid="stMetric"] {
    border: 1px solid rgba(49,51,63,.15);
    border-radius: 14px;
    padding: 10px 12px;
    min-height: 88px;
}
.stButton button, .stDownloadButton button {
    min-height: 44px;
    border-radius: 12px;
    font-weight: 600;
}
div[data-testid="stTabs"] button {
    min-height: 44px;
    padding-left: 12px;
    padding-right: 12px;
}
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
@media (max-width: 1180px) {
    .block-container { padding-left: .85rem; padding-right: .85rem; }
    h1 { font-size: 1.75rem !important; }
}
@media (max-width: 820px) {
    .block-container { padding-left: .5rem; padding-right: .5rem; }
    h1 { font-size: 1.5rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.25rem !important; }
    div[data-testid="stTabs"] { overflow-x: auto; }
}
</style>
""", unsafe_allow_html=True)

def normalize(w):
    w = np.asarray(w, dtype=float)
    s = np.nansum(w)
    return w / s if s > 0 else np.repeat(1 / max(len(w),1), len(w))

def sharpe(r, rf=0.0):
    r = r.dropna()
    if len(r) < 5 or r.std(ddof=1) == 0:
        return np.nan
    rf_d = (1 + rf) ** (1 / TRADING_DAYS) - 1
    return (r.mean() - rf_d) / r.std(ddof=1) * np.sqrt(TRADING_DAYS)

def ann_vol(r):
    r = r.dropna()
    return r.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(r) > 2 else np.nan

def ann_return(r):
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    total = (1 + r).prod()
    years = len(r) / TRADING_DAYS
    if total <= 0 or years <= 0:
        return np.nan
    return total ** (1 / years) - 1

def max_drawdown(r):
    r = r.dropna()
    if r.empty:
        return np.nan
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    return (wealth / peak - 1).min()

def var_cvar(r, alpha=.95):
    r = r.dropna()
    if len(r) < 10:
        return np.nan, np.nan
    q = np.quantile(r, 1 - alpha)
    tail = r[r <= q]
    return -q, (-tail.mean() if len(tail) else np.nan)

def risk_contribution(weights, cov):
    w = np.asarray(weights, dtype=float)
    pvol = math.sqrt(max(float(w.T @ cov @ w), 0))
    if pvol <= 0:
        return np.zeros_like(w), np.zeros_like(w)
    mrc = cov @ w / pvol
    crc = w * mrc
    return crc, crc / pvol

def risk_parity_weights(cov, max_weight=1.0):
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    x0 = np.repeat(1 / n, n)
    target = np.repeat(1 / n, n)

    def obj(w):
        _, pct = risk_contribution(w, cov)
        return np.sum((pct - target) ** 2)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return normalize(res.x) if res.success else x0

def avg_corr(corr):
    if corr.shape[0] <= 1:
        return 0.0
    vals = corr.values[np.triu_indices_from(corr.values, k=1)]
    return float(np.nanmean(vals))

def diversification_score(weights, corr):
    w = normalize(weights)
    hhi = np.sum(w ** 2)
    hhi_min = 1 / len(w)
    conc = (hhi - hhi_min) / max(1 - hhi_min, 1e-9)
    ac = avg_corr(corr)
    ac_scaled = np.clip((ac + 1) / 2, 0, 1)
    score = 100 * (1 - 0.55 * conc - 0.45 * ac_scaled)
    return int(np.clip(round(score), 0, 100))

def fmt_pct(x, d=1):
    return "—" if pd.isna(x) else f"{x*100:.{d}f}%"

def fmt_num(x, d=2):
    return "—" if pd.isna(x) else f"{x:.{d}f}"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(ticker, period):
    return yf.Ticker(ticker).history(period=period, auto_adjust=True)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fx(period):
    return yf.Ticker("KRW=X").history(period=period, auto_adjust=True)

st.title("📊 Portfolio Risk Terminal")
st.caption("iPad · Safari 최적화 / 한국·미국 주식 / Sharpe / Risk Parity / 상관계수 / MDD / VaR·CVaR")

with st.expander("⚙️ 분석 설정", expanded=False):
    c1, c2 = st.columns(2)
    period = c1.selectbox("가격 히스토리", ["1y", "2y", "5y"], index=1)
    rf = c2.number_input("무위험수익률(연, %)", 0.0, 20.0, 3.0, 0.1) / 100
    c3, c4 = st.columns(2)
    corr_window = c3.selectbox("상관계수 기간", [20, 60, 120], index=1)
    cov_window = c4.selectbox("Risk Parity 기간", [60, 120, 252], index=1)
    max_rp = st.slider("Risk Parity 종목당 최대 비중", .20, 1.0, .50, .05)

default = pd.DataFrame([
    {"ticker":"005930.KS","name":"삼성전자","market":"KR","quantity":0.0,"override_weight":0.30},
    {"ticker":"000660.KS","name":"SK하이닉스","market":"KR","quantity":0.0,"override_weight":0.20},
    {"ticker":"NVDA","name":"NVIDIA","market":"US","quantity":0.0,"override_weight":0.20},
    {"ticker":"BE","name":"Bloom Energy","market":"US","quantity":0.0,"override_weight":0.15},
    {"ticker":"BWXT","name":"BWX Technologies","market":"US","quantity":0.0,"override_weight":0.05},
    {"ticker":"CASH_KRW","name":"현금","market":"CASH","quantity":0.0,"override_weight":0.10},
])

uploaded = st.file_uploader("포트폴리오 CSV 불러오기", type=["csv"])
if uploaded is not None:
    try:
        tmp = pd.read_csv(uploaded)
        need = {"ticker","name","market","quantity","override_weight"}
        if not need.issubset(tmp.columns):
            st.error("CSV 컬럼: ticker, name, market, quantity, override_weight 가 필요합니다.")
        else:
            default = tmp
    except Exception as e:
        st.error(f"CSV 읽기 실패: {e}")

with st.expander("✏️ 보유종목 편집", expanded=True):
    edited = st.data_editor(
        default,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", help="한국: 005930.KS / 미국: NVDA"),
            "name": st.column_config.TextColumn("종목명"),
            "market": st.column_config.SelectboxColumn("시장", options=["KR","US","CASH"]),
            "quantity": st.column_config.NumberColumn("수량/현금액", min_value=0.0, format="%.4f"),
            "override_weight": st.column_config.NumberColumn(
                "직접 비중", min_value=0.0, max_value=1.0, step=.01,
                help="직접 비중이 하나라도 있으면 이 비중을 사용합니다. 실제 수량 기반 자동계산을 쓰려면 전부 비우세요."
            ),
        },
    )
    st.download_button(
        "현재 포트폴리오 CSV 저장",
        edited.to_csv(index=False).encode("utf-8-sig"),
        file_name="portfolio.csv",
        mime="text/csv",
        use_container_width=True,
    )

df = edited.copy()
df["ticker"] = df["ticker"].astype(str).str.strip()
df["market"] = df["market"].astype(str).str.upper().str.strip()
df = df[df["ticker"] != ""]

non_cash = df[df["market"] != "CASH"].copy()
if non_cash.empty:
    st.error("분석할 주식이 없습니다.")
    st.stop()

need_fx = (non_cash["market"] == "US").any()
fx_close = None
latest_fx = 1.0

if need_fx:
    try:
        fx_hist = fetch_fx(period)
        fx_close = fx_hist["Close"].dropna().rename("USDKRW")
        latest_fx = float(fx_close.iloc[-1])
    except Exception:
        st.warning("USD/KRW 환율을 불러오지 못했습니다. 미국주식 원화 환산이 부정확할 수 있습니다.")

returns = {}
prices = {}
errors = []

with st.spinner("최신 가격과 리스크 지표 계산 중..."):
    for _, row in non_cash.iterrows():
        t = row["ticker"]
        try:
            h = fetch_history(t, period)
            close = h["Close"].dropna().rename(t)
            if close.empty:
                raise ValueError("가격 데이터 없음")
            prices[t] = float(close.iloc[-1])

            if row["market"] == "US" and fx_close is not None:
                al = pd.concat([close, fx_close], axis=1).dropna()
                krw_price = al[t] * al["USDKRW"]
                returns[t] = krw_price.pct_change().rename(t)
            else:
                returns[t] = close.pct_change().rename(t)
        except Exception as e:
            errors.append(f"{t}: {e}")

if errors:
    st.warning("일부 가격 데이터 오류: " + " | ".join(errors))

if not returns:
    st.error("가격 데이터를 불러오지 못했습니다.")
    st.stop()

R = pd.concat(returns.values(), axis=1).dropna(how="all")
R.columns = list(returns.keys())

# Current weights
ow = pd.to_numeric(df["override_weight"], errors="coerce")
use_override = (ow.fillna(0) > 0).any()

if use_override:
    raw = ow.fillna(0).to_numpy(dtype=float)
    df["current_weight"] = normalize(raw)
    df["market_value_krw"] = np.nan
    weight_mode = "직접 비중"
else:
    vals = []
    for _, row in df.iterrows():
        qty = float(pd.to_numeric(row["quantity"], errors="coerce") or 0)
        if row["market"] == "CASH":
            val = qty
        else:
            px = prices.get(row["ticker"], np.nan)
            if pd.isna(px):
                val = np.nan
            elif row["market"] == "US":
                val = qty * px * latest_fx
            else:
                val = qty * px
        vals.append(val)
    df["market_value_krw"] = vals
    total = df["market_value_krw"].fillna(0).sum()
    if total <= 0:
        st.error("수량 기반 계산을 사용하려면 보유수량 또는 현금액을 입력하세요.")
        st.stop()
    df["current_weight"] = df["market_value_krw"].fillna(0) / total
    weight_mode = "실제 수량"

risky = df[(df["market"] != "CASH") & (df["ticker"].isin(R.columns))].copy()
tickers = risky["ticker"].tolist()
weights = risky["current_weight"].to_numpy(dtype=float)

pret = R[tickers].fillna(0).mul(pd.Series(weights, index=tickers), axis=1).sum(axis=1).dropna()

corr = R[tickers].tail(corr_window).corr()
cov = R[tickers].tail(cov_window).cov().values * TRADING_DAYS

risky_sum = weights.sum()
rp = risk_parity_weights(cov, max_rp) * risky_sum
_, rc = risk_contribution(normalize(weights), cov)

s1m = sharpe(pret.tail(21), rf)
s3m = sharpe(pret.tail(63), rf)
s6m = sharpe(pret.tail(126), rf)
s1y = sharpe(pret.tail(252), rf)
vol = ann_vol(pret)
mdd = max_drawdown(pret)
aret = ann_return(pret)
var95, cvar95 = var_cvar(pret)
div_score = diversification_score(weights, corr)
avgc20 = avg_corr(R[tickers].tail(20).corr())
avgc120 = avg_corr(R[tickers].tail(120).corr())

# Main metrics: 3 + 2 for iPad
m1, m2, m3 = st.columns(3)
m1.metric("Sharpe (1Y)", fmt_num(s1y))
m2.metric("연환산 변동성", fmt_pct(vol))
m3.metric("MDD", fmt_pct(mdd))

m4, m5 = st.columns(2)
m4.metric("95% CVaR", fmt_pct(cvar95))
m5.metric("Diversification", f"{div_score}/100")

msgs = []
if avgc20 >= .80:
    msgs.append(f"20D 평균 상관 {avgc20:.2f}: 높은 편")
if avgc20 - avgc120 >= .15:
    msgs.append(f"평균 상관 120D {avgc120:.2f} → 20D {avgc20:.2f}: 빠르게 상승")
if not msgs:
    msgs.append("단기 상관구조에서 뚜렷한 경고 없음")
st.info("오늘의 리스크 요약 · " + " · ".join(msgs))

tabs = st.tabs(["Dashboard", "Risk", "Correlation", "Stress", "Rebalance"])

with tabs[0]:
    st.subheader("현재 포트폴리오")
    view = df[["ticker","name","market","current_weight"]].copy()
    view["현재가격"] = view["ticker"].map(prices)
    if weight_mode == "실제 수량":
        view["평가금액(KRW)"] = df["market_value_krw"]
    st.dataframe(
        view.style.format({
            "current_weight":"{:.1%}",
            "현재가격":"{:,.2f}",
            "평가금액(KRW)":"{:,.0f}",
        }),
        use_container_width=True,
    )
    st.caption(f"비중 계산: {weight_mode}" + (f" · USD/KRW {latest_fx:,.2f}" if need_fx else ""))

    st.subheader("Sharpe Ratio")
    sdf = pd.DataFrame({"기간":["1M","3M","6M","1Y"], "Sharpe":[s1m,s3m,s6m,s1y]})
    st.plotly_chart(px.bar(sdf, x="기간", y="Sharpe", text_auto=".2f"), use_container_width=True)

    st.subheader("누적 수익률")
    wealth = (1 + pret.fillna(0)).cumprod() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wealth.index, y=wealth.values, mode="lines", name="Portfolio"))
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    a,b = st.columns(2)
    a.metric("연환산 수익률", fmt_pct(aret))
    b.metric("95% Daily VaR", fmt_pct(var95))
    c,d = st.columns(2)
    c.metric("20D 평균 상관", fmt_num(avgc20))
    d.metric("120D 평균 상관", fmt_num(avgc120))

with tabs[1]:
    st.subheader("종목별 Risk Contribution")
    rt = risky[["ticker","name","current_weight"]].copy()
    rt["risk_contribution"] = rc
    rt["risk_parity_weight"] = rp
    rt["gap"] = rt["current_weight"] - rt["risk_parity_weight"]

    st.dataframe(
        rt.style.format({
            "current_weight":"{:.1%}",
            "risk_contribution":"{:.1%}",
            "risk_parity_weight":"{:.1%}",
            "gap":"{:+.1%}",
        }),
        use_container_width=True,
    )

    fig = go.Figure()
    fig.add_bar(name="현재 비중", x=rt["name"], y=rt["current_weight"])
    fig.add_bar(name="Risk Contribution", x=rt["name"], y=rt["risk_contribution"])
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(barmode="group", height=390, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader(f"{corr_window}거래일 상관계수")
    names = risky.set_index("ticker")["name"].to_dict()
    cl = corr.rename(index=names, columns=names)
    fig = px.imshow(
        cl, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
        color_continuous_scale="RdBu_r"
    )
    fig.update_layout(height=500, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    vals = []
    for w in [20,60,120]:
        vals.append({"기간":f"{w}D", "평균상관":avg_corr(R[tickers].tail(w).corr())})
    st.plotly_chart(
        px.bar(pd.DataFrame(vals), x="기간", y="평균상관", text_auto=".2f"),
        use_container_width=True
    )

with tabs[3]:
    st.subheader("스트레스 테스트")
    st.caption("각 자산 가격이 즉시 몇 % 움직이는지 입력하면 포트폴리오 1차 충격을 계산합니다.")
    stress = df[["ticker","name","market","current_weight"]].copy()
    stress["shock_pct"] = stress["market"].map({"KR":-8.0,"US":-10.0,"CASH":0.0}).fillna(0.0)

    se = st.data_editor(
        stress,
        use_container_width=True,
        disabled=["ticker","name","market","current_weight"],
        column_config={
            "current_weight": st.column_config.NumberColumn("현재비중", format="%.4f"),
            "shock_pct": st.column_config.NumberColumn("충격(%)", step=1.0),
        }
    )
    impact = (se["current_weight"] * se["shock_pct"] / 100).sum()
    st.metric("예상 포트폴리오 충격", f"{impact*100:.2f}%")

with tabs[4]:
    st.subheader("현재 비중 vs Risk Parity")
    rb = risky[["ticker","name","current_weight"]].copy()
    rb["risk_parity_weight"] = rp
    rb["difference"] = rb["risk_parity_weight"] - rb["current_weight"]

    cash_w = df.loc[df["market"]=="CASH","current_weight"].sum()
    if cash_w > 0:
        rb = pd.concat([rb, pd.DataFrame([{
            "ticker":"CASH_KRW",
            "name":"현금",
            "current_weight":cash_w,
            "risk_parity_weight":cash_w,
            "difference":0.0
        }])], ignore_index=True)

    st.dataframe(
        rb.style.format({
            "current_weight":"{:.1%}",
            "risk_parity_weight":"{:.1%}",
            "difference":"{:+.1%}",
        }),
        use_container_width=True,
    )

    fig = go.Figure()
    fig.add_bar(name="현재", x=rb["name"], y=rb["current_weight"])
    fig.add_bar(name="Risk Parity", x=rb["name"], y=rb["risk_parity_weight"])
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(barmode="group", height=390, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    if weight_mode == "실제 수량":
        total_value = df["market_value_krw"].fillna(0).sum()
        rb["매수/매도금액(KRW)"] = rb["difference"] * total_value
        st.dataframe(
            rb[["name","매수/매도금액(KRW)"]].style.format({"매수/매도금액(KRW)":"{:+,.0f}"}),
            use_container_width=True,
        )

st.divider()
st.caption("리스크 관리 보조용 도구입니다. 무료 시세 데이터는 지연·누락될 수 있으며 과거 상관·공분산은 미래를 보장하지 않습니다.")
