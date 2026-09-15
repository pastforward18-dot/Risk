
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

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
h1 { font-size: 2rem !important; }
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
@media (max-width: 820px) {
    .block-container { padding-left: .5rem; padding-right: .5rem; }
    h1 { font-size: 1.5rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.25rem !important; }
}
</style>
""", unsafe_allow_html=True)

def normalize(w):
    w = np.asarray(w, dtype=float)
    s = np.nansum(w)
    return w / s if s > 0 else np.zeros(len(w), dtype=float)

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
    return total ** (1 / years) - 1 if total > 0 and years > 0 else np.nan

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


def pairwise_corr_matrix(return_df, tickers, n_obs):
    """
    각 종목쌍마다 '공통 거래일'을 먼저 맞춘 뒤 최근 n_obs개 관측치로
    상관계수를 계산합니다. 한국/미국 휴장일이 달라도 종목이 빠지지 않도록
    단순 tail(n_obs)보다 안정적으로 계산하는 방식입니다.
    """
    tickers = [t for t in tickers if t in return_df.columns]
    out = pd.DataFrame(np.nan, index=tickers, columns=tickers, dtype=float)

    for i, t1 in enumerate(tickers):
        s1 = return_df[t1].dropna()
        if len(s1) >= 2:
            out.loc[t1, t1] = 1.0

        for j in range(i + 1, len(tickers)):
            t2 = tickers[j]
            pair = pd.concat(
                [return_df[t1].rename(t1), return_df[t2].rename(t2)],
                axis=1
            ).dropna().tail(n_obs)

            # 너무 적은 관측치로 계산하면 값이 왜곡되므로 최소 표본 수 적용
            min_obs = min(10, max(5, n_obs // 3))
            if len(pair) >= min_obs:
                value = pair[t1].corr(pair[t2])
                out.loc[t1, t2] = value
                out.loc[t2, t1] = value

    return out

def correlation_pairs(corr_df, name_map):
    rows = []
    cols = list(corr_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            t1, t2 = cols[i], cols[j]
            value = corr_df.loc[t1, t2]
            if pd.isna(value):
                continue

            if value >= 0.75:
                level = "🔴 높음"
            elif value >= 0.50:
                level = "🟠 보통 이상"
            elif value <= -0.30:
                level = "🔵 역상관"
            else:
                level = "🟢 낮음"

            rows.append({
                "종목 1": name_map.get(t1, t1),
                "종목 2": name_map.get(t2, t2),
                "상관계수": float(value),
                "판정": level,
            })

    if not rows:
        return pd.DataFrame(columns=["종목 1", "종목 2", "상관계수", "판정"])

    return pd.DataFrame(rows).sort_values(
        ["상관계수"], ascending=False
    ).reset_index(drop=True)

def diversification_score(weights, corr):
    w = normalize(weights)
    if len(w) == 0 or np.sum(w) == 0:
        return 0

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


def normalize_date_index(series):
    """
    Yahoo Finance는 시장별로 서로 다른 timezone을 붙일 수 있습니다.
    상관계수/환율 결합에서는 시간대가 아니라 거래 '날짜'가 중요하므로
    모든 시계열을 timezone 없는 날짜(00:00) 인덱스로 통일합니다.
    """
    s = series.copy().dropna()
    idx = pd.to_datetime(s.index)

    # tz-aware index라면 현지 날짜를 보존한 채 timezone만 제거
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass

    idx = pd.DatetimeIndex(idx).normalize()
    s.index = idx

    # 혹시 동일 날짜가 중복되면 마지막 관측값 사용
    if s.index.has_duplicates:
        s = s.groupby(level=0).last()

    return s.sort_index()

st.title("📊 Portfolio Risk Terminal")
st.caption("iPad · Safari 안정화 v3 / 미국주식·환율 날짜 정렬 수정 / 원화·달러 현금 / 전체종목 Correlation")

with st.expander("⚙️ 분석 설정", expanded=False):
    c1, c2 = st.columns(2)
    period = c1.selectbox("가격 히스토리", ["1y", "2y", "5y"], index=1)
    rf = c2.number_input("무위험수익률(연, %)", 0.0, 20.0, 3.0, 0.1) / 100

    c3, c4 = st.columns(2)
    corr_window = c3.selectbox("상관계수 기간", [20, 60, 120], index=1)
    cov_window = c4.selectbox("Risk Parity 기간", [60, 120, 252], index=1)

    max_rp = st.slider("Risk Parity 종목당 최대 비중", .20, 1.0, .50, .05)

mode = st.radio(
    "비중 계산 방식",
    ["실제 수량으로 자동 계산", "직접 비중 입력"],
    horizontal=True,
)

default = pd.DataFrame([
    {"ticker":"005930.KS","name":"삼성전자","market":"KR","quantity":150.0,"manual_weight":0.0},
    {"ticker":"000660.KS","name":"SK하이닉스","market":"KR","quantity":40.0,"manual_weight":0.0},
    {"ticker":"OII","name":"OII","market":"US","quantity":84.0,"manual_weight":0.0},
    {"ticker":"SOXL","name":"SOXL","market":"US","quantity":58.0,"manual_weight":0.0},
    {"ticker":"CASH_KRW","name":"원화 현금","market":"CASH","quantity":0.0,"manual_weight":0.0},
    {"ticker":"CASH_USD","name":"달러 현금","market":"USD_CASH","quantity":0.0,"manual_weight":0.0},
])

if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = default.copy()

uploaded = st.file_uploader("포트폴리오 CSV 불러오기", type=["csv"])

if uploaded is not None:
    try:
        tmp = pd.read_csv(uploaded)
        tmp = tmp.rename(columns={"override_weight":"manual_weight"})

        for col in ["ticker","name","market","quantity","manual_weight"]:
            if col not in tmp.columns:
                tmp[col] = "" if col in ["ticker","name","market"] else 0.0

        st.session_state.portfolio_df = tmp[
            ["ticker","name","market","quantity","manual_weight"]
        ].copy()
    except Exception as e:
        st.error(f"CSV 읽기 실패: {e}")

with st.expander("✏️ 보유종목 편집", expanded=True):
    edited = st.data_editor(
        st.session_state.portfolio_df,
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "name": st.column_config.TextColumn("종목명"),
            "market": st.column_config.SelectboxColumn(
                "시장",
                options=["KR","US","CASH","USD_CASH"],
                help="CASH=원화 현금, USD_CASH=달러 현금"
            ),
            "quantity": st.column_config.NumberColumn(
                "수량/현금액",
                min_value=0.0,
                step=1.0,
                format="%.4f",
                help="KR/US는 보유 주식 수, CASH는 원화 금액, USD_CASH는 달러 금액을 입력"
            ),
            "manual_weight": st.column_config.NumberColumn(
                "직접 비중",
                min_value=0.0,
                max_value=1.0,
                step=.01,
                format="%.2f",
            ),
        },
    )

    st.session_state.portfolio_df = edited.copy()

    st.download_button(
        "현재 포트폴리오 CSV 저장",
        edited.to_csv(index=False).encode("utf-8-sig"),
        file_name="portfolio.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Clean safely
df = edited.copy()

for col in ["ticker","name","market"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

df["ticker"] = df["ticker"].str.upper()
df["market"] = df["market"].str.upper()

df["quantity"] = pd.to_numeric(
    df["quantity"], errors="coerce"
).fillna(0.0)

df["manual_weight"] = pd.to_numeric(
    df["manual_weight"], errors="coerce"
).fillna(0.0)

df = df[
    (df["ticker"] != "") &
    (df["market"].isin(["KR","US","CASH","USD_CASH"]))
].copy()

if df.empty:
    st.info("종목을 한 개 이상 입력해 주세요.")
    st.stop()

stocks = df[df["market"].isin(["KR","US"])].copy()

if stocks.empty:
    st.info("분석할 주식을 한 개 이상 입력해 주세요.")
    st.stop()

# FX: 미국주식 또는 달러 현금이 있으면 USD/KRW 필요
need_fx = (
    (stocks["market"] == "US").any()
    or (df["market"] == "USD_CASH").any()
)
latest_fx = 1.0
fx_close = None

if need_fx:
    try:
        fx_hist = fetch_fx(period)
        fx_close = normalize_date_index(
            fx_hist["Close"].rename("USDKRW")
        )
        latest_fx = float(fx_close.iloc[-1])
    except Exception as e:
        st.warning(f"USD/KRW 환율 로딩 실패: {e}")

returns = {}
prices = {}
errors = []
fetch_status = {}

with st.spinner("최신 가격과 리스크 지표 계산 중..."):
    for _, row in stocks.iterrows():
        t = row["ticker"]

        try:
            h = fetch_history(t, period)

            if h.empty or "Close" not in h.columns:
                raise ValueError("가격 데이터 없음")

            close = normalize_date_index(
                h["Close"].rename(t)
            )

            if close.empty:
                raise ValueError("가격 데이터 없음")

            prices[t] = float(close.iloc[-1])

            if row["market"] == "US":
                if fx_close is None or fx_close.empty:
                    raise ValueError("USD/KRW 환율 데이터 없음")

                # 미국주식과 환율을 timezone timestamp가 아닌 '날짜'로 결합
                al = pd.concat(
                    [close, fx_close],
                    axis=1,
                    join="inner"
                ).dropna()

                if len(al) < 10:
                    raise ValueError(
                        f"미국주식-환율 공통 날짜 부족({len(al)}개)"
                    )

                krw_price = (al[t] * al["USDKRW"]).rename(t)
                returns[t] = krw_price.pct_change(fill_method=None).rename(t)
            else:
                returns[t] = close.pct_change(fill_method=None).rename(t)

            valid_ret = returns[t].dropna()
            if len(valid_ret) < 10:
                raise ValueError(
                    f"유효 수익률 관측치 부족({len(valid_ret)}개)"
                )
            fetch_status[t] = {
                "status": "✅ 정상",
                "observations": int(len(valid_ret)),
                "start": valid_ret.index.min() if len(valid_ret) else None,
                "end": valid_ret.index.max() if len(valid_ret) else None,
                "reason": "",
            }

        except Exception as e:
            errors.append(f"{t}: {e}")
            fetch_status[t] = {
                "status": "❌ 제외",
                "observations": 0,
                "start": None,
                "end": None,
                "reason": str(e),
            }

if errors:
    st.warning("일부 가격 데이터 오류: " + " | ".join(errors))

if not returns:
    st.error("가격 데이터를 불러올 수 없습니다. 티커를 확인해 주세요.")
    st.stop()

R = pd.concat(
    returns.values(),
    axis=1
).dropna(how="all")

R.columns = list(returns.keys())

# Weights
if mode == "실제 수량으로 자동 계산":
    values = []

    for _, row in df.iterrows():
        qty = float(row["quantity"])

        if row["market"] == "CASH":
            # 원화 현금: quantity에 원화 금액을 그대로 입력
            val = qty
        elif row["market"] == "USD_CASH":
            # 달러 현금: quantity에 USD 금액을 입력하면 환율로 자동 환산
            val = qty * latest_fx
        else:
            px = prices.get(row["ticker"], np.nan)

            if pd.isna(px):
                val = 0.0
            elif row["market"] == "US":
                val = qty * px * latest_fx
            else:
                val = qty * px

        values.append(val)

    df["market_value_krw"] = values
    total = df["market_value_krw"].sum()

    if total <= 0:
        st.error("수량 또는 현금액을 입력해 주세요.")
        st.stop()

    df["current_weight"] = df["market_value_krw"] / total
    weight_mode = "실제 수량 자동 계산"

else:
    total_manual = df["manual_weight"].sum()

    if total_manual <= 0:
        st.error("직접 비중을 입력해 주세요. 30% = 0.30")
        st.stop()

    df["current_weight"] = df["manual_weight"] / total_manual
    df["market_value_krw"] = np.nan
    weight_mode = "직접 비중 입력"

risky = df[
    (df["market"].isin(["KR","US"])) &
    (df["ticker"].isin(R.columns)) &
    (df["current_weight"] > 0)
].copy()

if risky.empty:
    st.error("유효한 보유종목이 없습니다.")
    st.stop()

tickers = risky["ticker"].tolist()
weights = risky["current_weight"].to_numpy(dtype=float)

pret = (
    R[tickers]
    .fillna(0)
    .mul(pd.Series(weights, index=tickers), axis=1)
    .sum(axis=1)
)

# 달러 현금은 원화 기준에서 환율 변동 위험이 있으므로
# 전체 포트폴리오 수익률/변동성 계산에는 USD/KRW 수익률을 반영합니다.
usd_cash_weight = df.loc[
    df["market"] == "USD_CASH",
    "current_weight"
].sum()

if usd_cash_weight > 0 and fx_close is not None:
    fx_ret = fx_close.pct_change().rename("USD_CASH_FX")
    pret = pd.concat([pret.rename("portfolio"), fx_ret], axis=1).dropna()
    pret = (
        pret["portfolio"]
        + usd_cash_weight * pret["USD_CASH_FX"]
    )

pret = pret.dropna()

corr = pairwise_corr_matrix(R, tickers, corr_window)

# Risk Parity 공분산은 모든 종목의 공통 거래일만 사용해 행렬 안정성을 높입니다.
cov_sample = R[tickers].dropna(how="any").tail(cov_window)
if len(cov_sample) < 10:
    cov_sample = R[tickers].fillna(0).tail(cov_window)
cov = cov_sample.cov().values * TRADING_DAYS

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
avgc20 = avg_corr(pairwise_corr_matrix(R, tickers, 20))
avgc120 = avg_corr(pairwise_corr_matrix(R, tickers, 120))

# Header
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
    msgs.append(
        f"평균 상관 120D {avgc120:.2f} → 20D {avgc20:.2f}: 상승"
    )

if not msgs:
    msgs.append("단기 상관구조에서 뚜렷한 경고 없음")

st.info("오늘의 리스크 요약 · " + " · ".join(msgs))

tabs = st.tabs([
    "Dashboard",
    "Risk",
    "Correlation",
    "Stress",
    "Rebalance"
])

with tabs[0]:
    st.subheader("현재 포트폴리오")

    view = df[[
        "ticker",
        "name",
        "market",
        "quantity",
        "current_weight"
    ]].copy()

    def display_price(row):
        if row["market"] == "USD_CASH":
            return latest_fx
        if row["market"] == "CASH":
            return 1.0
        return prices.get(row["ticker"], np.nan)

    view["현재가격/환율"] = view.apply(display_price, axis=1)

    if weight_mode == "실제 수량 자동 계산":
        view["평가금액(KRW)"] = df["market_value_krw"]

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker":"Ticker",
            "name":"종목명",
            "market":"시장",
            "quantity":st.column_config.NumberColumn(
                "수량/현금액", format="%.4f"
            ),
            "current_weight":st.column_config.NumberColumn(
                "현재비중", format="%.2f"
            ),
            "현재가격/환율":st.column_config.NumberColumn(
                "현재가격/환율", format="%.2f"
            ),
            "평가금액(KRW)":st.column_config.NumberColumn(
                "평가금액(KRW)", format="%.0f"
            ),
        }
    )

    st.caption(
        f"비중 계산: {weight_mode}"
        + (f" · USD/KRW {latest_fx:,.2f}" if need_fx else "")
        + " · USD_CASH는 달러 금액을 입력하면 원화 평가액/비중이 환율에 따라 자동 갱신"
    )

    st.subheader("Sharpe Ratio")
    sharpe_df = pd.DataFrame({
        "Sharpe":[s1m,s3m,s6m,s1y]
    }, index=["1M","3M","6M","1Y"])

    st.bar_chart(sharpe_df)

    st.subheader("누적 수익률")
    wealth = (1 + pret.fillna(0)).cumprod() - 1
    st.line_chart(
        pd.DataFrame(
            {"Portfolio":wealth}
        )
    )

    a,b = st.columns(2)
    a.metric("연환산 수익률", fmt_pct(aret))
    b.metric("95% Daily VaR", fmt_pct(var95))

    c,d = st.columns(2)
    c.metric("20D 평균 상관", fmt_num(avgc20))
    d.metric("120D 평균 상관", fmt_num(avgc120))

with tabs[1]:
    st.subheader("종목별 Risk Contribution")

    rt = risky[[
        "ticker",
        "name",
        "current_weight"
    ]].copy()

    rt["risk_contribution"] = rc
    rt["risk_parity_weight"] = rp
    rt["gap"] = (
        rt["current_weight"] -
        rt["risk_parity_weight"]
    )

    st.dataframe(
        rt,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker":"Ticker",
            "name":"종목명",
            "current_weight":st.column_config.NumberColumn(
                "현재비중", format="%.3f"
            ),
            "risk_contribution":st.column_config.NumberColumn(
                "위험기여도", format="%.3f"
            ),
            "risk_parity_weight":st.column_config.NumberColumn(
                "Risk Parity", format="%.3f"
            ),
            "gap":st.column_config.NumberColumn(
                "비중-적정비중", format="%.3f"
            ),
        }
    )

    risk_chart = rt.set_index("name")[
        ["current_weight","risk_contribution"]
    ]

    risk_chart.columns = [
        "현재 비중",
        "Risk Contribution"
    ]

    st.bar_chart(risk_chart)

with tabs[2]:
    st.subheader("Correlation · 종목 간 상관계수")
    st.caption(
        "한국/미국/환율 데이터의 timezone을 제거하고 거래 날짜 기준으로 정렬한 뒤, "
        "각 종목쌍의 공통 관측일을 맞춰 상관계수를 계산합니다."
    )

    corr_view_window = st.radio(
        "조회 기간",
        [20, 60, 120],
        index=1,
        horizontal=True,
        format_func=lambda x: f"{x}D",
        key="corr_view_window"
    )

    corr_tickers = [t for t in risky["ticker"].tolist() if t in R.columns]
    name_map = risky.set_index("ticker")["name"].to_dict()
    corr_view = pairwise_corr_matrix(R, corr_tickers, corr_view_window)

    # 어떤 종목이 분석에 들어왔고 왜 빠졌는지 바로 확인할 수 있게 표시
    total_stock_count = len(stocks)
    included_count = len(corr_tickers)
    excluded_count = max(total_stock_count - included_count, 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("보유 주식 종목", f"{total_stock_count}개")
    c2.metric("Correlation 포함", f"{included_count}개")
    c3.metric("제외", f"{excluded_count}개")

    if included_count >= 2:
        st.subheader(f"{corr_view_window}D 상관계수 매트릭스")

        corr_named = corr_view.rename(
            index=name_map,
            columns=name_map
        )

        st.dataframe(
            corr_named,
            use_container_width=True,
            column_config={
                col: st.column_config.NumberColumn(col, format="%.2f")
                for col in corr_named.columns
            }
        )

        st.caption(
            "+1에 가까울수록 같이 움직이고, 0에 가까울수록 관계가 약하며, "
            "-1에 가까울수록 반대로 움직입니다."
        )

        pairs_df = correlation_pairs(corr_view, name_map)

        st.subheader("상관관계 높은 종목쌍")
        if pairs_df.empty:
            st.info("계산 가능한 종목쌍이 아직 없습니다.")
        else:
            st.dataframe(
                pairs_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "종목 1": "종목 1",
                    "종목 2": "종목 2",
                    "상관계수": st.column_config.NumberColumn(
                        "상관계수", format="%.2f"
                    ),
                    "판정": "판정",
                }
            )

            highest = pairs_df.iloc[0]
            if highest["상관계수"] >= 0.75:
                st.warning(
                    f"가장 높은 상관: {highest['종목 1']} ↔ {highest['종목 2']} "
                    f"= {highest['상관계수']:.2f}. 두 종목의 위험이 동시에 커질 가능성을 주의하세요."
                )

        st.subheader("기간별 평균 상관계수")
        avg_corr_df = pd.DataFrame({
            "평균상관":[
                avg_corr(pairwise_corr_matrix(R, corr_tickers, 20)),
                avg_corr(pairwise_corr_matrix(R, corr_tickers, 60)),
                avg_corr(pairwise_corr_matrix(R, corr_tickers, 120))
            ]
        }, index=["20D","60D","120D"])
        st.bar_chart(avg_corr_df)
    else:
        st.warning(
            "Correlation 계산에는 가격 데이터가 정상적으로 들어온 주식이 최소 2개 필요합니다."
        )

    with st.expander("🔎 종목별 데이터 상태 확인", expanded=False):
        status_rows = []
        risky_ticker_set = set(corr_tickers)

        for _, row in stocks.iterrows():
            t = row["ticker"]
            info = fetch_status.get(t, {})

            observations = int(info.get("observations", 0) or 0)

            if t not in returns or observations < 10:
                include_status = "❌ 유효 데이터 부족"
            elif t not in risky_ticker_set:
                include_status = "⚪ 비중 0%라 제외"
            else:
                include_status = "✅ Correlation 포함"

            start_value = info.get("start")
            end_value = info.get("end")

            status_rows.append({
                "종목명": row["name"],
                "Ticker": t,
                "시장": row["market"],
                "상태": include_status,
                "수익률 관측치": observations,
                "데이터 시작": (
                    start_value.strftime("%Y-%m-%d")
                    if start_value is not None and hasattr(start_value, "strftime")
                    else "-"
                ),
                "데이터 종료": (
                    end_value.strftime("%Y-%m-%d")
                    if end_value is not None and hasattr(end_value, "strftime")
                    else "-"
                ),
                "오류/제외 이유": info.get("reason", ""),
            })

        status_df = pd.DataFrame(status_rows)
        st.dataframe(status_df, use_container_width=True, hide_index=True)

        st.caption(
            "여기서 OII·SOXL 같은 종목이 'Correlation 포함'으로 표시되면 "
            "위 매트릭스에도 반드시 나타나야 합니다. 오류가 있으면 마지막 열에서 원인을 확인할 수 있습니다."
        )

with tabs[3]:
    st.subheader("스트레스 테스트")

    st.caption(
        "각 자산이 즉시 몇 % 움직인다고 가정할지 입력하세요. "
        "USD_CASH의 충격(%)은 USD/KRW 환율 변화율로 생각하면 됩니다."
    )

    stress = df[[
        "ticker",
        "name",
        "market",
        "current_weight"
    ]].copy()

    stress["shock_pct"] = stress["market"].map({
        "KR":-8.0,
        "US":-10.0,
        "CASH":0.0,
        "USD_CASH":0.0
    }).fillna(0.0)

    stress_edit = st.data_editor(
        stress,
        use_container_width=True,
        hide_index=True,
        disabled=[
            "ticker",
            "name",
            "market",
            "current_weight"
        ],
        column_config={
            "ticker":"Ticker",
            "name":"종목명",
            "market":"시장",
            "current_weight":st.column_config.NumberColumn(
                "현재비중", format="%.3f"
            ),
            "shock_pct":st.column_config.NumberColumn(
                "충격(%)", step=1.0
            ),
        }
    )

    stress_edit["impact"] = (
        stress_edit["current_weight"] *
        stress_edit["shock_pct"] / 100
    )

    impact = stress_edit["impact"].sum()

    st.metric(
        "예상 포트폴리오 충격",
        f"{impact*100:.2f}%"
    )

    impact_chart = (
        stress_edit[["name","impact"]]
        .set_index("name")
    )

    st.bar_chart(impact_chart)

with tabs[4]:
    st.subheader("현재 비중 vs Risk Parity")

    rb = risky[[
        "ticker",
        "name",
        "current_weight"
    ]].copy()

    rb["risk_parity_weight"] = rp
    rb["difference"] = (
        rb["risk_parity_weight"] -
        rb["current_weight"]
    )

    cash_rows = []

    krw_cash_w = df.loc[
        df["market"] == "CASH",
        "current_weight"
    ].sum()

    usd_cash_w = df.loc[
        df["market"] == "USD_CASH",
        "current_weight"
    ].sum()

    if krw_cash_w > 0:
        cash_rows.append({
            "ticker":"CASH_KRW",
            "name":"원화 현금",
            "current_weight":krw_cash_w,
            "risk_parity_weight":krw_cash_w,
            "difference":0.0,
        })

    if usd_cash_w > 0:
        cash_rows.append({
            "ticker":"CASH_USD",
            "name":"달러 현금",
            "current_weight":usd_cash_w,
            "risk_parity_weight":usd_cash_w,
            "difference":0.0,
        })

    if cash_rows:
        rb = pd.concat(
            [rb, pd.DataFrame(cash_rows)],
            ignore_index=True
        )

    st.dataframe(
        rb,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker":"Ticker",
            "name":"종목명",
            "current_weight":st.column_config.NumberColumn(
                "현재비중", format="%.3f"
            ),
            "risk_parity_weight":st.column_config.NumberColumn(
                "Risk Parity", format="%.3f"
            ),
            "difference":st.column_config.NumberColumn(
                "조정 필요", format="%.3f"
            ),
        }
    )

    rb_chart = rb.set_index("name")[
        ["current_weight","risk_parity_weight"]
    ]

    rb_chart.columns = [
        "현재 비중",
        "Risk Parity"
    ]

    st.bar_chart(rb_chart)

    if weight_mode == "실제 수량 자동 계산":
        total_value = df["market_value_krw"].sum()

        rb["매수/매도금액(KRW)"] = (
            rb["difference"] * total_value
        )

        st.subheader("리밸런싱 필요 금액")

        st.dataframe(
            rb[[
                "name",
                "매수/매도금액(KRW)"
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "name":"종목명",
                "매수/매도금액(KRW)":st.column_config.NumberColumn(
                    "매수(+)/매도(-) 금액",
                    format="%.0f"
                ),
            }
        )

st.divider()

st.caption(
    "리스크 관리 보조용입니다. 무료 시세 데이터에는 지연·누락이 있을 수 있으며 "
    "과거 상관계수와 공분산은 미래를 보장하지 않습니다."
)
