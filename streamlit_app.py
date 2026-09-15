
import math
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

TRADING_DAYS = 252

st.set_page_config(
    page_title="Portfolio Risk Terminal V3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {
    max-width: 1450px;
    padding-top: .65rem;
    padding-bottom: 2.5rem;
}
h1 { font-size: 1.95rem !important; }
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
@media (max-width: 900px) {
    .block-container { padding-left: .55rem; padding-right: .55rem; }
    h1 { font-size: 1.5rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.25rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Core math
# ============================================================
def normalize(w):
    w = np.asarray(w, dtype=float)
    s = np.nansum(w)
    if s <= 0:
        return np.zeros(len(w), dtype=float)
    return w / s


def ann_vol(r):
    r = pd.Series(r).dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 2 else np.nan


def ann_return(r):
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return np.nan
    total = float((1 + r).prod())
    years = len(r) / TRADING_DAYS
    if total <= 0 or years <= 0:
        return np.nan
    return total ** (1 / years) - 1


def sharpe(r, rf=0.0):
    r = pd.Series(r).dropna()
    if len(r) < 5 or r.std(ddof=1) == 0:
        return np.nan
    rf_daily = (1 + rf) ** (1 / TRADING_DAYS) - 1
    return float((r.mean() - rf_daily) / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(r):
    r = pd.Series(r).dropna()
    if r.empty:
        return np.nan
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1).min())


def hist_var_cvar(r, alpha=.95):
    r = pd.Series(r).dropna()
    if len(r) < 20:
        return np.nan, np.nan
    q = float(np.quantile(r, 1 - alpha))
    tail = r[r <= q]
    return -q, float(-tail.mean()) if len(tail) else np.nan


def ewma_cvar(r, alpha=.95, lam=.94):
    """Exponentially weighted tail loss; recent observations receive larger weights."""
    r = pd.Series(r).dropna()
    if len(r) < 20:
        return np.nan

    vals = r.to_numpy(dtype=float)
    n = len(vals)

    ages = (n - 1) - np.arange(n)
    weights = (1 - lam) * (lam ** ages)
    weights = weights / weights.sum()

    order = np.argsort(vals)
    sorted_vals = vals[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)

    cutoff = 1 - alpha
    idx = int(np.searchsorted(cum_w, cutoff, side="left"))
    idx = min(idx, n - 1)
    q = sorted_vals[idx]

    mask = vals <= q
    tail_w = weights[mask]
    tail_vals = vals[mask]
    if tail_w.sum() <= 0:
        return np.nan

    return float(-(tail_vals * tail_w).sum() / tail_w.sum())


def risk_contribution(weights, cov):
    w = np.asarray(weights, dtype=float)
    sigma2 = float(w.T @ cov @ w)
    sigma = math.sqrt(max(sigma2, 0))
    if sigma <= 0:
        return np.zeros_like(w), np.zeros_like(w)

    mrc = cov @ w / sigma
    crc = w * mrc
    pct = crc / sigma
    return crc, pct


def risk_parity_weights(cov, max_weight=1.0):
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Ensure feasibility.
    effective_max = max(float(max_weight), 1 / n + 1e-6)
    x0 = np.repeat(1 / n, n)
    target = np.repeat(1 / n, n)

    def objective(w):
        _, rc = risk_contribution(w, cov)
        return float(np.sum((rc - target) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, effective_max)] * n

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500}
    )

    return normalize(res.x) if res.success else x0


def beta_to(x, y):
    z = pd.concat([pd.Series(x).rename("x"), pd.Series(y).rename("y")], axis=1).dropna()
    if len(z) < 15 or z["y"].var(ddof=1) == 0:
        return np.nan
    return float(z["x"].cov(z["y"]) / z["y"].var(ddof=1))


def pair_corr(a, b, window):
    z = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna().tail(window)
    if len(z) < 10:
        return np.nan, len(z)
    return float(z["a"].corr(z["b"])), len(z)


def avg_pairwise_corr(series_dict, tickers, window):
    vals = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c, n = pair_corr(series_dict[tickers[i]], series_dict[tickers[j]], window)
            if pd.notna(c):
                vals.append(c)
    return float(np.mean(vals)) if vals else np.nan


def diversification_ratio(weights, return_df):
    w = np.asarray(weights, dtype=float)
    if len(w) == 0 or return_df.empty:
        return np.nan

    x = return_df.dropna(how="all").fillna(0)
    if len(x) < 20:
        return np.nan

    cov = x.cov().values * TRADING_DAYS
    asset_vols = np.sqrt(np.maximum(np.diag(cov), 0))
    port_vol = math.sqrt(max(float(w.T @ cov @ w), 0))
    if port_vol <= 0:
        return np.nan
    return float(np.dot(w, asset_vols) / port_vol)


def effective_number_of_bets(rc):
    rc = np.asarray(rc, dtype=float)
    x = np.abs(rc)
    if x.sum() <= 0:
        return np.nan
    p = x / x.sum()
    return float(1 / np.sum(p ** 2))


def fmt_pct(x, digits=1):
    return "—" if pd.isna(x) else f"{x*100:.{digits}f}%"


def fmt_num(x, digits=2):
    return "—" if pd.isna(x) else f"{x:.{digits}f}"


# ============================================================
# Price data
# ============================================================
def normalize_daily_series(s):
    s = pd.Series(s).dropna().copy()
    if s.empty:
        return s

    idx = pd.DatetimeIndex(s.index)
    # Keep the exchange-local calendar date, but remove timezone info.
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()

    s.index = idx
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_close(ticker, period):
    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype=float)
    return normalize_daily_series(hist["Close"].rename(ticker))


def us_krw_value_series(usd_close, fx_close):
    if usd_close.empty or fx_close.empty:
        return pd.Series(dtype=float)

    # Reindex FX to US trading days; forward/back-fill nearby holidays.
    fx_aligned = fx_close.reindex(usd_close.index).ffill().bfill()
    out = usd_close * fx_aligned
    return out.dropna()


def current_fx_value(fx_close):
    return float(fx_close.dropna().iloc[-1]) if len(fx_close.dropna()) else np.nan


def build_asset_data(asset_rows, period):
    fx_close = fetch_close("KRW=X", period)
    latest_fx = current_fx_value(fx_close)

    local_closes = {}
    krw_values = {}
    krw_returns = {}
    errors = {}

    for _, row in asset_rows.iterrows():
        ticker = row["ticker"]
        market = row["market"]

        if market not in ["KR", "US"]:
            continue

        try:
            close = fetch_close(ticker, period)
            if close.empty:
                raise ValueError("가격 데이터 없음")

            local_closes[ticker] = close

            if market == "US":
                v = us_krw_value_series(close, fx_close)
                if v.empty:
                    raise ValueError("USD/KRW 결합 데이터 없음")
                krw_values[ticker] = v
            else:
                krw_values[ticker] = close

            krw_returns[ticker] = krw_values[ticker].pct_change().dropna()

        except Exception as e:
            errors[ticker] = str(e)

    return local_closes, krw_values, krw_returns, fx_close, latest_fx, errors


def build_current_asset_returns(df, krw_returns, fx_close):
    series = {}
    for _, row in df.iterrows():
        t = row["ticker"]
        m = row["market"]

        if m in ["KR", "US"] and t in krw_returns:
            series[t] = krw_returns[t]
        elif m == "USD_CASH":
            series[t] = fx_close.pct_change().dropna().rename(t)
        elif m == "CASH":
            # KRW cash is zero-volatility; excluded from covariance/ERC.
            continue

    return series


def weighted_portfolio_return(series_dict, weight_map):
    active = [k for k in weight_map if k in series_dict and weight_map[k] != 0]
    if not active:
        return pd.Series(dtype=float)

    calendar = None
    for k in active:
        idx = series_dict[k].index
        calendar = idx if calendar is None else calendar.union(idx)

    calendar = calendar.sort_values()
    total = pd.Series(0.0, index=calendar)

    for k in active:
        r = series_dict[k].reindex(calendar).fillna(0.0)
        total = total + float(weight_map[k]) * r

    return total.sort_index()


# ============================================================
# Actual transaction NAV
# ============================================================
TRADE_ACTIONS = [
    "BUY",
    "SELL",
    "DEPOSIT_KRW",
    "WITHDRAW_KRW",
    "DEPOSIT_USD",
    "WITHDRAW_USD",
    "OPENING_POSITION",
]


def transaction_nav(trades, market_map, local_closes, fx_close):
    """
    Reconstruct mark-to-market NAV.
    External flows: deposits/withdrawals and OPENING_POSITION initial value.
    Daily return removes external flows:
      (NAV_t - NAV_{t-1} - external_flow_t) / NAV_{t-1}
    """
    if trades is None or trades.empty:
        return pd.DataFrame(), []

    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce").dt.normalize()
    t["action"] = t["action"].fillna("").astype(str).str.upper().str.strip()
    t["ticker"] = t["ticker"].fillna("").astype(str).str.upper().str.strip()
    t["market"] = t["market"].fillna("").astype(str).str.upper().str.strip()
    t["quantity"] = pd.to_numeric(t["quantity"], errors="coerce").fillna(0.0)
    t["price"] = pd.to_numeric(t["price"], errors="coerce").fillna(0.0)
    t["fee"] = pd.to_numeric(t["fee"], errors="coerce").fillna(0.0)

    t = t[
        t["date"].notna()
        & t["action"].isin(TRADE_ACTIONS)
    ].copy()

    if t.empty:
        return pd.DataFrame(), []

    start = t["date"].min()

    ends = []
    if not fx_close.empty:
        ends.append(fx_close.index.max())
    for s in local_closes.values():
        if not s.empty:
            ends.append(s.index.max())

    if not ends:
        return pd.DataFrame(), ["가격 데이터가 없어 실제 NAV를 재구성하지 못했습니다."]

    end = max(ends)
    calendar = pd.date_range(start=start, end=end, freq="B")

    # Prepare forward-filled price tables.
    px = {}
    for ticker, close in local_closes.items():
        px[ticker] = close.reindex(calendar).ffill()

    fx = fx_close.reindex(calendar).ffill().bfill()
    warnings = []

    # Move weekend events to next business day.
    def effective_day(d):
        pos = calendar.searchsorted(d)
        if pos >= len(calendar):
            return calendar[-1]
        return calendar[pos]

    t["effective_date"] = t["date"].map(effective_day)

    events = {}
    for idx, row in t.iterrows():
        events.setdefault(row["effective_date"], []).append(row)

    positions = {}
    krw_cash = 0.0
    usd_cash = 0.0

    records = []

    for d in calendar:
        external_flow_krw = 0.0

        for row in events.get(d, []):
            action = row["action"]
            ticker = row["ticker"]
            market = row["market"] or market_map.get(ticker, "")
            qty = float(row["quantity"])
            fee = float(row["fee"])
            fx_d = float(fx.loc[d]) if d in fx.index and pd.notna(fx.loc[d]) else np.nan

            if action == "DEPOSIT_KRW":
                krw_cash += qty
                external_flow_krw += qty
                continue

            if action == "WITHDRAW_KRW":
                krw_cash -= qty
                external_flow_krw -= qty
                continue

            if action == "DEPOSIT_USD":
                usd_cash += qty
                if pd.notna(fx_d):
                    external_flow_krw += qty * fx_d
                continue

            if action == "WITHDRAW_USD":
                usd_cash -= qty
                if pd.notna(fx_d):
                    external_flow_krw -= qty * fx_d
                continue

            if ticker == "":
                warnings.append(f"{d.date()}: {action}에 ticker가 없습니다.")
                continue

            price = float(row["price"])
            if price <= 0:
                if ticker not in px or pd.isna(px[ticker].loc[d]):
                    warnings.append(f"{d.date()} {ticker}: 체결가/과거가격 없음")
                    continue
                price = float(px[ticker].loc[d])

            positions.setdefault(ticker, 0.0)

            if action == "OPENING_POSITION":
                positions[ticker] += qty
                if market == "US" and pd.notna(fx_d):
                    external_flow_krw += qty * price * fx_d
                elif market == "KR":
                    external_flow_krw += qty * price
                continue

            if action == "BUY":
                positions[ticker] += qty
                if market == "US":
                    usd_cash -= qty * price + fee
                else:
                    krw_cash -= qty * price + fee

            elif action == "SELL":
                positions[ticker] -= qty
                if market == "US":
                    usd_cash += qty * price - fee
                else:
                    krw_cash += qty * price - fee

        # Mark to market.
        stock_value_krw = 0.0
        fx_d = float(fx.loc[d]) if d in fx.index and pd.notna(fx.loc[d]) else np.nan

        for ticker, qty in positions.items():
            if qty == 0:
                continue
            market = market_map.get(ticker, "")
            if ticker not in px or pd.isna(px[ticker].loc[d]):
                continue
            price = float(px[ticker].loc[d])

            if market == "US":
                if pd.notna(fx_d):
                    stock_value_krw += qty * price * fx_d
            else:
                stock_value_krw += qty * price

        usd_cash_krw = usd_cash * fx_d if pd.notna(fx_d) else 0.0
        nav = krw_cash + usd_cash_krw + stock_value_krw

        records.append({
            "date": d,
            "nav": nav,
            "external_flow_krw": external_flow_krw,
            "krw_cash": krw_cash,
            "usd_cash": usd_cash,
            "stock_value_krw": stock_value_krw,
        })

    nav = pd.DataFrame(records).set_index("date")

    prev = nav["nav"].shift(1)
    nav["return"] = np.where(
        prev > 0,
        (nav["nav"] - prev - nav["external_flow_krw"]) / prev,
        np.nan
    )

    if (nav["krw_cash"] < -1).any() or (nav["usd_cash"] < -0.01).any():
        warnings.append(
            "일부 기간의 현금잔고가 음수입니다. 입출금 내역 또는 체결가격을 확인하세요."
        )

    return nav, warnings


# ============================================================
# Portfolio risk score
# ============================================================
def portfolio_risk_score(weights, avg_corr_120, cvar95, max_beta, max_sector_weight):
    w = np.asarray(weights, dtype=float)
    n = max(len(w), 1)

    hhi = float(np.sum(w ** 2))
    hhi_min = 1 / n
    conc = np.clip((hhi - hhi_min) / max(1 - hhi_min, 1e-9), 0, 1)

    corr_component = 0.5 if pd.isna(avg_corr_120) else np.clip((avg_corr_120 + 1) / 2, 0, 1)
    tail_component = 0.5 if pd.isna(cvar95) else np.clip(cvar95 / 0.06, 0, 1)
    beta_component = 0.5 if pd.isna(max_beta) else np.clip(abs(max_beta) / 2.0, 0, 1)
    sector_component = np.clip(max_sector_weight / 0.70, 0, 1)

    components = {
        "종목 집중도": conc,
        "상관 집중도": corr_component,
        "꼬리위험(CVaR)": tail_component,
        "시장 베타": beta_component,
        "섹터 집중도": sector_component,
    }

    score = 100 * (
        0.25 * conc
        + 0.20 * corr_component
        + 0.20 * tail_component
        + 0.20 * beta_component
        + 0.15 * sector_component
    )

    return int(round(np.clip(score, 0, 100))), components


# ============================================================
# UI - Settings and holdings
# ============================================================
st.title("📊 Portfolio Risk Terminal V3")
st.caption(
    "실제 매매이력 NAV · Risk Contribution · Risk Parity · Correlation · Beta · "
    "CVaR · Stress Test · Effective Bets · 원화/달러 현금"
)

with st.expander("⚙️ 분석 설정", expanded=False):
    c1, c2 = st.columns(2)
    history_period = c1.selectbox("가격 히스토리", ["2y", "5y", "10y"], index=1)
    rf = c2.number_input("무위험수익률(연, %)", 0.0, 20.0, 3.0, .1) / 100

    c3, c4 = st.columns(2)
    cov_window = c3.selectbox("Risk Parity 공분산 기간", [60, 120, 252], index=1)
    max_rp_weight = c4.slider("Risk Parity 종목당 최대비중", .20, 1.00, .50, .05)

    c5, c6 = st.columns(2)
    rc_alert = c5.slider("종목 위험기여도 경고", .20, .70, .35, .05)
    sector_alert = c6.slider("섹터 위험기여도 경고", .30, .90, .60, .05)

mode = st.radio(
    "현재 비중 계산 방식",
    ["실제 수량으로 자동 계산", "직접 비중 입력"],
    horizontal=True,
)

default_holdings = pd.DataFrame([
    {"ticker":"005930.KS","name":"삼성전자","market":"KR","sector":"반도체","quantity":150.0,"manual_weight":0.0},
    {"ticker":"000660.KS","name":"SK하이닉스","market":"KR","sector":"반도체","quantity":40.0,"manual_weight":0.0},
    {"ticker":"NVDA","name":"NVIDIA","market":"US","sector":"반도체/AI","quantity":0.0,"manual_weight":0.0},
    {"ticker":"SOXL","name":"SOXL","market":"US","sector":"반도체 ETF","quantity":0.0,"manual_weight":0.0},
    {"ticker":"CASH_KRW","name":"원화 현금","market":"CASH","sector":"현금","quantity":0.0,"manual_weight":0.0},
    {"ticker":"CASH_USD","name":"달러 현금","market":"USD_CASH","sector":"달러현금","quantity":0.0,"manual_weight":0.0},
])

if "holdings_v3" not in st.session_state:
    st.session_state["holdings_v3"] = default_holdings.copy()

uploaded_holdings = st.file_uploader(
    "포트폴리오 CSV 불러오기",
    type=["csv"],
    key="holdings_upload"
)

if uploaded_holdings is not None:
    try:
        h = pd.read_csv(uploaded_holdings)
        h = h.rename(columns={"override_weight":"manual_weight"})

        for col in ["ticker","name","market","sector","quantity","manual_weight"]:
            if col not in h.columns:
                h[col] = "" if col in ["ticker","name","market","sector"] else 0.0

        st.session_state["holdings_v3"] = h[
            ["ticker","name","market","sector","quantity","manual_weight"]
        ].copy()
    except Exception as e:
        st.error(f"포트폴리오 CSV 오류: {e}")

with st.expander("✏️ 보유종목 편집", expanded=True):
    holdings = st.data_editor(
        st.session_state["holdings_v3"],
        num_rows="dynamic",
        use_container_width=True,
        key="holdings_editor_v3",
        column_config={
            "ticker": st.column_config.TextColumn(
                "Ticker",
                help="삼성전자 005930.KS / SK하이닉스 000660.KS / 미국주식 NVDA"
            ),
            "name": st.column_config.TextColumn("종목명"),
            "market": st.column_config.SelectboxColumn(
                "시장",
                options=["KR","US","CASH","USD_CASH"],
                help="CASH=원화 현금 / USD_CASH=달러 현금"
            ),
            "sector": st.column_config.TextColumn(
                "섹터",
                help="위험집중 분석용. 예: 반도체, 전력, 원전, 에너지"
            ),
            "quantity": st.column_config.NumberColumn(
                "수량/현금액",
                min_value=0.0,
                step=1.0,
                format="%.4f",
                help="주식=보유수량 / CASH=원화 / USD_CASH=달러"
            ),
            "manual_weight": st.column_config.NumberColumn(
                "직접 비중",
                min_value=0.0,
                max_value=1.0,
                step=.01,
                format="%.2f",
                help="직접 비중 모드에서 30%=0.30"
            ),
        },
    )

    st.session_state["holdings_v3"] = holdings.copy()

    st.download_button(
        "현재 포트폴리오 CSV 저장",
        holdings.to_csv(index=False).encode("utf-8-sig"),
        file_name="portfolio.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Clean holdings.
df = holdings.copy()

for col in ["ticker","name","market","sector"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

df["ticker"] = df["ticker"].str.upper()
df["market"] = df["market"].str.upper()
df["sector"] = df["sector"].replace("", "미분류")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
df["manual_weight"] = pd.to_numeric(df["manual_weight"], errors="coerce").fillna(0.0)

df = df[
    (df["ticker"] != "")
    & df["market"].isin(["KR","US","CASH","USD_CASH"])
].copy()

if df.empty:
    st.info("보유종목을 입력해 주세요.")
    st.stop()


# ============================================================
# Actual trades editor
# ============================================================
default_trades = pd.DataFrame(columns=[
    "date","action","ticker","market","quantity","price","fee"
])

if "trades_v3" not in st.session_state:
    st.session_state["trades_v3"] = default_trades.copy()

with st.expander("🧾 실제 매매이력 NAV (선택)", expanded=False):
    st.caption(
        "실제 Sharpe/MDD/CVaR를 계산하려면 매매이력을 입력하세요. "
        "입출금은 DEPOSIT/WITHDRAW, 기존 보유분으로 시작하면 OPENING_POSITION을 사용합니다. "
        "price=0이면 해당 날짜 종가를 사용합니다."
    )

    uploaded_trades = st.file_uploader(
        "매매이력 CSV 불러오기",
        type=["csv"],
        key="trade_upload"
    )

    if uploaded_trades is not None:
        try:
            tr = pd.read_csv(uploaded_trades)
            for col in ["date","action","ticker","market","quantity","price","fee"]:
                if col not in tr.columns:
                    tr[col] = "" if col in ["date","action","ticker","market"] else 0.0

            st.session_state["trades_v3"] = tr[
                ["date","action","ticker","market","quantity","price","fee"]
            ].copy()
        except Exception as e:
            st.error(f"매매이력 CSV 오류: {e}")

    trades = st.data_editor(
        st.session_state["trades_v3"],
        num_rows="dynamic",
        use_container_width=True,
        key="trade_editor_v3",
        column_config={
            "date": st.column_config.TextColumn(
                "날짜",
                help="YYYY-MM-DD"
            ),
            "action": st.column_config.SelectboxColumn(
                "구분",
                options=TRADE_ACTIONS
            ),
            "ticker": st.column_config.TextColumn(
                "Ticker",
                help="입출금은 비워도 됨"
            ),
            "market": st.column_config.SelectboxColumn(
                "시장",
                options=["KR","US","CASH","USD_CASH"]
            ),
            "quantity": st.column_config.NumberColumn(
                "수량/금액",
                min_value=0.0,
                format="%.4f"
            ),
            "price": st.column_config.NumberColumn(
                "체결가격",
                min_value=0.0,
                format="%.4f",
                help="0이면 해당 날짜 종가"
            ),
            "fee": st.column_config.NumberColumn(
                "수수료",
                min_value=0.0,
                format="%.4f"
            ),
        }
    )

    st.session_state["trades_v3"] = trades.copy()

    st.download_button(
        "매매이력 CSV 저장",
        trades.to_csv(index=False).encode("utf-8-sig"),
        file_name="transactions.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# Fetch all needed prices, including closed historical tickers.
# ============================================================
trade_tickers = pd.DataFrame(columns=["ticker","market"])
if trades is not None and not trades.empty:
    tmp = trades.copy()
    tmp["ticker"] = tmp["ticker"].fillna("").astype(str).str.upper().str.strip()
    tmp["market"] = tmp["market"].fillna("").astype(str).str.upper().str.strip()
    trade_tickers = tmp[
        (tmp["ticker"] != "")
        & tmp["market"].isin(["KR","US"])
    ][["ticker","market"]].drop_duplicates()

asset_fetch_rows = pd.concat([
    df[df["market"].isin(["KR","US"])][["ticker","market"]],
    trade_tickers
], ignore_index=True).drop_duplicates("ticker")

with st.spinner("가격·환율·벤치마크·리스크 데이터를 계산 중..."):
    local_closes, krw_values, krw_returns, fx_close, latest_fx, data_errors = build_asset_data(
        asset_fetch_rows,
        history_period
    )

if data_errors:
    st.warning(
        "일부 가격 데이터 오류: "
        + " | ".join([f"{k}: {v}" for k,v in data_errors.items()])
    )


# ============================================================
# Current valuation and weights
# ============================================================
if mode == "실제 수량으로 자동 계산":
    market_values = []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        market = row["market"]
        qty = float(row["quantity"])

        if market == "CASH":
            value = qty

        elif market == "USD_CASH":
            value = qty * latest_fx if pd.notna(latest_fx) else 0.0

        elif ticker in local_closes and not local_closes[ticker].empty:
            price = float(local_closes[ticker].iloc[-1])
            if market == "US":
                value = qty * price * latest_fx
            else:
                value = qty * price

        else:
            value = 0.0

        market_values.append(value)

    df["market_value_krw"] = market_values
    total_value = float(df["market_value_krw"].sum())

    if total_value <= 0:
        st.error("수량/현금액을 입력해 주세요.")
        st.stop()

    df["current_weight"] = df["market_value_krw"] / total_value
    weight_mode = "실제 수량 자동 계산"

else:
    manual_total = float(df["manual_weight"].sum())
    if manual_total <= 0:
        st.error("직접 비중 모드에서는 비중을 입력해 주세요.")
        st.stop()

    df["current_weight"] = df["manual_weight"] / manual_total
    df["market_value_krw"] = np.nan
    total_value = np.nan
    weight_mode = "직접 비중 입력"


# ============================================================
# Current portfolio returns and risk sleeve
# ============================================================
asset_return_series = build_current_asset_returns(df, krw_returns, fx_close)

weight_map = {
    row["ticker"]: float(row["current_weight"])
    for _, row in df.iterrows()
    if row["current_weight"] > 0
}

current_port_ret = weighted_portfolio_return(asset_return_series, weight_map)

risk_assets = df[
    (df["current_weight"] > 0)
    & df["market"].isin(["KR","US","USD_CASH"])
    & df["ticker"].isin(asset_return_series.keys())
].copy()

risk_tickers = risk_assets["ticker"].tolist()
risk_weights = risk_assets["current_weight"].to_numpy(dtype=float)

if risk_tickers:
    risk_calendar = None
    for t in risk_tickers:
        idx = asset_return_series[t].index
        risk_calendar = idx if risk_calendar is None else risk_calendar.union(idx)

    risk_return_df = pd.DataFrame(index=risk_calendar.sort_values())
    for t in risk_tickers:
        risk_return_df[t] = asset_return_series[t].reindex(risk_return_df.index).fillna(0.0)

    cov = risk_return_df.tail(cov_window).cov().values * TRADING_DAYS
    _, rc_pct = risk_contribution(risk_weights, cov)

    risky_sleeve_weight = float(risk_weights.sum())
    rp_norm = risk_parity_weights(cov, max_rp_weight)
    rp_weights = rp_norm * risky_sleeve_weight

    div_ratio = diversification_ratio(risk_weights, risk_return_df.tail(252))
    effective_bets = effective_number_of_bets(rc_pct)

else:
    risk_return_df = pd.DataFrame()
    cov = np.empty((0,0))
    rc_pct = np.array([])
    rp_weights = np.array([])
    div_ratio = np.nan
    effective_bets = np.nan


# ============================================================
# Actual NAV from transactions
# ============================================================
market_map = {
    row["ticker"]: row["market"]
    for _, row in pd.concat([
        df[["ticker","market"]],
        trade_tickers
    ], ignore_index=True).drop_duplicates("ticker").iterrows()
}

actual_nav, nav_warnings = transaction_nav(
    trades,
    market_map,
    local_closes,
    fx_close
)

actual_returns = (
    actual_nav["return"].dropna()
    if not actual_nav.empty and "return" in actual_nav
    else pd.Series(dtype=float)
)

has_actual_nav = len(actual_returns) >= 20

metric_basis = st.radio(
    "성과·꼬리위험 지표 기준",
    ["현재 구성 백테스트", "실제 매매이력 NAV"],
    index=1 if has_actual_nav else 0,
    horizontal=True,
    disabled=False
)

if metric_basis == "실제 매매이력 NAV" and not has_actual_nav:
    st.warning("유효한 실제 NAV 수익률이 20거래일 미만이라 현재 구성 백테스트를 사용합니다.")
    metric_returns = current_port_ret
    metric_basis_label = "현재 구성 백테스트"
else:
    metric_returns = actual_returns if metric_basis == "실제 매매이력 NAV" else current_port_ret
    metric_basis_label = metric_basis


# ============================================================
# Benchmarks / factor data
# ============================================================
benchmark_defs = {
    "KOSPI": "^KS11",
    "SOX": "^SOX",
    "QQQ": "QQQ",
}

benchmark_returns = {}
benchmark_errors = {}

for label, ticker in benchmark_defs.items():
    try:
        s = fetch_close(ticker, history_period)
        if s.empty:
            raise ValueError("데이터 없음")
        benchmark_returns[label] = s.pct_change().dropna()
    except Exception as e:
        benchmark_errors[label] = str(e)

vix_close = fetch_close("^VIX", history_period)
tnx_close = fetch_close("^TNX", history_period)

vix_ret = vix_close.pct_change().dropna() if not vix_close.empty else pd.Series(dtype=float)
tnx_change = tnx_close.diff().dropna() if not tnx_close.empty else pd.Series(dtype=float)

beta_rows = []
for label, br in benchmark_returns.items():
    row = {"Benchmark": label}
    for w in [20,60,120]:
        row[f"{w}D Beta"] = beta_to(metric_returns.tail(w), br.tail(w * 2))
    beta_rows.append(row)

beta_df = pd.DataFrame(beta_rows)

all_betas = []
for col in ["20D Beta","60D Beta","120D Beta"]:
    if col in beta_df.columns:
        all_betas.extend(pd.to_numeric(beta_df[col], errors="coerce").dropna().tolist())

max_abs_beta = max([abs(x) for x in all_betas], default=np.nan)


# ============================================================
# Tail metrics, correlation, risk score
# ============================================================
vol = ann_vol(metric_returns)
aret = ann_return(metric_returns)
mdd = max_drawdown(metric_returns)
var95, hist_cvar95 = hist_var_cvar(metric_returns, .95)
ewma_cvar95 = ewma_cvar(metric_returns, .95, .94)

sharpe_windows = {
    "1M": sharpe(metric_returns.tail(21), rf),
    "3M": sharpe(metric_returns.tail(63), rf),
    "6M": sharpe(metric_returns.tail(126), rf),
    "1Y": sharpe(metric_returns.tail(252), rf),
}

stock_tickers_for_corr = [
    t for t in df.loc[
        (df["market"].isin(["KR","US"]))
        & (df["current_weight"] > 0),
        "ticker"
    ].tolist()
    if t in krw_returns
]

avg_corr20 = avg_pairwise_corr(krw_returns, stock_tickers_for_corr, 20)
avg_corr60 = avg_pairwise_corr(krw_returns, stock_tickers_for_corr, 60)
avg_corr120 = avg_pairwise_corr(krw_returns, stock_tickers_for_corr, 120)

sector_weight = (
    df.groupby("sector")["current_weight"].sum().sort_values(ascending=False)
)
max_sector_weight = float(sector_weight.iloc[0]) if len(sector_weight) else 0.0

risk_score, score_components = portfolio_risk_score(
    df["current_weight"].to_numpy(dtype=float),
    avg_corr120,
    hist_cvar95,
    max_abs_beta,
    max_sector_weight,
)


# ============================================================
# Headline dashboard
# ============================================================
c1, c2, c3 = st.columns(3)
c1.metric("Risk Score", f"{risk_score}/100")
c2.metric("Sharpe (1Y)", fmt_num(sharpe_windows["1Y"]))
c3.metric("연환산 변동성", fmt_pct(vol))

c4, c5, c6 = st.columns(3)
c4.metric("MDD", fmt_pct(mdd))
c5.metric("95% Hist. CVaR", fmt_pct(hist_cvar95))
c6.metric("95% EWMA CVaR", fmt_pct(ewma_cvar95))

c7, c8 = st.columns(2)
c7.metric("Diversification Ratio", fmt_num(div_ratio))
c8.metric("Effective Bets", fmt_num(effective_bets))

st.caption(
    f"성과지표 기준: {metric_basis_label} · 현재 비중 계산: {weight_mode}"
    + (f" · USD/KRW {latest_fx:,.2f}" if pd.notna(latest_fx) else "")
)


# ============================================================
# Risk alerts
# ============================================================
alerts = []

if len(rc_pct):
    for i, row in risk_assets.reset_index(drop=True).iterrows():
        rc = float(rc_pct[i])
        if rc >= rc_alert:
            alerts.append(
                f"🔴 {row['name']} 위험기여도 {rc:.0%} — 설정 경고선 {rc_alert:.0%} 초과"
            )

if len(rc_pct):
    rc_table_for_sector = risk_assets[["sector"]].copy().reset_index(drop=True)
    rc_table_for_sector["rc"] = rc_pct
    sector_rc = rc_table_for_sector.groupby("sector")["rc"].sum().sort_values(ascending=False)

    for sector, value in sector_rc.items():
        if value >= sector_alert:
            alerts.append(
                f"🔴 {sector} 섹터 위험기여도 {value:.0%} — 설정 경고선 {sector_alert:.0%} 초과"
            )
else:
    sector_rc = pd.Series(dtype=float)

if pd.notna(avg_corr20) and avg_corr20 >= .80:
    alerts.append(f"🟠 20D 평균 상관계수 {avg_corr20:.2f} — 분산효과 약화")

if pd.notna(avg_corr20) and pd.notna(avg_corr120) and (avg_corr20 - avg_corr120) >= .15:
    alerts.append(
        f"🟠 평균 상관계수 120D {avg_corr120:.2f} → 20D {avg_corr20:.2f} 급상승"
    )

if pd.notna(hist_cvar95) and hist_cvar95 >= .04:
    alerts.append(f"🟠 95% Historical CVaR {hist_cvar95:.1%} — 일간 꼬리위험 높음")

if pd.notna(max_abs_beta) and max_abs_beta >= 1.5:
    alerts.append(f"🟠 시장 베타 최대치 {max_abs_beta:.2f} — 고베타 포트폴리오")

if alerts:
    st.warning("\n\n".join(alerts))
else:
    st.success("현재 설정한 경고 기준에서 뚜렷한 리스크 경보가 없습니다.")


tabs = st.tabs([
    "Dashboard",
    "Risk",
    "Correlation",
    "Beta",
    "Stress",
    "Actual NAV",
    "Rebalance",
])


# ============================================================
# Dashboard tab
# ============================================================
with tabs[0]:
    st.subheader("현재 포트폴리오")

    view = df[[
        "ticker","name","market","sector","quantity","current_weight"
    ]].copy()

    def current_price_display(row):
        if row["market"] == "CASH":
            return 1.0
        if row["market"] == "USD_CASH":
            return latest_fx
        if row["ticker"] in local_closes and not local_closes[row["ticker"]].empty:
            return float(local_closes[row["ticker"]].iloc[-1])
        return np.nan

    view["현재가격/환율"] = view.apply(current_price_display, axis=1)

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
            "sector":"섹터",
            "quantity":st.column_config.NumberColumn("수량/현금액", format="%.4f"),
            "current_weight":st.column_config.NumberColumn("현재비중", format="%.3f"),
            "현재가격/환율":st.column_config.NumberColumn("현재가격/환율", format="%.2f"),
            "평가금액(KRW)":st.column_config.NumberColumn("평가금액(KRW)", format="%.0f"),
        }
    )

    st.subheader("Sharpe Ratio")
    st.bar_chart(
        pd.DataFrame(
            {"Sharpe": list(sharpe_windows.values())},
            index=list(sharpe_windows.keys())
        )
    )

    st.subheader("누적 수익률")
    wealth = (1 + metric_returns.fillna(0)).cumprod() - 1
    st.line_chart(pd.DataFrame({"Portfolio": wealth}))

    d1, d2 = st.columns(2)

    with d1:
        st.subheader("섹터 비중")
        st.dataframe(
            sector_weight.rename("비중").reset_index(),
            use_container_width=True,
            hide_index=True,
            column_config={
                "sector":"섹터",
                "비중":st.column_config.NumberColumn("비중", format="%.3f")
            }
        )

    with d2:
        st.subheader("Risk Score 구성")
        comp_df = pd.DataFrame({
            "항목": list(score_components.keys()),
            "위험도": list(score_components.values())
        })
        st.dataframe(
            comp_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "항목":"항목",
                "위험도":st.column_config.NumberColumn("0~1 위험도", format="%.2f")
            }
        )

    st.caption(
        "Risk Score는 업계 표준지표가 아니라 종목집중·상관·CVaR·베타·섹터집중을 합친 "
        "대시보드용 0~100 커스텀 위험점수입니다. 높을수록 위험집중이 큽니다."
    )


# ============================================================
# Risk tab
# ============================================================
with tabs[1]:
    st.subheader("Risk Contribution / Risk Parity")

    if not len(risk_assets):
        st.info("위험기여도를 계산할 자산이 없습니다.")
    else:
        rt = risk_assets[[
            "ticker","name","market","sector","current_weight"
        ]].copy().reset_index(drop=True)

        rt["risk_contribution"] = rc_pct
        rt["risk_parity_weight"] = rp_weights
        rt["difference"] = rt["risk_parity_weight"] - rt["current_weight"]
        rt["경고"] = np.where(
            rt["risk_contribution"] >= rc_alert,
            "🔴 과다",
            ""
        )

        st.dataframe(
            rt,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker":"Ticker",
                "name":"종목명",
                "market":"시장",
                "sector":"섹터",
                "current_weight":st.column_config.NumberColumn("현재비중", format="%.3f"),
                "risk_contribution":st.column_config.NumberColumn("위험기여도", format="%.3f"),
                "risk_parity_weight":st.column_config.NumberColumn("Risk Parity", format="%.3f"),
                "difference":st.column_config.NumberColumn("조정 필요", format="%.3f"),
                "경고":"상태",
            }
        )

        rc_chart = rt.set_index("name")[["current_weight","risk_contribution"]]
        rc_chart.columns = ["현재 비중","위험기여도"]
        st.bar_chart(rc_chart)

        st.subheader("섹터별 위험기여도")

        sector_rc_df = sector_rc.rename("risk_contribution").reset_index()
        sector_rc_df["상태"] = np.where(
            sector_rc_df["risk_contribution"] >= sector_alert,
            "🔴 과다",
            ""
        )

        st.dataframe(
            sector_rc_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "sector":"섹터",
                "risk_contribution":st.column_config.NumberColumn(
                    "위험기여도", format="%.3f"
                ),
                "상태":"상태"
            }
        )

        r1, r2 = st.columns(2)
        r1.metric("Diversification Ratio", fmt_num(div_ratio))
        r2.metric("Effective Number of Bets", fmt_num(effective_bets))

        st.caption(
            "Effective Bets는 위험기여도가 균등할수록 높습니다. "
            "보유종목이 6개여도 위험이 2개 종목에 몰리면 2에 가까운 값이 나올 수 있습니다."
        )


# ============================================================
# Correlation tab
# ============================================================
with tabs[2]:
    st.subheader("20D / 60D / 120D Correlation")

    if len(stock_tickers_for_corr) < 2:
        st.info("상관계수를 계산할 주식이 2개 이상 필요합니다.")
    else:
        names = df.set_index("ticker")["name"].to_dict()

        pair_rows = []
        for i in range(len(stock_tickers_for_corr)):
            for j in range(i+1, len(stock_tickers_for_corr)):
                a = stock_tickers_for_corr[i]
                b = stock_tickers_for_corr[j]

                c20, n20 = pair_corr(krw_returns[a], krw_returns[b], 20)
                c60, n60 = pair_corr(krw_returns[a], krw_returns[b], 60)
                c120, n120 = pair_corr(krw_returns[a], krw_returns[b], 120)

                delta = (
                    c20 - c120
                    if pd.notna(c20) and pd.notna(c120)
                    else np.nan
                )

                if pd.notna(c20) and c20 >= .80:
                    status = "🔴 높음"
                elif pd.notna(c20) and c20 >= .60:
                    status = "🟠 중간"
                else:
                    status = "🟢 낮음"

                if pd.notna(delta) and delta >= .15:
                    status += " / 상승"

                pair_rows.append({
                    "Pair": f"{names.get(a,a)} ↔ {names.get(b,b)}",
                    "20D": c20,
                    "60D": c60,
                    "120D": c120,
                    "20D-120D": delta,
                    "공통관측치": n120,
                    "상태": status,
                })

        pairs = pd.DataFrame(pair_rows).sort_values(
            ["20D"],
            ascending=False,
            na_position="last"
        )

        st.dataframe(
            pairs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pair":"종목쌍",
                "20D":st.column_config.NumberColumn("20D", format="%.2f"),
                "60D":st.column_config.NumberColumn("60D", format="%.2f"),
                "120D":st.column_config.NumberColumn("120D", format="%.2f"),
                "20D-120D":st.column_config.NumberColumn("상관 변화", format="%+.2f"),
                "공통관측치":"120D 관측치",
                "상태":"상태",
            }
        )

        st.subheader("전체 상관계수 매트릭스")

        window = st.radio(
            "표시 기간",
            [20,60,120],
            horizontal=True,
            key="corr_window_radio"
        )

        matrix = pd.DataFrame(
            np.eye(len(stock_tickers_for_corr)),
            index=stock_tickers_for_corr,
            columns=stock_tickers_for_corr
        )

        for i, a in enumerate(stock_tickers_for_corr):
            for j, b in enumerate(stock_tickers_for_corr):
                if i == j:
                    matrix.loc[a,b] = 1.0
                elif i < j:
                    c, _ = pair_corr(krw_returns[a], krw_returns[b], window)
                    matrix.loc[a,b] = c
                    matrix.loc[b,a] = c

        matrix = matrix.rename(index=names, columns=names)
        st.dataframe(matrix.round(2), use_container_width=True)

        st.subheader("평균 상관 추세")
        st.bar_chart(
            pd.DataFrame({
                "평균상관":[avg_corr20,avg_corr60,avg_corr120]
            }, index=["20D","60D","120D"])
        )

        with st.expander("🔎 종목별 데이터 상태 확인"):
            rows = []
            for _, row in df[df["market"].isin(["KR","US"])].iterrows():
                t = row["ticker"]

                if t in krw_returns and len(krw_returns[t]) >= 10:
                    status = "✅ Correlation 포함"
                    obs = len(krw_returns[t])
                    start = krw_returns[t].index.min().date()
                    end = krw_returns[t].index.max().date()
                    reason = ""
                else:
                    status = "❌ 유효 데이터 부족"
                    obs = len(krw_returns.get(t, []))
                    start = "-"
                    end = "-"
                    reason = data_errors.get(t, "수익률 관측치 부족")

                rows.append({
                    "종목명":row["name"],
                    "Ticker":t,
                    "시장":row["market"],
                    "상태":status,
                    "수익률 관측치":obs,
                    "데이터 시작":start,
                    "데이터 종료":end,
                    "오류/제외 이유":reason,
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# Beta tab
# ============================================================
with tabs[3]:
    st.subheader("Benchmark Beta")

    if beta_df.empty:
        st.info("벤치마크 데이터를 불러오지 못했습니다.")
    else:
        st.dataframe(
            beta_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Benchmark":"Benchmark",
                "20D Beta":st.column_config.NumberColumn("20D Beta", format="%.2f"),
                "60D Beta":st.column_config.NumberColumn("60D Beta", format="%.2f"),
                "120D Beta":st.column_config.NumberColumn("120D Beta", format="%.2f"),
            }
        )

    b1, b2 = st.columns(2)

    with b1:
        if not tnx_change.empty:
            rate_beta = beta_to(metric_returns.tail(252), tnx_change.tail(504))
            st.metric("10Y 금리 민감도", fmt_num(rate_beta, 3))
            st.caption(
                "^TNX 일간 금리변화(percentage point)에 대한 과거 단순 민감도입니다."
            )
        else:
            rate_beta = np.nan

    with b2:
        if not vix_ret.empty:
            vix_beta = beta_to(metric_returns.tail(252), vix_ret.tail(504))
            st.metric("VIX 민감도", fmt_num(vix_beta, 3))
            st.caption(
                "VIX 일간 %변화에 대한 과거 단순 민감도입니다."
            )
        else:
            vix_beta = np.nan

    st.caption(
        "Beta는 과거 공분산을 이용한 단순 민감도이므로 미래 손실을 보장하는 값이 아닙니다. "
        "특히 한국장과 미국장의 거래시간 차이 때문에 단기 Beta는 왜곡될 수 있습니다."
    )


# ============================================================
# Stress tab
# ============================================================
with tabs[4]:
    st.subheader("Scenario Stress Test")

    scenarios = {
        "반도체 급락": {
            "factor":"SOX",
            "equity_shock":-0.15,
            "fx_shock":0.03,
            "rate_shock":0.25,
            "vix_target":35.0,
        },
        "NASDAQ 조정": {
            "factor":"QQQ",
            "equity_shock":-0.10,
            "fx_shock":0.02,
            "rate_shock":0.25,
            "vix_target":30.0,
        },
        "한국 증시 급락": {
            "factor":"KOSPI",
            "equity_shock":-0.10,
            "fx_shock":0.05,
            "rate_shock":0.00,
            "vix_target":30.0,
        },
        "금리 충격": {
            "factor":"QQQ",
            "equity_shock":-0.05,
            "fx_shock":0.02,
            "rate_shock":0.50,
            "vix_target":30.0,
        },
        "Risk-off 복합": {
            "factor":"QQQ",
            "equity_shock":-0.15,
            "fx_shock":0.05,
            "rate_shock":0.25,
            "vix_target":40.0,
        },
    }

    scenario_name = st.selectbox("시나리오", list(scenarios.keys()))
    sc = scenarios[scenario_name]

    factor_beta = np.nan
    if sc["factor"] in benchmark_returns:
        factor_beta = beta_to(
            metric_returns.tail(252),
            benchmark_returns[sc["factor"]].tail(504)
        )

    usd_exposure = float(
        df.loc[
            df["market"].isin(["US","USD_CASH"]),
            "current_weight"
        ].sum()
    )

    equity_impact = (
        factor_beta * sc["equity_shock"]
        if pd.notna(factor_beta)
        else 0.0
    )

    fx_impact = usd_exposure * sc["fx_shock"]

    rate_beta = beta_to(
        metric_returns.tail(252),
        tnx_change.tail(504)
    ) if not tnx_change.empty else np.nan

    rate_impact = (
        rate_beta * sc["rate_shock"]
        if pd.notna(rate_beta)
        else 0.0
    )

    vix_beta = beta_to(
        metric_returns.tail(252),
        vix_ret.tail(504)
    ) if not vix_ret.empty else np.nan

    current_vix = float(vix_close.iloc[-1]) if not vix_close.empty else np.nan
    vix_shock_pct = (
        sc["vix_target"] / current_vix - 1
        if pd.notna(current_vix) and current_vix > 0
        else 0.0
    )

    vix_impact = (
        vix_beta * vix_shock_pct
        if pd.notna(vix_beta)
        else 0.0
    )

    estimated_impact = equity_impact + fx_impact + rate_impact + vix_impact

    st.metric(
        f"{scenario_name} 추정 충격",
        fmt_pct(estimated_impact, 2)
    )

    stress_breakdown = pd.DataFrame({
        "요인":["주식 Factor","USD/KRW","미10Y","VIX"],
        "가정":[
            f"{sc['factor']} {sc['equity_shock']:.0%}",
            f"{sc['fx_shock']:+.0%}",
            f"{sc['rate_shock']:+.2f}%p",
            f"VIX {sc['vix_target']:.0f}",
        ],
        "추정기여":[
            equity_impact,
            fx_impact,
            rate_impact,
            vix_impact,
        ]
    })

    st.dataframe(
        stress_breakdown,
        use_container_width=True,
        hide_index=True,
        column_config={
            "요인":"요인",
            "가정":"가정",
            "추정기여":st.column_config.NumberColumn("추정 손익기여", format="%.3f"),
        }
    )

    st.caption(
        "위 시나리오는 과거 단순 Beta/민감도를 합산한 1차 근사치입니다. "
        "극단적 시장에서는 상관관계·유동성이 급변하므로 실제 손실은 더 클 수 있습니다."
    )

    st.divider()
    st.subheader("종목별 직접 충격")

    stress = df[[
        "ticker","name","market","current_weight"
    ]].copy()

    stress["shock_pct"] = stress["market"].map({
        "KR":-8.0,
        "US":-10.0,
        "CASH":0.0,
        "USD_CASH":0.0,
    }).fillna(0.0)

    stress_edit = st.data_editor(
        stress,
        use_container_width=True,
        hide_index=True,
        disabled=["ticker","name","market","current_weight"],
        key="direct_stress_v3",
        column_config={
            "ticker":"Ticker",
            "name":"종목명",
            "market":"시장",
            "current_weight":st.column_config.NumberColumn("현재비중", format="%.3f"),
            "shock_pct":st.column_config.NumberColumn("충격(%)", step=1.0),
        }
    )

    direct_impact = float(
        (stress_edit["current_weight"] * stress_edit["shock_pct"] / 100).sum()
    )

    st.metric("직접 입력 예상 충격", fmt_pct(direct_impact, 2))

    st.divider()
    st.subheader("Historical Replay")

    hist_scenarios = {
        "2025년 4월 변동성 구간": ("2025-04-02","2025-04-08"),
        "2022년 긴축 충격 구간": ("2022-01-03","2022-10-14"),
        "사용자 지정": (None,None),
    }

    hist_name = st.selectbox(
        "과거 구간",
        list(hist_scenarios.keys()),
        key="hist_scenario"
    )

    if hist_name == "사용자 지정":
        hc1, hc2 = st.columns(2)
        start_date = hc1.date_input("시작일", value=date(2025,4,2))
        end_date = hc2.date_input("종료일", value=date(2025,4,8))
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
    else:
        s, e = hist_scenarios[hist_name]
        start_ts = pd.Timestamp(s)
        end_ts = pd.Timestamp(e)

    replay = current_port_ret.loc[
        (current_port_ret.index >= start_ts)
        & (current_port_ret.index <= end_ts)
    ]

    if len(replay) >= 2:
        replay_total = float((1 + replay).prod() - 1)
        replay_mdd = max_drawdown(replay)
        replay_worst = float(replay.min())

        rr1, rr2, rr3 = st.columns(3)
        rr1.metric("구간 수익률", fmt_pct(replay_total))
        rr2.metric("구간 MDD", fmt_pct(replay_mdd))
        rr3.metric("최악의 하루", fmt_pct(replay_worst))
    else:
        st.info("선택한 구간의 가격 데이터가 부족합니다.")


# ============================================================
# Actual NAV tab
# ============================================================
with tabs[5]:
    st.subheader("실제 매매이력 기반 NAV")

    if actual_nav.empty:
        st.info(
            "매매이력을 입력하면 실제 계좌 NAV와 실제 Sharpe/MDD/CVaR가 계산됩니다."
        )
    else:
        if nav_warnings:
            for w in sorted(set(nav_warnings)):
                st.warning(w)

        st.line_chart(actual_nav[["nav"]].rename(columns={"nav":"Actual NAV"}))

        if has_actual_nav:
            n1, n2, n3 = st.columns(3)
            n1.metric("Actual Sharpe 1Y", fmt_num(sharpe(actual_returns.tail(252), rf)))
            n2.metric("Actual MDD", fmt_pct(max_drawdown(actual_returns)))
            _, actual_cvar = hist_var_cvar(actual_returns)
            n3.metric("Actual 95% CVaR", fmt_pct(actual_cvar))

        st.dataframe(
            actual_nav.tail(30).reset_index(),
            use_container_width=True,
            hide_index=True,
            column_config={
                "date":"날짜",
                "nav":st.column_config.NumberColumn("NAV", format="%.0f"),
                "external_flow_krw":st.column_config.NumberColumn("외부 현금흐름", format="%.0f"),
                "krw_cash":st.column_config.NumberColumn("KRW Cash", format="%.0f"),
                "usd_cash":st.column_config.NumberColumn("USD Cash", format="%.2f"),
                "stock_value_krw":st.column_config.NumberColumn("주식평가액", format="%.0f"),
                "return":st.column_config.NumberColumn("일간수익률", format="%.4f"),
            }
        )

        st.caption(
            "입출금은 외부 현금흐름으로 제거한 뒤 일간 수익률을 계산합니다. "
            "따라서 단순 계좌잔고 증가가 투자수익으로 오인되는 문제를 줄입니다."
        )


# ============================================================
# Rebalance tab
# ============================================================
with tabs[6]:
    st.subheader("현재 비중 vs Risk Parity")

    if not len(risk_assets):
        st.info("Risk Parity를 계산할 위험자산이 없습니다.")
    else:
        rb = risk_assets[[
            "ticker","name","market","sector","current_weight"
        ]].copy().reset_index(drop=True)

        rb["risk_parity_weight"] = rp_weights
        rb["difference"] = rb["risk_parity_weight"] - rb["current_weight"]

        cash_rows = []

        krw_cash_w = float(
            df.loc[df["market"]=="CASH","current_weight"].sum()
        )

        if krw_cash_w > 0:
            cash_rows.append({
                "ticker":"CASH_KRW",
                "name":"원화 현금",
                "market":"CASH",
                "sector":"현금",
                "current_weight":krw_cash_w,
                "risk_parity_weight":krw_cash_w,
                "difference":0.0,
            })

        if cash_rows:
            rb = pd.concat(
                [rb,pd.DataFrame(cash_rows)],
                ignore_index=True
            )

        st.dataframe(
            rb,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker":"Ticker",
                "name":"종목명",
                "market":"시장",
                "sector":"섹터",
                "current_weight":st.column_config.NumberColumn("현재비중", format="%.3f"),
                "risk_parity_weight":st.column_config.NumberColumn("Risk Parity", format="%.3f"),
                "difference":st.column_config.NumberColumn("조정 필요", format="%.3f"),
            }
        )

        chart = rb.set_index("name")[[
            "current_weight",
            "risk_parity_weight"
        ]]
        chart.columns = ["현재 비중","Risk Parity"]
        st.bar_chart(chart)

        if weight_mode == "실제 수량 자동 계산" and pd.notna(total_value):
            rb["매수/매도금액(KRW)"] = rb["difference"] * total_value

            st.subheader("리밸런싱 필요 금액")
            st.dataframe(
                rb[["name","매수/매도금액(KRW)"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "name":"종목명",
                    "매수/매도금액(KRW)":st.column_config.NumberColumn(
                        "매수(+)/매도(-)",
                        format="%.0f"
                    ),
                }
            )

        st.caption(
            "Risk Parity는 원화 현금을 0변동성 준비자산으로 유지하고, "
            "주식 + USD_CASH(환율위험)를 위험자산 슬리브로 묶어 Equal Risk Contribution을 계산합니다."
        )


st.divider()
st.caption(
    "이 대시보드는 투자판단 보조용입니다. 무료 시세 데이터는 지연·누락될 수 있으며, "
    "Beta·상관계수·Risk Parity·Stress Test는 과거 데이터 기반 추정치입니다."
)
