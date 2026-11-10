import csv
import io
import math
import os
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="우리 동네 대기질 대시보드",
    page_icon="🌫️",
    layout="wide",
)

# ------------------------------------------------------------------
# 설정값
# ------------------------------------------------------------------
CSV_NAME = "last_amb_hour_time.csv"  # main.py와 같은 폴더(깃허브 저장소 루트)에 두세요
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_PATH = os.path.join(BASE_DIR, CSV_NAME)

TIME_KEYS = ["측정일시", "일시", "시간", "일자", "날짜", "datetime", "date", "time", "ymd"]
STATION_KEYS = ["측정소명", "측정소", "지점명", "관측소", "지점", "station", "stn", "시군구", "지역", "시도"]
ITEM_KEYS = ["측정항목", "항목명", "항목", "item", "pollutant", "물질"]
VALUE_KEYS = ["측정값", "농도", "값", "value", "result"]
HOUR_NAMES = ["시", "시간", "hour", "hh", "측정시"]
EXCLUDE_WORDS = ["코드", "번호", "code", "id", "순번", "위도", "경도", "lat", "lon", "year", "연도", "주소"]
EXCLUDE_EXACT = ["년", "월", "일", "시", "분", "hour", "day", "month", "year"]

COMBINED_TIME = "일시(일자+시 자동결합)"
NONE_OPTION = "(없음)"

GRADE_LABELS = ["좋음", "보통", "나쁨", "매우나쁨"]
GRADE_COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
GRADE_EMOJI = {"좋음": "🔵", "보통": "🟢", "나쁨": "🟠", "매우나쁨": "🔴"}
GRADE_COLOR_MAP = dict(zip(GRADE_LABELS, GRADE_COLORS))
# 환경부 통합대기환경지수 기준: 좋음/보통/나쁨 상한값, 그 이상은 매우나쁨 (PM: ㎍/㎥, 나머지: ppm)
GRADE_TH = {
    "PM10": [30, 80, 150],
    "PM25": [15, 35, 75],
    "O3": [0.030, 0.090, 0.150],
    "NO2": [0.030, 0.060, 0.200],
    "CO": [2, 9, 15],
    "SO2": [0.020, 0.050, 0.150],
}
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


# ------------------------------------------------------------------
# 도우미 함수
# ------------------------------------------------------------------
def grade_kind(name: str):
    """열 이름으로 PM10 / PM25 여부 판단 (등급 기준 적용용)."""
    n = str(name).lower().replace(" ", "").replace("_", "").replace("-", "")
    if "pm2.5" in n or "pm25" in n or "초미세" in n:
        return "PM25"
    if "pm10" in n or "미세먼지" in n:
        return "PM10"
    if n.startswith("o3") or "오존" in n:
        return "O3"
    if n.startswith("no2") or "이산화질소" in n:
        return "NO2"
    if n.startswith("so2") or "아황산" in n:
        return "SO2"
    if re.match(r"^co($|\()", n) or "일산화" in n:
        return "CO"
    return None


def get_grade(value, kind):
    if kind is None or pd.isna(value):
        return None
    th = GRADE_TH[kind]
    for i, t in enumerate(th):
        if value <= t:
            return GRADE_LABELS[i]
    return GRADE_LABELS[-1]


def fmt(v):
    if pd.isna(v):
        return "-"
    return f"{v:,.3f}" if abs(v) < 1 else f"{v:,.1f}"


def is_known_pollutant(name: str) -> bool:
    n = str(name).lower().replace(" ", "")
    if n == "co":
        return True
    keys = ["pm10", "pm2.5", "pm25", "o3", "no2", "so2", "미세", "오존", "이산화", "일산화", "아황산"]
    return any(k in n for k in keys)


def parse_datetime(s: pd.Series) -> pd.Series:
    """여러 형태의 날짜·시간 값을 datetime으로 변환 (24시 표기, YYYYMMDDHH 지원)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    txt = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    is_num = txt.str.fullmatch(r"\d{8,12}")
    if is_num.mean() > 0.9:
        n = int(txt[is_num].str.len().mode().iloc[0])
        if n == 8:
            return pd.to_datetime(txt, format="%Y%m%d", errors="coerce")
        if n == 10:
            base = pd.to_datetime(txt.str[:8], format="%Y%m%d", errors="coerce")
            hour = pd.to_numeric(txt.str[8:10], errors="coerce")
            return base + pd.to_timedelta(hour, unit="h")
        if n == 12:
            base = pd.to_datetime(txt.str[:8], format="%Y%m%d", errors="coerce")
            hh = pd.to_numeric(txt.str[8:10], errors="coerce")
            mm = pd.to_numeric(txt.str[10:12], errors="coerce")
            return base + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")
        return pd.to_datetime(txt, errors="coerce")
    colon = txt.str.extract(r"^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}):(\d{1,2})$")
    if colon[0].notna().mean() > 0.9:
        base = pd.to_datetime(colon[0].str.replace("/", "-").str.replace(".", "-", regex=False), errors="coerce")
        return base + pd.to_timedelta(pd.to_numeric(colon[1], errors="coerce"), unit="h")
    h24 = txt.str.extract(r"^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})[ T]+24(?::\d{2}){0,2}$")[0]
    if h24.notna().any():
        fixed = txt.where(h24.isna(), h24 + " 00:00")
        out = pd.to_datetime(fixed, errors="coerce")
        return out + pd.to_timedelta(h24.notna().astype(int), unit="D")
    return pd.to_datetime(txt, errors="coerce")


def looks_like_time(s: pd.Series) -> bool:
    sample = s.dropna().head(300)
    if sample.empty:
        return False
    try:
        parsed = parse_datetime(sample)
    except Exception:
        return False
    if parsed.notna().mean() < 0.8:
        return False
    years = parsed.dropna().dt.year
    return bool(years.between(1990, 2100).all())


def to_number(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")
    return out.where(out >= 0)  # 음수(-999 등 결측 코드)는 결측 처리


def order_candidates(columns, keys):
    cands = []
    for key in keys:
        for c in columns:
            if key in str(c).lower() and c not in cands:
                cands.append(c)
    return cands


def decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_airkorea_export(raw: bytes):
    """에어코리아 '측정소별 시간 자료' 내려받기 형식 (제목줄 + 항목/단위/등급·1시간 3단 머리글)."""
    lines = decode_bytes(raw).splitlines()
    sample = "\n".join(lines[:15])
    delim = max([",", "\t", ";"], key=lambda d: sample.count(d))
    rows = [[c.strip() for c in r] for r in csv.reader(lines, delimiter=delim)]

    hdr = next((i for i, r in enumerate(rows[:15]) if r and r[0].replace(" ", "") in ("날짜", "일시", "측정일시")), None)
    if hdr is None:
        return None
    pat = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}[: ]\s*\d{1,2}$")
    start = next((i for i in range(hdr + 1, min(len(rows), hdr + 8)) if rows[i] and pat.match(rows[i][0])), None)
    if start is None:
        return None

    width = max(len(r) for r in rows[hdr:])

    def pad(r):
        return r + [""] * (width - len(r))

    def ffill(r):
        out = pad(r)
        for j in range(1, width):
            if out[j] == "":
                out[j] = out[j - 1]
        return out

    between = [pad(r) for r in rows[hdr + 1:start] if r]
    sub_row = next((r for r in between if any("등급" in c for c in r)), None)
    unit_row = next((r for r in between if r is not sub_row and any(c for c in r[1:])), None)
    names = ffill(rows[hdr])
    units = ffill(unit_row) if unit_row else [""] * width

    value_idx, grade_idx = {}, {}
    for j in range(1, width):
        name = names[j]
        if not name:
            continue
        sub = sub_row[j] if sub_row else ""
        if "등급" in sub:
            grade_idx[name] = j
        else:
            value_idx[name] = (j, units[j])
    if not value_idx:
        return None

    data = [pad(r) for r in rows[start:] if r and pat.match(r[0])]
    times = pd.Series([re.sub(r"\s+", ":", r[0], count=1) for r in data])

    title = next((c for r in rows[:hdr] for c in r if c), "")
    station = title.split("(")[0].strip() or "측정소"
    address = title[title.find("(") + 1:title.rfind(")")].strip() if "(" in title else ""

    df = pd.DataFrame({"일시": parse_datetime(times)})
    df["측정소"] = station
    if address:
        df["측정소주소"] = address
    for name, (j, unit) in value_idx.items():
        u = unit.strip().strip("()")
        label = f"{name} ({u})" if u else name
        df[label] = to_number(pd.Series([r[j] for r in data]))
        if name in grade_idx:
            df[f"__등급__{label}"] = [r[grade_idx[name]] for r in data]
    return df.dropna(subset=["일시"]).sort_values("일시").reset_index(drop=True)


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def parse_csv(raw: bytes) -> pd.DataFrame:
    try:
        special = parse_airkorea_export(raw)
    except Exception:  # noqa: BLE001
        special = None
    if special is not None and len(special) > 0:
        return special
    last_err = None
    df = None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
            if df.shape[1] == 1:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if df is None:
        raise last_err
    df.columns = [str(c).strip() for c in df.columns]
    # 일자 + 시 열이 따로 있으면 하나로 결합한 열을 추가
    date_c = next((c for c in order_candidates(df.columns, TIME_KEYS) if looks_like_time(df[c])), None)
    hour_c = next((c for c in df.columns if str(c).lower() in HOUR_NAMES and not looks_like_time(df[c])), None)
    if date_c and hour_c and date_c != hour_c:
        hours = pd.to_numeric(df[hour_c].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        df[COMBINED_TIME] = parse_datetime(df[date_c]) + pd.to_timedelta(hours, unit="h")
    return df


def detect_time_col(df: pd.DataFrame):
    if COMBINED_TIME in df.columns:  # 일자 + 시가 따로 있는 데이터는 결합 열이 정답
        return COMBINED_TIME
    cands = order_candidates(df.columns, TIME_KEYS)
    cands += [c for c in df.columns if c not in cands]
    for c in cands:
        if looks_like_time(df[c]):
            return c
    return None


def detect_station_col(df: pd.DataFrame, time_col):
    for c in order_candidates(df.columns, STATION_KEYS):
        if c != time_col and df[c].nunique() <= 500 and not pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def add_bands(fig, kind, ymax, row=None, col=1):
    th = GRADE_TH[kind]
    lows = [0] + th
    highs = th + [max(ymax * 1.05, th[-1] * 1.3)]
    for lo, hi, color in zip(lows, highs, GRADE_COLORS):
        kw = dict(y0=lo, y1=hi, fillcolor=color, opacity=0.10, line_width=0)
        if row:
            kw.update(row=row, col=col)
        fig.add_hrect(**kw)


# ------------------------------------------------------------------
# 제목 / 데이터 입력
# ------------------------------------------------------------------
st.title("🌫️ 우리 동네 대기질 대시보드")

st.sidebar.header("⚙️ 설정")
uploaded = st.sidebar.file_uploader("CSV 파일 직접 올리기 (선택)", type=["csv"])

raw_bytes = None
source_label = ""
if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    source_label = f"업로드한 파일: {uploaded.name}"
elif os.path.exists(LOCAL_PATH):
    with open(LOCAL_PATH, "rb") as f:
        raw_bytes = f.read()
    source_label = f"저장소 파일: {CSV_NAME}"

if raw_bytes is None:
    st.info(
        f"`{CSV_NAME}` 파일을 찾지 못했습니다.\n\n"
        "- 깃허브 저장소에 `main.py`와 같은 위치로 CSV를 올리거나\n"
        "- 왼쪽 사이드바에서 CSV 파일을 직접 올려주세요."
    )
    st.stop()

try:
    raw_df = parse_csv(raw_bytes)
except Exception as e:  # noqa: BLE001
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

# ------------------------------------------------------------------
# 열 설정 (자동 인식 + 수정 가능)
# ------------------------------------------------------------------
auto_time = detect_time_col(raw_df)
auto_station = detect_station_col(raw_df, auto_time)

with st.sidebar.expander("🔧 열 설정 (자동 인식이 틀렸을 때)", expanded=auto_time is None):
    cols = list(raw_df.columns)
    time_col = st.selectbox(
        "시간 열",
        cols,
        index=cols.index(auto_time) if auto_time in cols else 0,
    )
    st_opts = [NONE_OPTION] + [c for c in cols if c != time_col]
    station_col = st.selectbox(
        "측정소(지역) 열",
        st_opts,
        index=st_opts.index(auto_station) if auto_station in st_opts else 0,
    )

work = raw_df.copy()
work["__시각__"] = parse_datetime(work[time_col])
work = work.dropna(subset=["__시각__"])
if work.empty:
    st.error("시간 열을 날짜로 해석하지 못했습니다. 사이드바의 '열 설정'에서 시간 열을 다시 골라주세요.")
    st.stop()

has_station = station_col != NONE_OPTION
if has_station:
    work[station_col] = work[station_col].astype(str).str.strip()

single_station = None  # 측정소가 1곳뿐인 파일이면 이름만 보여주고 선택·비교 기능은 숨김
if has_station and work[station_col].nunique() == 1:
    single_station = work[station_col].iloc[0]
    has_station = False

# 세로(long) 형태 데이터 자동 처리: 항목명 열 + 값 열 -> 가로(wide)로 변환
item_c = next(
    (c for c in order_candidates(work.columns, ITEM_KEYS) if not pd.api.types.is_numeric_dtype(work[c]) and work[c].nunique() <= 30),
    None,
)
value_c = next((c for c in order_candidates(work.columns, VALUE_KEYS) if c != item_c), None)
if item_c and value_c:
    work["__값__"] = to_number(work[value_c])
    idx = ["__시각__"] + ([station_col] if has_station else [])
    work = work.pivot_table(index=idx, columns=item_c, values="__값__", aggfunc="mean").reset_index()
    work.columns.name = None

# 오염물질(숫자) 열 후보
skip = {time_col, station_col, "__시각__", COMBINED_TIME}
numeric_cols = []
for c in work.columns:
    if c in skip:
        continue
    name = str(c).lower()
    if name in EXCLUDE_EXACT or any(w in name for w in EXCLUDE_WORDS):
        continue
    conv = to_number(work[c])
    if conv.notna().mean() >= 0.3:
        numeric_cols.append(c)

if not numeric_cols:
    st.error("분석할 숫자 열을 찾지 못했습니다. 파일 내용을 확인해주세요.")
    st.dataframe(raw_df.head(20), width="stretch")
    st.stop()

known = [c for c in numeric_cols if is_known_pollutant(c)]
default_sel = (known or numeric_cols)[:6]

with st.sidebar.expander("🔧 열 설정 (자동 인식이 틀렸을 때)", expanded=False):
    pollutants = st.multiselect("분석할 항목(숫자 열)", numeric_cols, default=default_sel)

if not pollutants:
    st.warning("분석할 항목을 하나 이상 선택해주세요.")
    st.stop()

for c in pollutants:
    work[c] = to_number(work[c])

# ------------------------------------------------------------------
# 필터 (측정소 / 기간)
# ------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 보기 조건")

all_work = work  # 측정소 비교용 (측정소 필터 적용 전)

if has_station:
    stations = sorted(work[station_col].dropna().unique().tolist())
    station_choice = st.sidebar.selectbox("측정소(지역)", ["전체 평균"] + stations)
    if station_choice != "전체 평균":
        work = work[work[station_col] == station_choice]
else:
    station_choice = single_station or "전체"

tmin, tmax = work["__시각__"].min().date(), work["__시각__"].max().date()
if tmin == tmax:
    d_range = (tmin, tmax)
else:
    d_range = st.sidebar.date_input("기간", value=(tmin, tmax), min_value=tmin, max_value=tmax)
    if not isinstance(d_range, (tuple, list)) or len(d_range) != 2:
        st.sidebar.info("시작일과 종료일을 모두 선택해주세요.")
        st.stop()

d0, d1 = pd.Timestamp(d_range[0]), pd.Timestamp(d_range[1]) + pd.Timedelta(days=1)
mask = (work["__시각__"] >= d0) & (work["__시각__"] < d1)
work = work[mask]
all_work = all_work[(all_work["__시각__"] >= d0) & (all_work["__시각__"] < d1)]

if work.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다. 측정소나 기간을 바꿔보세요.")
    st.stop()

# 시간별 대표값 (여러 측정소면 평균)
ts = work.groupby("__시각__")[pollutants].mean().sort_index()

# ------------------------------------------------------------------
# 상단 요약
# ------------------------------------------------------------------
addr = ""
if "측정소주소" in work.columns and not work["측정소주소"].dropna().empty:
    addr = f" ({work['측정소주소'].dropna().iloc[0]})"
st.caption(
    f"{source_label} · 측정소: **{station_choice}**{addr} · 기간: {d_range[0]} ~ {d_range[1]} · "
    f"데이터 {len(work):,}행"
)


def latest_grade(p, t):
    """파일에 공식 등급 열이 있으면 그 값을 우선 사용."""
    gcol = f"__등급__{p}"
    if gcol in work.columns:
        sub = work.loc[work["__시각__"] == t, gcol].dropna().astype(str)
        sub = sub[sub.isin(GRADE_LABELS)]
        if len(sub):
            return sub.iloc[0]
    return None


card_items = pollutants[:6]
for start in range(0, len(card_items), 3):
    cards = st.columns(3)
    for col, p in zip(cards, card_items[start:start + 3]):
        valid = ts[p].dropna()
        with col:
            if valid.empty:
                st.metric(p, "-")
                st.caption("유효한 값이 없습니다.")
                continue
            latest, avg = valid.iloc[-1], valid.mean()
            st.metric(
                f"{p} (최근값)",
                fmt(latest),
                delta=f"{latest - avg:+.3g} (평균 대비)",
                delta_color="inverse",
            )
            g = latest_grade(p, valid.index[-1]) or get_grade(latest, grade_kind(p))
            grade_txt = f"{GRADE_EMOJI.get(g, '⚪')} {g} · " if g else ""
            st.caption(f"{grade_txt}평균 {fmt(avg)} · 최고 {fmt(valid.max())} · 최저 {fmt(valid.min())}")
            st.caption(f"측정 시각: {valid.index[-1]:%Y-%m-%d %H시}")

st.markdown("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 시간 추이", "🗓️ 일별·시간대 패턴", "📍 측정소 비교", "🔗 상관관계·분포", "📄 원본 데이터"]
)

# ------------------------------------------------------------------
# 탭1: 시간 추이
# ------------------------------------------------------------------
with tab1:
    c1, c2 = st.columns([2, 1])
    default_idx = 1 if len(ts) > 3000 else 0
    unit = c1.radio("집계 단위", ["시간별 (원본)", "일평균", "주평균"], index=default_idx, horizontal=True)
    show_bands = c2.checkbox("등급 구간 배경색 표시", value=True)

    rule = {"시간별 (원본)": None, "일평균": "D", "주평균": "W"}[unit]
    ts_plot = ts if rule is None else ts.resample(rule).mean()

    n = len(pollutants)
    ncols = 2 if n >= 4 else 1
    nrows = math.ceil(n / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        vertical_spacing=0.12 if nrows > 1 else 0.1,
        horizontal_spacing=0.06,
        subplot_titles=[str(p) for p in pollutants],
    )
    palette = px.colors.qualitative.Set2
    for i, p in enumerate(pollutants):
        r, c = i // ncols + 1, i % ncols + 1
        y = ts_plot[p]
        fig.add_trace(
            go.Scatter(
                x=y.index,
                y=y.values,
                mode="lines+markers" if len(y) <= 100 else "lines",
                marker=dict(size=5),
                name=str(p),
                line=dict(width=1.8, color=palette[i % len(palette)]),
                connectgaps=False,
                hovertemplate="%{x|%Y-%m-%d %H시}<br>%{y:.4g}<extra>" + str(p) + "</extra>",
            ),
            row=r,
            col=c,
        )
        k = grade_kind(p)
        if show_bands and k and y.notna().any():
            add_bands(fig, k, float(y.max()), row=r, col=c)
    fig.update_layout(
        height=max(380, 300 * nrows),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")
    if show_bands and any(grade_kind(p) for p in pollutants):
        st.caption("배경색: 🔵 좋음  🟢 보통  🟠 나쁨  🔴 매우나쁨 (환경부 통합대기환경지수 기준)")

# ------------------------------------------------------------------
# 탭2: 일별 · 시간대 패턴
# ------------------------------------------------------------------
with tab2:
    p = st.selectbox("항목 선택", pollutants, key="pattern_item")
    kind = grade_kind(p)
    if (ts.index.max() - ts.index.min()).days < 3:
        st.caption("ℹ️ 데이터 기간이 3일 미만이라 일별·요일 패턴은 참고용으로만 봐주세요.")

    daily = ts[p].resample("D").mean().dropna().reset_index()
    daily.columns = ["날짜", "값"]
    if kind:
        daily["등급"] = daily["값"].apply(lambda v: get_grade(v, kind))
        fig_d = px.bar(
            daily,
            x="날짜",
            y="값",
            color="등급",
            color_discrete_map=GRADE_COLOR_MAP,
            category_orders={"등급": GRADE_LABELS},
            title=f"{p} 일평균 (일평균 값 기준 등급 색상)",
        )
    else:
        fig_d = px.bar(daily, x="날짜", y="값", title=f"{p} 일평균")
        fig_d.update_traces(marker_color="#3b82f6")
    fig_d.update_layout(template="plotly_white", height=380, yaxis_title=f"{p} 일평균", margin=dict(t=50, b=10))
    st.plotly_chart(fig_d, width="stretch")

    left, right = st.columns(2)

    tmp = ts[[p]].copy()
    tmp["시"] = tmp.index.hour
    tmp["요일"] = tmp.index.dayofweek

    hourly = tmp.groupby("시")[p].mean().reset_index()
    fig_h = px.line(hourly, x="시", y=p, markers=True, title=f"{p} 시간대별 평균 (하루 중 변화)")
    fig_h.update_layout(
        template="plotly_white", height=360, xaxis_title="시각(시)", yaxis_title=f"{p} 평균", margin=dict(t=50, b=10)
    )
    left.plotly_chart(fig_h, width="stretch")

    heat = tmp.pivot_table(index="요일", columns="시", values=p, aggfunc="mean").reindex(range(7))
    fig_hm = px.imshow(
        heat.values,
        x=[f"{h}시" for h in heat.columns],
        y=WEEKDAYS,
        color_continuous_scale="YlOrRd",
        aspect="auto",
        title=f"{p} 요일 × 시간대 평균",
        labels=dict(color=p),
    )
    fig_hm.update_layout(template="plotly_white", height=360, margin=dict(t=50, b=10))
    right.plotly_chart(fig_hm, width="stretch")

# ------------------------------------------------------------------
# 탭3: 측정소 비교
# ------------------------------------------------------------------
with tab3:
    if not has_station:
        if single_station:
            st.info(f"이 파일에는 측정소가 '{single_station}' 1곳뿐이라 측정소 비교는 할 수 없습니다. 여러 측정소 파일을 합치면 비교 그래프가 나타납니다.")
        else:
            st.info("측정소(지역) 열이 없어 비교 그래프를 만들 수 없습니다. 사이드바 '열 설정'에서 지정해보세요.")
    else:
        p3 = st.selectbox("항목 선택", pollutants, key="station_item")
        comp = all_work.groupby(station_col)[p3].mean().dropna().sort_values(ascending=False).reset_index()
        comp.columns = ["측정소", "평균"]
        if comp.empty:
            st.warning("비교할 데이터가 없습니다.")
        else:
            shown = comp.head(40)
            k3 = grade_kind(p3)
            if k3:
                shown = shown.assign(등급=shown["평균"].apply(lambda v: get_grade(v, k3)))
                fig_s = px.bar(
                    shown.iloc[::-1],
                    x="평균",
                    y="측정소",
                    orientation="h",
                    color="등급",
                    color_discrete_map=GRADE_COLOR_MAP,
                    category_orders={"등급": GRADE_LABELS},
                )
            else:
                fig_s = px.bar(shown.iloc[::-1], x="평균", y="측정소", orientation="h", color="평균", color_continuous_scale="YlOrRd")
            fig_s.update_layout(
                template="plotly_white",
                title=f"측정소별 {p3} 평균 (선택 기간)",
                height=max(380, 24 * len(shown) + 120),
                xaxis_title=f"{p3} 평균",
                margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig_s, width="stretch")
            if len(comp) > 40:
                st.caption(f"측정소가 {len(comp)}곳이라 평균이 높은 상위 40곳만 표시했습니다.")
            c_a, c_b = st.columns(2)
            c_a.metric("가장 높은 곳", comp.iloc[0]["측정소"], f"{comp.iloc[0]['평균']:.3g}", delta_color="off")
            c_b.metric("가장 낮은 곳", comp.iloc[-1]["측정소"], f"{comp.iloc[-1]['평균']:.3g}", delta_color="off")

# ------------------------------------------------------------------
# 탭4: 상관관계 · 분포
# ------------------------------------------------------------------
with tab4:
    if len(pollutants) >= 2:
        corr = ts[pollutants].corr()
        fig_c = px.imshow(
            corr,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title="항목 간 상관관계 (1에 가까울수록 같이 오르내림)",
        )
        fig_c.update_layout(template="plotly_white", height=420, margin=dict(t=50, b=10))
        st.plotly_chart(fig_c, width="stretch")
    else:
        st.info("상관관계는 항목을 2개 이상 선택하면 볼 수 있습니다.")

    p4 = st.selectbox("분포를 볼 항목", pollutants, key="dist_item")
    vals = ts[p4].dropna()
    if vals.empty:
        st.warning("유효한 값이 없습니다.")
    else:
        fig_hist = px.histogram(vals, nbins=40, marginal="box", labels={"value": p4, "count": "빈도"}, title=f"{p4} 분포")
        fig_hist.update_layout(template="plotly_white", height=420, showlegend=False, yaxis_title="빈도", margin=dict(t=50, b=10))
        st.plotly_chart(fig_hist, width="stretch")

# ------------------------------------------------------------------
# 탭5: 원본 데이터
# ------------------------------------------------------------------
with tab5:
    st.write(f"전체 열: {', '.join(map(str, raw_df.columns))}")
    show = work.drop(columns=["__시각__"], errors="ignore").rename(
        columns=lambda c: str(c).replace("__등급__", "등급: ") if str(c).startswith("__등급__") else c
    )
    st.dataframe(show.head(1000), width="stretch")
    st.caption("표에는 최대 1,000행까지만 표시됩니다.")
    st.download_button(
        "선택 조건 데이터 CSV 내려받기",
        data=show.to_csv(index=False).encode("utf-8-sig"),
        file_name="대기질_선택데이터.csv",
        mime="text/csv",
    )
