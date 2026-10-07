# -*- coding: utf-8 -*-
"""
SMT 공정 품질관리 시스템 (현장 관리자용)
========================================
검사 이미지를 촬영 시각순으로 공급하면, 최종 멀티모달 모델(YOLO11n + FiLM, 이미지 + 해당 이미지의
센서 5초값)이 이미지마다 정상 / 불량 여부와 불량 유형을 즉시 판정한다.
개별 판정은 이미지 단위로 독립적이며, 공정 단위 경보는 검사 결과를 구간으로 집계한 불량률 관리도(p 관리도)와
센서 관리도(마할라노비스 거리 + EWMA)로 판단한다.
"""

import sys
import time
import json
import zipfile
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------------------------------------------
# 기본 설정
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMG_DIR = DATA_DIR / "stream_images"
ZIP_DIR = DATA_DIR / "stream_zips"
ONNX_PATH = DATA_DIR / "model" / "model.onnx"
SCALER_PATH = DATA_DIR / "model" / "sensor_scaler.json"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

WINDOW = 60   # 추이 차트에 보여주는 최근 검사 건수
WIN_N = 20    # 불량률 집계 구간 (최근 N건)
P0 = 0.10     # 허용 불량률 (이를 넘으면 주의, p 관리도 관리상한을 넘으면 경고)

SENSOR_LABELS = [("temp_mean", "온도"), ("hum_mean", "습도"), ("vib_mean", "진동"),
                 ("acc_mean", "가속도"), ("noise_mean", "소음")]
LEVEL_TEXT = {0: "정상", 1: "주의", 2: "경고"}

# --------------------------------------------------------------------------------------
# 스타일 (라이트 테마)
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; color: #0b0b0b; }
    section[data-testid="stSidebar"] { background-color: #f9f9f7; border-right: 1px solid rgba(11,11,11,0.08); }

    .kpi-card {
        background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10); border-radius: 8px;
        padding: 14px 16px; text-align: center; box-shadow: 0 1px 2px rgba(11,11,11,0.04);
    }
    .kpi-label { font-size: 0.74rem; color: #898781; letter-spacing: .03em; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
    .kpi-sub { font-size: 0.78rem; margin-top: 4px; color: #52514e; }
    .status-good { color: #006300; } .status-critical { color: #b42318; }
    .status-warning { color: #9a6700; } .status-info { color: #1c5cab; }

    .verdict-bad {
        background: rgba(208,59,59,0.07); border: 1px solid #d03b3b; border-left: 4px solid #b42318;
        border-radius: 8px; padding: 16px 22px; color: #b42318; min-height: 98px;
    }
    .verdict-bad .t1 { font-size: 1.1rem; font-weight: 700; }
    .verdict-bad .t2 { font-size: 0.9rem; font-weight: 500; margin-top: 6px; color: #7a1f17; }
    .verdict-good {
        background: rgba(10,99,12,0.06); border: 1px solid #0ca30c; border-left: 4px solid #006300;
        border-radius: 8px; padding: 16px 22px; color: #006300; min-height: 98px;
    }
    .verdict-good .t1 { font-size: 1.1rem; font-weight: 700; }
    .verdict-good .t2 { font-size: 0.9rem; font-weight: 500; margin-top: 6px; color: #2f6a31; }
    .verdict-warn {
        background: rgba(212,160,23,0.09); border: 1px solid #d4a017; border-left: 4px solid #9a6700;
        border-radius: 8px; padding: 16px 22px; color: #9a6700; min-height: 98px;
    }
    .verdict-warn .t1 { font-size: 1.1rem; font-weight: 700; }
    .verdict-warn .t2 { font-size: 0.9rem; font-weight: 500; margin-top: 6px; color: #6b4a00; }
    .banner-tag { font-size: 0.72rem; font-weight: 700; letter-spacing: .04em; opacity: .85; margin-bottom: 4px; }
    .side-label { font-size: 0.78rem; font-weight: 700; color: #52514e; letter-spacing: .02em; margin: 2px 0 6px 2px; }
    .control-label { font-size: 0.82rem; font-weight: 700; color: #52514e; margin-bottom: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# 데이터 로딩
# --------------------------------------------------------------------------------------
def ensure_images():
    """data/stream_images 에 이미지가 없으면 data/stream_zips 의 원본 zip 에서 풀어 둔다 (최초 1회)."""
    n_have = len(list(IMG_DIR.glob("*.jpg"))) if IMG_DIR.exists() else 0
    if n_have >= 559:
        return n_have
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for z in sorted(ZIP_DIR.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                base = Path(name).name
                if base.lower().endswith(".jpg") and not (IMG_DIR / base).exists():
                    (IMG_DIR / base).write_bytes(zf.read(name))
    return len(list(IMG_DIR.glob("*.jpg")))


@st.cache_data
def load_stream():
    return pd.read_csv(DATA_DIR / "stream_sequence.csv", parse_dates=["timestamp"])


@st.cache_data
def load_baseline():
    with open(DATA_DIR / "spc_baseline.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    with open(DATA_DIR / "predictions.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="최종 모델 로딩 중...")
def load_detector():
    """(검출기, 오류 메시지). onnxruntime / opencv 가 설치되지 않았으면 (None, 메시지)."""
    try:
        from fusion_infer import FusionDetector
        return FusionDetector(ONNX_PATH, SCALER_PATH), None
    except ImportError as e:
        return None, str(e)


def run_detection(row, conf_th):
    """현재 이미지 + 그 이미지의 실제 센서 요약통계 35개를 함께 모델에 넣어 실시간 검출."""
    from fusion_infer import draw_detections
    det, _ = load_detector()
    feat35 = [row["f_" + c] for c in det.cols]
    im, dets = det.predict(IMG_DIR / row["filename"], feat35, conf=conf_th, iou=0.7)
    dets = sorted(dets, key=lambda d: -d["confidence"])
    return draw_detections(im, dets), [(d["class_name"], d["confidence"]) for d in dets]


@st.cache_data
def compute_results(conf_th: float, win_n: int, p0: float):
    """이미지별 판정과 공정 경보 수준 계산 (사전 계산된 모델 예측 사용).

    이미지 판정: 신뢰도 임계값 이상의 불량 클래스 박스가 1개라도 있으면 '불량', 없으면 '정상'.
                 불량 유형은 불량 박스 중 신뢰도가 가장 높은 클래스.
    불량률 관리도(p 관리도): 최근 win_n건의 불량 판정 비율. 허용 불량률 p0 와 관리상한
                 UCL = p0 + 3 * sqrt(p0 (1 - p0) / win_n) 로 판단 (초과 = 경고, p0 초과 = 주의).
    센서 관리도: 단일점 거리 D 가 한계 초과 = 경고, EWMA 가 한계 초과 = 주의.
    """
    df = load_stream().copy()
    base = load_baseline()
    preds = load_predictions()
    n_def, max_def, top_def, kinds = [], [], [], []
    for f in df["filename"]:
        d = [x for x in preds[f] if x["class_name"].startswith("불량_") and x["confidence"] >= conf_th]
        n_def.append(len(d))
        max_def.append(max([x["confidence"] for x in d], default=0.0))
        top_def.append(max(d, key=lambda x: x["confidence"])["class_name"].replace("불량_", "") if d else "")
        c = Counter(x["class_name"].replace("불량_", "") for x in d)
        kinds.append(", ".join(f"{k} {v}" for k, v in c.most_common()))
    df["n_def"], df["max_def"], df["top_def"], df["kinds"] = n_def, max_def, top_def, kinds
    df["pred"] = np.where(df["n_def"] > 0, "불량", "정상")
    truth_bad = df["label"] == "불량"
    pred_bad = df["pred"] == "불량"
    df["match"] = np.where(truth_bad == pred_bad, "일치", np.where(pred_bad, "오검출", "미검출"))

    # 불량률 관리도
    ucl_rate = min(1.0, p0 + 3 * np.sqrt(p0 * (1 - p0) / win_n))
    is_def = pred_bad.astype(float)
    df["rate"] = is_def.rolling(win_n, min_periods=min(win_n, 5)).mean()
    df["rate_level"] = np.where(df["rate"] > ucl_rate, 2, np.where(df["rate"] > p0, 1, 0))
    df.loc[df["rate"].isna(), "rate_level"] = 0
    df["ucl_rate"] = ucl_rate

    # 센서 관리도
    df["sensor_level"] = np.where(df["D"] > base["single_point_ucl"], 2, np.where(df["ewma"] > base["ewma_ucl"], 1, 0))
    df["level"] = np.maximum(df["rate_level"], df["sensor_level"])
    return df


def channel_share(row, base):
    """현재 샘플의 채널별 이상 기여도(%) = 기준선 대비 편차 제곱의 비율."""
    z2 = np.array([((row[k] - base["mu"][i]) / base["sigma"][i]) ** 2 for i, (k, _) in enumerate(SENSOR_LABELS)])
    return z2 / max(z2.sum(), 1e-12) * 100


def build_events(d, p0):
    """경보 수준이 올라간 시점을 이력으로 만든다."""
    rows = []
    for col, src in [("rate_level", "불량률"), ("sensor_level", "센서")]:
        lv = d[col].to_numpy()
        prev = 0
        for i in range(len(d)):
            if lv[i] > prev and lv[i] >= 1:
                r = d.iloc[i]
                if src == "불량률":
                    detail = f"구간 불량률 {r['rate']:.0%} (허용 {p0:.0%}, 관리상한 {r['ucl_rate']:.0%})"
                else:
                    detail = f"거리 D {r['D']:.2f}, EWMA {r['ewma']:.2f} / 주요 채널 {r['top_channel_1']} {r['top_pct_1']:.0f}%"
                rows.append({"time": r["timestamp"], "no": i + 1, "구분": src, "수준": LEVEL_TEXT[int(lv[i])], "내용": detail})
            prev = lv[i]
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("time", ascending=False)
    return out


if not (DATA_DIR / "stream_sequence.csv").exists():
    st.error("data/stream_sequence.csv 가 없습니다. README 의 '스트림 데이터 만들기'를 참고하세요.")
    st.stop()
with st.spinner("검사 이미지 준비 중 (최초 1회)..."):
    n_imgs = ensure_images()
if n_imgs < 559:
    st.error(f"검사 이미지가 {n_imgs}장뿐입니다. data/stream_zips 폴더에 Validation 원천 zip 3개가 있는지 확인하세요.")
    st.stop()

baseline = load_baseline()

# 검출 임계값은 본문 컨트롤 바에서 선택 (위젯 값은 session_state 로 먼저 읽음)
conf_th = float(st.session_state.get("conf_th", 0.25))
win_n, p0 = WIN_N, P0
df = compute_results(conf_th, win_n, p0)
N = len(df)

# --------------------------------------------------------------------------------------
# 세션 상태
# --------------------------------------------------------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False   # 시작 버튼을 눌러야 재생 시작
if "idx" not in st.session_state:
    st.session_state.idx = 0

# 좌측 탭: 시작 / 정지 / 처음 버튼만
with st.sidebar:
    st.markdown("<div class='side-label'>품질검사 진행</div>", unsafe_allow_html=True)
    sb1, sb2, sb3 = st.columns(3)
    with sb1:
        if st.button("시작", use_container_width=True):
            if st.session_state.idx >= N - 1:
                st.session_state.idx = 0
            st.session_state.running = True
    with sb2:
        if st.button("정지", use_container_width=True):
            st.session_state.running = False
    with sb3:
        if st.button("처음", use_container_width=True):
            st.session_state.idx = 0
            st.session_state.running = False

# --------------------------------------------------------------------------------------
# 헤더
# --------------------------------------------------------------------------------------
head_l, head_r = st.columns([3, 1])
with head_l:
    st.markdown("## 실시간 품질 검사")
    st.caption("Line C-3 / 사전공정 / 이미지별 정상, 불량 및 불량 유형 판정과 공정 경보")
with head_r:
    live_running = st.session_state.running
    live_text = "재생 중" if live_running else "일시정지"
    live_class = "status-good" if live_running else "status-warning"
    st.markdown(
        f"<div style='text-align:right; padding-top:10px;' class='{live_class}'><b>{live_text}</b></div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# 제어 / 설정 (본문 상단 컨트롤 바)
# --------------------------------------------------------------------------------------
with st.container(border=True):
    cb4, cb5, cb6 = st.columns(3)
    with cb4:
        speed = st.slider("재생 간격 (초/장)", 0.1, 3.0, 0.6, 0.1,
                          help="시연용 설정입니다. 이미지 한 장을 몇 초 동안 보여준 뒤 다음 이미지로 넘길지 정합니다. "
                               "실제 촬영 간격(약 21초)과는 무관합니다.")
    with cb5:
        manual_idx = st.slider("재생 위치", 0, N - 1, st.session_state.idx,
                               help="전체 이미지 중 현재 몇 번째 이미지를 검사 중인지 나타냅니다. 끌어서 이동할 수 있습니다.")
        if manual_idx != st.session_state.idx:
            st.session_state.idx = manual_idx
    with cb6:
        st.slider("검출 신뢰도 임계값", 0.05, 0.90, 0.25, 0.05, key="conf_th",
                  help="불량 클래스 박스의 신뢰도가 이 값 이상이면 불량으로 판정합니다. "
                       "낮추면 미검출이 줄고 오검출이 늘어납니다.")

st.write("")

# --------------------------------------------------------------------------------------
# 현재 검사 건
# --------------------------------------------------------------------------------------
idx = min(st.session_state.idx, N - 1)
row = df.iloc[idx]
hist = df.iloc[: idx + 1]
win = df.iloc[max(0, idx - WINDOW + 1): idx + 1]
is_bad = row["pred"] == "불량"
ucl_rate = float(row["ucl_rate"])
single_ucl, ewma_ucl = baseline["single_point_ucl"], baseline["ewma_ucl"]

# --------------------------------------------------------------------------------------
# 판정 배너 (이미지 판정 / 공정 경보)
# --------------------------------------------------------------------------------------
if is_bad:
    st.markdown(
        f"<div class='verdict-bad'><div class='banner-tag'>이미지 판정</div>"
        f"<div class='t1'>불량 판정 / {row['top_def']}</div>"
        f"<div class='t2'>불량 박스 {int(row['n_def'])}개 ({row['kinds']}) / 최고 신뢰도 {row['max_def']:.2f}</div></div>",
        unsafe_allow_html=True)
else:
    st.markdown(
        f"<div class='verdict-good'><div class='banner-tag'>이미지 판정</div>"
        f"<div class='t1'>정상 판정</div>"
        f"<div class='t2'>신뢰도 {conf_th:.2f} 이상의 불량 박스가 검출되지 않았습니다.</div></div>",
        unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------------------
# KPI 행 (검사 건)
# --------------------------------------------------------------------------------------
k1, k2, k3 = st.columns(3)
k1.markdown(
    f"<div class='kpi-card'><div class='kpi-label'>검사 일시</div>"
    f"<div class='kpi-value status-info'>{row['timestamp'].strftime('%H:%M:%S')}</div>"
    f"<div class='kpi-sub'>{row['timestamp'].strftime('%Y-%m-%d')} / {idx + 1} of {N}</div></div>",
    unsafe_allow_html=True)
k2.markdown(
    f"<div class='kpi-card'><div class='kpi-label'>판정</div>"
    f"<div class='kpi-value {'status-critical' if is_bad else 'status-good'}'>{row['pred']}</div>"
    f"<div class='kpi-sub'>{'불량 박스 ' + str(int(row['n_def'])) + '개' if is_bad else '불량 박스 없음'}</div></div>",
    unsafe_allow_html=True)
k3.markdown(
    f"<div class='kpi-card'><div class='kpi-label'>불량 유형</div>"
    f"<div class='kpi-value {'status-critical' if is_bad else 'status-good'}'>{row['top_def'] if is_bad else '-'}</div>"
    f"<div class='kpi-sub'>{'최고 신뢰도 ' + format(row['max_def'], '.2f') if is_bad else '해당 없음'}</div></div>",
    unsafe_allow_html=True)

st.write("")

# --------------------------------------------------------------------------------------
# 검사 이미지 / 판정 상세
# --------------------------------------------------------------------------------------
AXIS_STYLE = dict(gridcolor="#e1e0d9", zerolinecolor="#e1e0d9", linecolor="#c3c2b7")
GRAY, RED, BLUE, AMBER, PURPLE = "#c3c2b7", "#d03b3b", "#2a78d6", "#9a6700", "#4a3aa7"
IMG_W = 320     # 검사 이미지 표시 폭(px)
DET_H = 175     # 검출 내역 표 높이(px, 고정)
DIST_H = 150    # 불량 유형 분포 영역 높이(px, 고정)
LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0b0b0b",
              margin=dict(l=10, r=10, t=10, b=10))

col_img, col_info = st.columns([1, 1.8])

with col_img:
    st.markdown("#### 검사 이미지")
    detector, det_err = load_detector()
    if detector is None:
        st.error(f"모델 실행에 필요한 패키지가 없습니다 ({det_err}). 대시보드를 실행하는 같은 가상환경에서 "
                 "`python -m pip install -r requirements.txt` 를 실행한 뒤 다시 시작하세요.", icon=None)
        st.image(str(IMG_DIR / row["filename"]), width=IMG_W)
        boxes = []
    else:
        annotated, boxes = run_detection(row, conf_th)
        st.image(annotated, width=IMG_W)

with col_info:
    st.markdown("#### 검출 내역")
    if boxes:
        box_df = pd.DataFrame(
            [{"구분": "불량" if n.startswith("불량_") else "정상",
              "유형": n.replace("불량_", ""), "신뢰도": round(c, 3)} for n, c in boxes]
        )
        st.dataframe(box_df, use_container_width=True, hide_index=True, height=DET_H)
    else:
        st.dataframe(pd.DataFrame({"구분": [], "유형": [], "신뢰도": []}), use_container_width=True, hide_index=True, height=DET_H)

    # 현재 이미지의 불량 유형 분포 (정상 이미지에서는 비워 두되 영역 높이는 고정)
    type_counts = Counter(n.replace("불량_", "") for n, c in boxes if n.startswith("불량_"))
    with st.container(height=DIST_H, border=False):
        dist_slot = st.empty()          # 재생 중 이전 이미지의 요소가 남지 않도록 항상 같은 자리를 비우거나 채운다
        if type_counts:
            with dist_slot.container():
                st.markdown("#### 불량 유형 분포")
                tc = pd.Series(type_counts).sort_values()
                fig_t = go.Figure(go.Bar(x=tc.values, y=tc.index, orientation="h", marker_color=RED,
                                         hovertemplate="%{y}: %{x}개<extra></extra>"))
                fig_t.update_layout(height=DIST_H - 50, **LAYOUT,
                                    xaxis={**AXIS_STYLE, "title": "박스 수", "dtick": 1}, yaxis=AXIS_STYLE)
                st.plotly_chart(fig_t, use_container_width=True)
        else:
            dist_slot.empty()

st.markdown("#### 측정 환경")
st.caption("이 이미지와 함께 모델에 입력된 센서값(5초 구간 평균)과 기준선 대비 편차")
s_rows = {"구분": ["측정값", "기준 평균", "편차(σ)", "상태"]}
for i, (col_key, label) in enumerate(SENSOR_LABELS):
    mu, sigma = baseline["mu"][i], baseline["sigma"][i]
    z = (row[col_key] - mu) / sigma
    s_rows[label] = [f"{row[col_key]:.3f}", f"{mu:.3f}", f"{z:+.2f}", "기준 이탈" if abs(z) > 3 else "정상 범위"]
st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------
# 불량률 관리도 / 유형별 누적 발생
# --------------------------------------------------------------------------------------
st.write("")
c_rate, c_type = st.columns([1.5, 1])
xw = np.arange(win.index[0] + 1, win.index[-1] + 2)       # 검사 순번 (1부터)
hover_t = win["timestamp"].dt.strftime("%m-%d %H:%M:%S")

with c_rate:
    st.markdown("#### 불량률 관리도")
    st.caption(f"최근 {len(win)}건, 직전 {win_n}건 이동 불량률")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xw, y=win["rate"], mode="lines+markers", name="구간 불량률",
                             line=dict(color=BLUE, width=2.2), marker=dict(size=4), customdata=hover_t,
                             hovertemplate="검사 %{x} (%{customdata})<br>불량률 %{y:.0%}<extra></extra>"))
    over = win[win["rate_level"] == 2]
    if len(over):
        fig.add_trace(go.Scatter(x=over.index + 1, y=over["rate"], mode="markers", name="관리상한 초과",
                                 marker=dict(symbol="triangle-down", size=10, color=RED)))
    fig.add_hline(y=ucl_rate, line_dash="dash", line_color=RED, annotation_text="관리상한", annotation_position="top left")
    fig.add_hline(y=p0, line_dash="dot", line_color=AMBER, annotation_text="허용 불량률", annotation_position="top left")
    if not np.isnan(row["rate"]):
        fig.add_trace(go.Scatter(x=[idx + 1], y=[row["rate"]], mode="markers", name="현재",
                                 marker=dict(size=13, color=BLUE, line=dict(width=2, color="#0b0b0b"))))
    fig.update_layout(height=290, **LAYOUT, legend=dict(orientation="h", y=1.18),
                      xaxis={**AXIS_STYLE, "title": "검사 순번"},
                      yaxis={**AXIS_STYLE, "title": "불량률", "tickformat": ".0%", "range": [0, 1.05]})
    st.plotly_chart(fig, use_container_width=True)

with c_type:
    st.markdown("#### 불량 유형별 누적 발생")
    st.caption("현재 위치까지 누적, 불량 판정 이미지의 대표 유형 기준")
    bad_hist = hist[hist["pred"] == "불량"]
    if len(bad_hist):
        order = bad_hist["top_def"].value_counts().index[:5]
        fig_c = go.Figure()
        xs_all = np.arange(1, len(hist) + 1)
        for t in order:
            cum = (hist["top_def"] == t).cumsum().to_numpy()
            fig_c.add_trace(go.Scatter(x=xs_all, y=cum, mode="lines", name=t, line=dict(width=2)))
        fig_c.update_layout(height=290, **LAYOUT, legend=dict(orientation="h", y=1.18),
                            xaxis={**AXIS_STYLE, "title": "검사 순번"}, yaxis={**AXIS_STYLE, "title": "누적 건수"})
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.caption("아직 불량 판정이 없습니다.")

# --------------------------------------------------------------------------------------
# 센서 이상도 / 채널별 기여도
# --------------------------------------------------------------------------------------
c_sens, c_share = st.columns([1.5, 1])
with c_sens:
    st.markdown("#### 센서 이상도")
    st.caption("마할라노비스 거리와 EWMA(λ=0.2) 관리도")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xw, y=win["D"], mode="lines+markers", name="거리 D", customdata=hover_t,
                              line=dict(color="#898781", width=1.5), marker=dict(size=4),
                              hovertemplate="검사 %{x} (%{customdata})<br>D %{y:.2f}<extra></extra>"))
    fig2.add_trace(go.Scatter(x=xw, y=win["ewma"], mode="lines", name="EWMA",
                              line=dict(color=BLUE, width=2.5)))
    fig2.add_hline(y=ewma_ucl, line_dash="dash", line_color=PURPLE, annotation_text="EWMA 관리상한", annotation_position="bottom left")
    fig2.add_hline(y=single_ucl, line_dash="dot", line_color=AMBER, annotation_text="단일점 한계", annotation_position="top left")
    s_al = win[win["sensor_level"] == 2]
    if len(s_al):
        fig2.add_trace(go.Scatter(x=s_al.index + 1, y=s_al["D"], mode="markers", name="단일점 이탈",
                                  marker=dict(symbol="triangle-down", size=10, color=RED)))
    fig2.add_trace(go.Scatter(x=[idx + 1], y=[row["D"]], mode="markers", name="현재",
                              marker=dict(size=13, color=BLUE, line=dict(width=2, color="#0b0b0b"))))
    fig2.update_layout(height=270, **LAYOUT, legend=dict(orientation="h", y=1.2),
                       xaxis={**AXIS_STYLE, "title": "검사 순번"}, yaxis={**AXIS_STYLE, "title": "마할라노비스 거리"})
    st.plotly_chart(fig2, use_container_width=True)

with c_share:
    st.markdown("#### 채널별 기여도")
    st.caption("현재 이미지의 센서 이상 기여 비율")
    share = channel_share(row, baseline)
    names = [lab for _, lab in SENSOR_LABELS]
    fig_s = go.Figure(go.Bar(x=share[::-1], y=names[::-1], orientation="h",
                             marker_color=[RED if v == share.max() else GRAY for v in share[::-1]],
                             hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
    fig_s.update_layout(height=270, **LAYOUT, xaxis={**AXIS_STYLE, "title": "기여도(%)", "range": [0, 100]}, yaxis=AXIS_STYLE)
    st.plotly_chart(fig_s, use_container_width=True)

# --------------------------------------------------------------------------------------
# 센서 추이 (5채널 스몰멀티플)
# --------------------------------------------------------------------------------------
st.markdown("#### 센서 추이")
st.caption(f"최근 {len(win)}건, 음영은 기준선 평균 ±3σ, 붉은 점은 관리한계 이탈")
cols = st.columns(5)
for i, (col_key, label) in enumerate(SENSOR_LABELS):
    mu = baseline["mu"][i]
    sigma = baseline["sigma"][i]
    with cols[i]:
        fig_m = go.Figure()
        fig_m.add_hrect(y0=mu - 3 * sigma, y1=mu + 3 * sigma, fillcolor="rgba(10,163,12,0.08)", line_width=0)
        fig_m.add_trace(go.Scatter(x=xw, y=win[col_key], mode="lines", line=dict(color=BLUE, width=1.5), customdata=hover_t,
                                   hovertemplate="검사 %{x} (%{customdata})<br>%{y:.3f}<extra></extra>"))
        out = win[(win[col_key] - mu).abs() > 3 * sigma]
        if len(out):
            fig_m.add_trace(go.Scatter(x=out.index + 1, y=out[col_key], mode="markers",
                                       marker=dict(color=RED, size=5), hoverinfo="skip"))
        fig_m.add_trace(go.Scatter(x=[idx + 1], y=[row[col_key]], mode="markers",
                                   marker=dict(color="#0b0b0b", size=8), hoverinfo="skip"))
        fig_m.update_layout(height=170, margin=dict(l=5, r=5, t=26, b=5), title=dict(text=label, font=dict(size=12)),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0b0b0b",
                            showlegend=False, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE)
        st.plotly_chart(fig_m, use_container_width=True)

# --------------------------------------------------------------------------------------
# 경보 이력 / 검사 이력
# --------------------------------------------------------------------------------------
st.write("")
st.markdown("#### 경보 이력")
st.caption("경보 수준이 올라간 시점, 최근 10건")
events = build_events(hist, p0)
if len(events):
    ev = events.head(10).copy()
    ev["검사 일시"] = ev["time"].dt.strftime("%m-%d %H:%M:%S")
    ev["검사 순번"] = ev["no"]
    st.dataframe(ev[["검사 일시", "검사 순번", "구분", "수준", "내용"]], use_container_width=True, hide_index=True)
else:
    st.caption("아직 발생한 경보가 없습니다.")

st.markdown("#### 검사 이력")
st.caption("최근 검사 10건")
recent = hist.sort_values("timestamp", ascending=False).head(10)
show = recent[["timestamp", "pred", "top_def", "max_def", "label", "match"]].copy()
show["timestamp"] = show["timestamp"].dt.strftime("%H:%M:%S")
show["top_def"] = show["top_def"].replace("", "-")
show["max_def"] = show["max_def"].map(lambda v: f"{v:.2f}" if v > 0 else "-")
show.columns = ["검사 시각", "판정", "불량 유형", "최고 신뢰도", "정답 라벨", "정답 대조"]
st.dataframe(show, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------
# 상세 분석 (접이식)
# --------------------------------------------------------------------------------------
with st.expander("누적 판정 성과"):
    truth = hist["label"] == "불량"
    pred = hist["pred"] == "불량"
    tp = int((truth & pred).sum()); fn = int((truth & ~pred).sum())
    fp = int((~truth & pred).sum()); tn = int((~truth & ~pred).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("불량 검출률", f"{tp / max(tp + fn, 1):.1%}", f"{tp} / {tp + fn}")
    c2.metric("정상 오검출률", f"{fp / max(fp + tn, 1):.1%}", f"{fp} / {fp + tn}", delta_color="inverse")
    c3.metric("판정 정확도", f"{(tp + tn) / max(len(hist), 1):.1%}", f"{tp + tn} / {len(hist)}", delta_color="off")
    c4.metric("검사 건수", f"{len(hist)} / {N}")
    st.caption("현재 위치까지의 누적값이며 정답 라벨과 대조한 결과입니다.")

with st.expander("검사기준별 판정 유형"):
    st.caption("파일명의 검사기준 코드(정상/불량, 납부족/납볼)별로 모델이 판정한 유형의 분포입니다. 현재 위치까지의 누적값입니다.")
    rows = []
    for (label, grp), sub in hist.groupby(["label", "type_kr"]):
        kinds = sub["top_def"].replace("", "정상(불량 미검출)").value_counts()
        rows.append({"정답 라벨": label, "검사기준": grp.replace(" 검사군", ""), "건수": len(sub),
                     "판정 유형 분포": ", ".join(f"{k} {v}" for k, v in kinds.items())})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------
# 자동 재생 루프
# --------------------------------------------------------------------------------------
if st.session_state.running:
    if st.session_state.idx >= N - 1:
        st.session_state.running = False      # 끝까지 재생하면 정지
        st.rerun()
    time.sleep(speed)
    st.session_state.idx = min(st.session_state.idx + 1, N - 1)
    st.rerun()
