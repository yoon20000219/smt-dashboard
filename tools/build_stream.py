# -*- coding: utf-8 -*-
"""
Validation 이미지 스트림 데이터 생성기 (실제 촬영 시각순 + 실제 센서 1:1 매칭 + 모델 예측)

하는 일
  1. 공유 패키지 splits/*.txt 와 A_센서데이터_원본.csv 로 "이미지 파일 <-> 센서 이벤트" 매핑을 복원한다.
       규칙  : (공정, 정상/불량, 검사유형) 그룹 안에서 파일명을 촬영 카운터순으로 정렬한 순번 = A 데이터 같은 그룹의 event_id 순번
       검증  : 그룹별 Training/Validation 배치 패턴이 A 의 split 열과 일치해야 채택 (유일 후보일 때만),
               공유 패키지 sample 20장 중 매핑 가능한 19장은 센서 원본 25개 값이 소수점까지 일치.
  2. 대상 Validation 이미지를 실제 촬영 시각순으로 정렬하고 센서 25개 -> 요약통계 35개(공식 계산식) 변환.
  3. 최종 멀티모달 모델(ONNX)로 전체 이미지를 예측해 data/predictions.json 저장 (conf>=0.05 전부).
  4. 센서 5채널 평균으로 마할라노비스 거리 + EWMA 관리도 계산 (기준선은 data/spc_baseline.json).
  5. data/stream_sequence.csv, data/stream_meta.json 저장.

실행 예
  python tools/build_stream.py --zips <Validation/01.원천데이터> --pkg <SMT_final_model_share> --sensor_csv <A_센서데이터_원본.csv>
"""
import argparse, json, re, sys, zipfile, io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fusion_infer import FusionDetector  # noqa: E402

PAT = re.compile(r"(PR|SD)_(NOR|DEF)_(\w+?)_(\w)_(\d{8})-(\d{6})_(\d+)\.")
TYPE_KR = {"NB": "납부족", "PB": "납볼"}
CH_KEYS = ["temp", "hum", "vib", "acc", "noise"]
CH_FULL = ["temperature", "humidity", "vibration", "acceleration", "noise"]
CH_KO = ["온도", "습도", "진동", "가속도", "소음"]
INSPECT = {"온도": "리플로우 오븐 점검", "습도": "자재 보관 환경 점검", "진동": "실장 설비(마운터) 점검",
           "가속도": "컨베이어 이송계열 점검", "소음": "설비 가동 소음원 점검"}


def feats_full35(seq):                    # seq (N,5스텝,5채널) -> (N,35) 공식 predict.py 와 같은 계산식
    t = np.arange(5, dtype=np.float64); t -= t.mean()
    out = []
    for j in range(5):
        v = seq[:, :, j]; mean = v.mean(1); mx = v.max(1); mn = v.min(1); sd = v.std(1)
        slope = ((v - mean[:, None]) * t).sum(1) / (t * t).sum()
        out += [mean, mx, mn, sd, slope, mx - mn, np.abs(np.diff(v, axis=1)).mean(1)]
    return np.stack(out, 1)


def load_split_files(pkg):
    rows = []
    for sp in ("train", "test", "val"):
        for l in (pkg / "splits" / f"{sp}_files.txt").read_text(encoding="utf-8-sig").splitlines():
            l = l.strip()
            if l:
                m = PAT.match(l)
                rows.append(dict(f=l, sp=sp, proc=m[1], lab=m[2], typ=m[3], cnt=int(m[7])))
    return pd.DataFrame(rows)


def build_event_map(F, A):
    A = A.copy(); A["isval"] = A.split == "Validation"
    ag = {k: g.sort_values("event_id") for k, g in A.groupby(["process_stage", "y", "defect_type"])}
    mp = {}
    for fk, g in F.groupby(["proc", "lab", "typ"]):
        g = g.sort_values("cnt"); fs = (g.sp == "val").to_numpy()
        pst = "사전공정" if fk[0] == "PR" else "납땜공정"; y = 0 if fk[1] == "NOR" else 1
        cands = [k for k, a in ag.items() if len(a) == len(g) and k[0] == pst and k[1] == y and (a.isval.to_numpy() == fs).all()]
        if len(cands) == 1:
            mp.update(dict(zip(g.f, ag[cands[0]].event_id)))
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True); ap.add_argument("--pkg", required=True); ap.add_argument("--sensor_csv", required=True)
    ap.add_argument("--img_dir", default=str(ROOT / "data" / "stream_images"))
    a = ap.parse_args()
    pkg = Path(a.pkg); img_dir = Path(a.img_dir); img_dir.mkdir(parents=True, exist_ok=True)

    # 1) 이미지 풀기 + 메타
    rows = []
    for z in sorted(Path(a.zips).glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                m = PAT.match(Path(n).name)
                if not m:
                    continue
                (img_dir / Path(n).name).write_bytes(zf.read(n))
                rows.append(dict(filename=Path(n).name, proc=m[1], lab=m[2], typ=m[3],
                                 timestamp=pd.to_datetime(m[5] + m[6], format="%Y%m%d%H%M%S"), counter=int(m[7])))
    S = pd.DataFrame(rows).sort_values(["timestamp", "counter"]).reset_index(drop=True)

    # 2) 센서 매핑
    A = pd.read_csv(a.sensor_csv, encoding="utf-8-sig")
    mp = build_event_map(load_split_files(pkg), A)
    S["event_id"] = S.filename.map(mp)
    assert S.event_id.notna().all(), "매핑되지 않은 이미지가 있습니다"
    Ai = A.set_index("event_id")
    acols = [f"{c}_t{t}" for c in CH_KEYS for t in range(5)]
    assert (Ai.loc[S.event_id, "split"] == "Validation").all()
    assert ((S.lab == "DEF").astype(int).to_numpy() == Ai.loc[S.event_id, "y"].to_numpy()).all()
    raw = Ai.loc[S.event_id, acols].to_numpy(float)
    for j, c in enumerate(CH_FULL):
        for t in range(5):
            S[f"{c}_{t + 1}s"] = raw[:, j * 5 + t]
    seq = raw.reshape(len(S), 5, 5).transpose(0, 2, 1)             # (N, 5스텝, 5채널)
    F35 = feats_full35(seq)

    # 3) 모델 예측
    det = FusionDetector(ROOT / "data/model/model.onnx", ROOT / "data/model/sensor_scaler.json")
    feat_names = ["f_" + c for c in det.cols]
    preds = {}
    for i, r in S.iterrows():
        _, d = det.predict(img_dir / r.filename, F35[i], conf=0.05, iou=0.7)
        preds[r.filename] = [{"class_id": x["class_id"], "class_name": x["class_name"], "confidence": round(x["confidence"], 4),
                              "box_xyxy": [round(b, 1) for b in x["box_xyxy"]]} for x in d]
    (ROOT / "data" / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")

    # 4) 센서 SPC (마할라노비스 + EWMA)
    bl = json.loads((ROOT / "data" / "spc_baseline.json").read_text(encoding="utf-8"))
    mu = np.array(bl["mu"]); sd = np.array(bl["sigma"]); inv = np.linalg.inv(np.array(bl["cov"]))
    means = F35[:, [0, 7, 14, 21, 28]]                              # 채널별 평균 (온도, 습도, 진동, 가속도, 소음)
    X = means - mu
    D = np.sqrt(np.einsum("ij,jk,ik->i", X, inv, X))
    lam = bl["lambda"]; ew = np.zeros(len(D)); prev = bl["D_mean"]
    for i, d in enumerate(D):
        prev = lam * d + (1 - lam) * prev; ew[i] = prev
    single = D > bl["single_point_ucl"]; ewm = ew > bl["ewma_ucl"]
    z2 = ((means - mu) / sd) ** 2; contrib = z2 / z2.sum(1, keepdims=True) * 100
    order = np.argsort(-contrib, 1)
    S["process"] = S.proc.map({"PR": "사전공정", "SD": "납땜공정"})
    S["label"] = S.lab.map({"NOR": "정상", "DEF": "불량"})
    S["type_kr"] = S.typ.map(TYPE_KR) + " 검사군"
    S["path"] = "stream_images/" + S.filename
    for j, k in enumerate(["temp_mean", "hum_mean", "vib_mean", "acc_mean", "noise_mean"]):
        S[k] = means[:, j]
    S["D"] = D; S["ewma"] = ew
    S["sensor_alert"] = single | ewm
    S["sensor_alert_type"] = np.where(single & ewm, "EWMA+단일점", np.where(single, "단일점 이탈", np.where(ewm, "EWMA 상한 이탈", "")))
    S["top_channel_1"] = [CH_KO[o[0]] for o in order]; S["top_pct_1"] = [contrib[i, o[0]] for i, o in enumerate(order)]
    S["top_channel_2"] = [CH_KO[o[1]] for o in order]; S["top_pct_2"] = [contrib[i, o[1]] for i, o in enumerate(order)]
    S["inspect_target_sensor"] = [INSPECT[c] if al else "" for c, al in zip(S.top_channel_1, S.sensor_alert)]
    for j, n in enumerate(feat_names):
        S[n] = F35[:, j]
    keep = ["filename", "path", "timestamp", "process", "label", "type_kr", "counter", "event_id"] + \
           [f"{c}_{t + 1}s" for c in CH_FULL for t in range(5)] + \
           ["temp_mean", "hum_mean", "vib_mean", "acc_mean", "noise_mean", "D", "ewma", "sensor_alert", "sensor_alert_type",
            "top_channel_1", "top_pct_1", "top_channel_2", "top_pct_2", "inspect_target_sensor"] + feat_names
    S[keep].to_csv(ROOT / "data" / "stream_sequence.csv", index=False, encoding="utf-8")

    # 5) 스트림 특성 요약 (대시보드 '데이터 진단' 표시용)
    gap = S.groupby(["lab", "typ"]).timestamp.diff().dt.total_seconds().dropna()

    # 연속 프레임 유사도(64x64 정규화 상관) vs 무작위 쌍, 센서 lag-1 자기상관, 모델 이미지 단위 판정률
    import cv2

    def _f(fn):
        im = cv2.resize(cv2.imread(str(img_dir / fn), cv2.IMREAD_GRAYSCALE), (64, 64)).astype(np.float32)
        return ((im - im.mean()) / (im.std() + 1e-6)).ravel()

    rng = np.random.default_rng(0); sim = {}; ac = {}; rate = {}
    for (l, t), g in S.groupby(["lab", "typ"]):
        Fm = np.stack([_f(f) for f in g.filename]); n = len(Fm)
        cons = float(np.mean([Fm[i] @ Fm[i + 1] / Fm.shape[1] for i in range(n - 1)]))
        rnd = float(np.mean([Fm[i] @ Fm[j] / Fm.shape[1] for i, j in rng.integers(0, n, (400, 2)) if i != j]))
        sim[f"{l}_{t}"] = {"consecutive": round(cons, 3), "random": round(rnd, 3)}
        vals = [np.corrcoef(g[c].to_numpy()[:-1], g[c].to_numpy()[1:])[0, 1]
                for c in ["temp_mean", "hum_mean", "vib_mean", "acc_mean", "noise_mean"]]
        ac[f"{l}_{t}"] = round(float(np.mean(np.abs(vals))), 3)
        flag = [any(d["class_name"].startswith("불량_") and d["confidence"] >= 0.25 for d in preds[f]) for f in g.filename]
        rate[f"{l}_{t}"] = round(float(np.mean(flag)), 3)

    meta = {"n": int(len(S)), "t0": str(S.timestamp.min()), "t1": str(S.timestamp.max()),
            "sessions": {f"{l}_{t}": int(n) for (l, t), n in S.groupby(["lab", "typ"]).size().items()},
            "median_gap_sec": float(gap.median()),
            "mapping": "A_센서데이터_원본 event 순번 규칙 (split 배치 패턴 일치)",
            "frame_similarity": sim, "sensor_lag1_abs_autocorr": ac, "model_alert_rate_conf025": rate}
    (ROOT / "data" / "stream_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", len(S), "rows")


if __name__ == "__main__":
    main()
