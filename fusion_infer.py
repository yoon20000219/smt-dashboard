# -*- coding: utf-8 -*-
"""
최종 멀티모달 모델(YOLO11n + FiLM, 이미지 + 센서 2입력, ONNX) 추론 모듈.

공유 패키지(SMT_final_model_share)의 code/predict.py 와 같은 전처리·후처리를 따르되,
torch / ultralytics 없이 onnxruntime + numpy + opencv + Pillow 만으로 동작하도록 정리했다.

  입력 images : (1, 3, 640, 640) float32, RGB, 0~1, letterbox(회색 114)
  입력 sensor : (1, 35) float32, 요약통계 35개를 학습 평균·표준편차로 표준화한 값
  출력 output0: (1, 21, 8400)  행 0~3 = cx, cy, w, h (640 기준), 행 4~20 = 17개 클래스 점수
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMGSZ = 640
DEFECT_NAMES = ["미납", "납부족", "납쇼트", "납볼", "납좌표밀림", "납형성불량", "냉납", "밀림", "쇼트",
                "오삽", "미삽", "역삽", "뒤집힘", "일어섬", "납금감/핀홀", "납고드름"]
CLASS_NAMES = [f"불량_{n}" for n in DEFECT_NAMES] + ["정상"]

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    str(Path.home() / ".config" / "Ultralytics" / "Arial.Unicode.ttf"),
    str(Path.home() / "AppData" / "Roaming" / "Ultralytics" / "Arial.Unicode.ttf"),
]


def _load_font(size: int):
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


class FusionDetector:
    def __init__(self, onnx_path: Path, scaler_path: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        sc = json.load(open(scaler_path, encoding="utf-8"))["full35"]
        self.cols = sc["cols"]
        self.mu = np.array([sc["mean"][c] for c in self.cols], dtype=np.float64)
        self.sd = np.array([sc["std"][c] for c in self.cols], dtype=np.float64)

    # ---- 센서 ----
    def normalize_sensor(self, feat35) -> np.ndarray:
        """요약통계 35개(scaler cols 순서) -> 표준화 (1, 35) float32"""
        v = np.asarray(feat35, dtype=np.float64).reshape(1, -1)
        return ((v - self.mu) / self.sd).astype(np.float32)

    # ---- 이미지 ----
    @staticmethod
    def preprocess(im_bgr: np.ndarray):
        h, w = im_bgr.shape[:2]
        r = IMGSZ / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        im = cv2.resize(im_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else im_bgr
        canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
        top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
        canvas[top:top + nh, left:left + nw] = im
        x = canvas[:, :, ::-1].transpose(2, 0, 1)
        return np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0, (r, left, top)

    # ---- 후처리 (클래스별 NMS) ----
    @staticmethod
    def _nms(boxes, scores, iou_th):
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (a_i + a_r - inter + 1e-9)
            order = rest[iou <= iou_th]
        return keep

    def postprocess(self, pred, meta, orig_hw, conf, iou, max_det=300):
        p = pred[0].T                                   # (8400, 21)
        scores_all = p[:, 4:]
        cls = scores_all.argmax(1)
        score = scores_all.max(1)
        m = score >= conf
        p, cls, score = p[m], cls[m], score[m]
        if not len(score):
            return []
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        keep = self._nms(xyxy + cls[:, None] * 4096.0, score, iou)[:max_det]   # 클래스별 NMS
        r, left, top = meta
        H, W = orig_hw
        out = []
        for i in keep:
            x1, y1, x2, y2 = xyxy[i]
            box = [(x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r]
            box = [float(np.clip(box[0], 0, W)), float(np.clip(box[1], 0, H)),
                   float(np.clip(box[2], 0, W)), float(np.clip(box[3], 0, H))]
            out.append({"class_id": int(cls[i]), "class_name": CLASS_NAMES[int(cls[i])],
                        "confidence": float(score[i]), "box_xyxy": box})
        return out

    # ---- 추론 ----
    def predict(self, image_path, feat35, conf=0.25, iou=0.7):
        im = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)   # 한글 경로 대응
        if im is None:
            raise FileNotFoundError(image_path)
        x, meta = self.preprocess(im)
        s = self.normalize_sensor(feat35)
        pred = self.sess.run(None, {"images": x, "sensor": s})[0]
        return im, self.postprocess(pred, meta, im.shape[:2], conf, iou)


def draw_detections(im_bgr: np.ndarray, dets) -> np.ndarray:
    """검출 박스를 원본 이미지 위에 그려 RGB 배열로 반환 (불량 = 적색, 정상 = 녹색)."""
    img = Image.fromarray(im_bgr[:, :, ::-1])
    d = ImageDraw.Draw(img)
    font = _load_font(max(12, img.width // 36))
    for det in dets:
        defect = det["class_name"].startswith("불량_")
        color = (208, 59, 59) if defect else (12, 140, 12)
        x1, y1, x2, y2 = det["box_xyxy"]
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        tb = d.textbbox((x1, y1), label, font=font)
        th = tb[3] - tb[1]
        ty = max(0, y1 - th - 4)
        d.rectangle([x1, ty, x1 + (tb[2] - tb[0]) + 6, ty + th + 4], fill=color)
        d.text((x1 + 3, ty), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)
