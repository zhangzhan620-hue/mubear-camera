import cv2
import numpy as np
import onnxruntime as ort


class OnnxModel:
    def __init__(self, path: str, imgsz: int):
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.imgsz = imgsz

    def preprocess(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        gain = self.imgsz / max(h, w)
        nw, nh = int(round(w * gain)), int(round(h * gain))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - nw) // 2
        pad_y = (self.imgsz - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, gain, pad_x, pad_y

    def forward(self, frame: np.ndarray):
        blob, gain, pad_x, pad_y = self.preprocess(frame)
        out = self.sess.run(None, {self.input_name: blob})[0]
        return out[0].T, gain, pad_x, pad_y


class DetectModel(OnnxModel):
    def predict_persons(self, frame: np.ndarray, conf_thres: float) -> list:
        pred, gain, pad_x, pad_y = self.forward(frame)
        boxes_xy = pred[:, :4].copy()
        boxes_xy[:, 0] = (boxes_xy[:, 0] - pad_x) / gain
        boxes_xy[:, 1] = (boxes_xy[:, 1] - pad_y) / gain
        boxes_xy[:, 2] /= gain
        boxes_xy[:, 3] /= gain
        cls_scores = pred[:, 4:]
        cls_ids = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(len(pred)), cls_ids]

        mask = (scores >= conf_thres) & (cls_ids == 0)
        boxes_xy = boxes_xy[mask]
        scores = scores[mask]
        if len(boxes_xy) == 0:
            return []

        xyxy = np.empty_like(boxes_xy)
        xyxy[:, 0] = boxes_xy[:, 0] - boxes_xy[:, 2] / 2
        xyxy[:, 1] = boxes_xy[:, 1] - boxes_xy[:, 3] / 2
        xyxy[:, 2] = boxes_xy[:, 0] + boxes_xy[:, 2] / 2
        xyxy[:, 3] = boxes_xy[:, 1] + boxes_xy[:, 3] / 2

        keep = cv2.dnn.NMSBoxes(
            xyxy.tolist(), scores.tolist(), conf_thres, 0.5
        )
        keep = np.array(keep).flatten() if len(keep) else []
        return [(xyxy[i], float(scores[i])) for i in keep]


class PoseModel(OnnxModel):
    def predict(self, frame: np.ndarray, conf_thres: float):
        pred, gain, pad_x, pad_y = self.forward(frame)
        scores = pred[:, 4]
        mask = scores >= conf_thres
        pred = pred[mask]
        if len(pred) == 0:
            return [], [], []

        boxes_xy = pred[:, :4].copy()
        boxes_xy[:, 0] = (boxes_xy[:, 0] - pad_x) / gain
        boxes_xy[:, 1] = (boxes_xy[:, 1] - pad_y) / gain
        boxes_xy[:, 2] /= gain
        boxes_xy[:, 3] /= gain

        xyxy = np.empty_like(boxes_xy)
        xyxy[:, 0] = boxes_xy[:, 0] - boxes_xy[:, 2] / 2
        xyxy[:, 1] = boxes_xy[:, 1] - boxes_xy[:, 3] / 2
        xyxy[:, 2] = boxes_xy[:, 0] + boxes_xy[:, 2] / 2
        xyxy[:, 3] = boxes_xy[:, 1] + boxes_xy[:, 3] / 2

        kps = pred[:, 5:].reshape(len(pred), 17, 3).copy()
        kps[:, :, 0] = (kps[:, :, 0] - pad_x) / gain
        kps[:, :, 1] = (kps[:, :, 1] - pad_y) / gain
        return list(xyxy), [k[..., :2] for k in kps], [k[..., 2] for k in kps]
