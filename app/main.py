import time
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from . import capture, config, detector, qr, storage, uploader, voice

STATIC_DIR = Path(__file__).resolve().parent / "static"

_stage = {"name": "idle", "ts": 0.0}


def set_stage(name: str):
    _stage["name"] = name
    _stage["ts"] = time.time()

app = FastAPI(title="木熊智能摄像头")


@app.on_event("startup")
def startup():
    storage.init_db()
    if config.ENABLE_DETECTOR:
        detector.get_detector()
    uploader.start_cleanup_loop()


@app.get("/api/stage")
def api_stage():
    return dict(_stage)


@app.get("/api/health")
def api_health():
    from .face import get_eye_analyzer
    from .gesture import get_gesture_recognizer

    det = detector.get_detector()
    eye = get_eye_analyzer()
    gr = get_gesture_recognizer()
    return {
        "detector": {"available": det.available, "error": det.error},
        "pose": {"available": det.pose_available},
        "eye": {"available": eye.available, "error": eye.error},
        "gesture": {"available": gr.available, "error": gr.error},
    }


@app.get("/api/presence")
def api_presence():
    det = detector.get_detector()
    if not det.available:
        return {"persons": 0, "frontal": 0, "gesture": 0, "detector": False}
    frame = capture.get_camera().read_frame()
    result = det.quick_presence(frame)
    result["detector"] = True
    return result


@app.get("/logo.png")
def logo():
    path = config.BASE_DIR / "logo.png"
    if path.exists():
        return FileResponse(path, media_type="image/png")
    raise HTTPException(404, "no logo")


CN_NUM = "一二三四五六七八九十"


def face_issue_messages(analysis: dict) -> list:
    faces = analysis.get("faces", [])
    positions = [f["position"] for f in faces]
    use_ordinal = len(set(positions)) < len(positions)
    msgs = []
    for i, f in enumerate(faces):
        problems = []
        if not f["frontal"]:
            problems.append("请看镜头")
        elif not f["eyes_open"]:
            problems.append("请睁眼")
        if not problems:
            continue
        who = f"从左数第{CN_NUM[i]}位同学" if use_ordinal else f"{f['position']}的同学"
        msgs.append(who + "，".join(problems))
    return msgs


def shoot_once(cam, det, v):
    for word in ["3", "2", "1", "耶"]:
        set_stage(word)
        v.say(word, block=True)
        if word != "耶":
            time.sleep(0.4)
    set_stage("shooting")
    frames = cam.capture_burst()
    set_stage("analyzing")
    analyses = [det.analyze(f) for f in frames]
    scores = [det.score(a) for a in analyses]
    best_idx = scores.index(max(scores))
    return frames, analyses, scores, best_idx


@app.post("/capture")
def do_capture():
    cam = capture.get_camera()
    det = detector.get_detector()
    v = voice.get_voice()

    max_attempts = config.MAX_ATTEMPTS
    attempts = []
    for attempt in range(max_attempts):
        frames, analyses, scores, best_idx = shoot_once(cam, det, v)
        best = analyses[best_idx]
        counts = [a["person_count"] for a in analyses]
        count_unstable = len(set(counts)) > 1
        out_votes = sum(1 for a in analyses if not a["all_in_frame"])
        out_of_frame = out_votes >= (len(analyses) + 1) // 2
        issues = face_issue_messages(best)
        passed = best["person_count"] > 0 and not out_of_frame and not issues
        attempts.append({
            "passed": passed, "frames": frames, "analyses": analyses,
            "scores": scores, "best_idx": best_idx, "issues": issues,
            "out_of_frame": out_of_frame, "count_unstable": count_unstable,
        })
        if passed:
            break
        if attempt < max_attempts - 1:
            if out_of_frame:
                v.say("有人出画了，请往中间站一点", block=True)
            elif issues:
                v.say("，".join(issues[:3]), block=True)
            v.say("再来一张", block=True)

    best_attempt = max(attempts, key=lambda a: (a["passed"], a["scores"][a["best_idx"]]))
    best = best_attempt["analyses"][best_attempt["best_idx"]]
    retried = len(attempts) > 1
    count_unstable = best_attempt["count_unstable"]
    out_of_frame = best_attempt["out_of_frame"]
    issues = best_attempt["issues"]

    if not best_attempt["passed"]:
        if best["person_count"] == 0 and best["detector_available"]:
            v.say("没有拍到人，请站好一点再试")
        else:
            v.say("连续几次都没通过，" + "，".join(issues[:2]) + "，比个耶重新开始")
        return {
            "accepted": False,
            "session_id": None,
            "url": None,
            "public_url": None,
            "best_frame": best_attempt["best_idx"],
            "scores": best_attempt["scores"],
            "analysis": best,
            "retried": retried,
            "count_unstable": count_unstable,
            "issues": issues,
            "detector_error": det.error,
        }

    if retried:
        v.say("这次不错，拍好了")

    session_id = storage.save_session(
        best_attempt["frames"], best_attempt["analyses"],
        best_attempt["scores"], best_attempt["best_idx"], v.take_log(),
    )

    set_stage("uploading")
    public_url = None
    up = uploader.get_uploader()
    local_best = storage.best_photo_path(session_id)
    if up.available and local_best is not None:
        try:
            public_url = up.upload_photo(local_best, session_id)
            storage.set_public_url(session_id, public_url)
        except Exception:
            public_url = None
    set_stage("done")

    url = public_url or f"{config.base_url()}/photo/{session_id}"
    return {
        "session_id": session_id,
        "url": url,
        "public_url": public_url,
        "best_frame": best_attempt["best_idx"],
        "scores": best_attempt["scores"],
        "analysis": best,
        "retried": retried,
        "count_unstable": count_unstable,
        "detector_error": det.error,
    }


@app.get("/photo/{session_id}", response_class=HTMLResponse)
def photo_page(session_id: str):
    info = storage.get_session(session_id)
    if not info:
        raise HTTPException(404, "session not found")
    s = info["session"]
    img_url = f"/photo/{session_id}/download"
    qr_url = f"/photo/{session_id}/qr.png"
    return f"""<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>你的照片</title>
<style>
body{{font-family:sans-serif;text-align:center;background:#111;color:#eee;margin:0;padding:24px}}
img.photo{{max-width:100%;border-radius:12px}}
a.btn{{display:inline-block;margin-top:16px;padding:12px 32px;background:#4caf50;color:#fff;
border-radius:8px;text-decoration:none;font-size:18px}}
.meta{{color:#888;font-size:13px;margin-top:12px}}
</style></head>
<body>
<h2>木熊相机</h2>
<img class="photo" src="{img_url}" alt="photo">
<br>
<a class="btn" href="{img_url}" download>下载照片</a>
<div class="meta">拍摄时间 {s['created_at']} · 检测到 {s['person_count']} 人 ·
{'全部在画面内' if s['all_in_frame'] else '有人出画'} · 得分 {s['score']}</div>
<div style="margin-top:24px"><img src="{qr_url}" width="140" alt="qr"></div>
</body></html>"""


@app.get("/photo/{session_id}/download")
def photo_download(session_id: str):
    path = storage.best_photo_path(session_id)
    if path is None:
        raise HTTPException(404, "photo not found")
    return FileResponse(path, media_type="image/jpeg", filename=f"mubear_{session_id}.jpg")


@app.get("/display/{session_id}", response_class=HTMLResponse)
def display_page(session_id: str):
    info = storage.get_session(session_id)
    if not info:
        raise HTTPException(404, "session not found")
    qr_url = f"/photo/{session_id}/qr.png"
    img_url = f"/photo/{session_id}/download"
    return f"""<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>扫码取照片</title>
<style>
body{{font-family:sans-serif;text-align:center;background:#000;color:#fff;margin:0;padding:32px}}
img.qr{{width:320px;background:#fff;padding:16px;border-radius:16px}}
img.photo{{max-width:360px;border-radius:12px;margin-top:24px}}
h1{{font-size:36px}}
</style></head>
<body>
<h1>扫码下载你的照片</h1>
<img class="qr" src="{qr_url}" alt="qr">
<br>
<img class="photo" src="{img_url}" alt="photo">
</body></html>"""


@app.get("/photo/{session_id}/qr.png")
def photo_qr(session_id: str):
    info = storage.get_session(session_id)
    if info is None:
        raise HTTPException(404, "session not found")
    url = info["session"].get("public_url") or f"{config.base_url()}/photo/{session_id}"
    return Response(content=qr.qr_png_bytes(url), media_type="image/png")


@app.get("/api/sessions")
def api_sessions():
    return storage.list_sessions()


def _mjpeg_gen():
    cam = capture.get_camera()
    cam.open()
    while True:
        frame = cam.read_frame()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.06)


@app.get("/stream")
def stream():
    return StreamingResponse(
        _mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
