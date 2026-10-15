import shutil
import subprocess
import threading
import time
import webbrowser

import uvicorn

from app import config

BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def open_ui():
    time.sleep(2)
    url = f"http://127.0.0.1:{config.SERVER_PORT}"
    browser = shutil.which("chrome") or shutil.which("msedge")
    if browser is None:
        for p in BROWSER_CANDIDATES:
            try:
                with open(p):
                    browser = p
                    break
            except OSError:
                continue
    if browser:
        subprocess.Popen([browser, f"--app={url}"])
    else:
        webbrowser.open(url)


if __name__ == "__main__":
    from app.main import app
    print(f"访问地址: {config.base_url()}")
    threading.Thread(target=open_ui, daemon=True).start()
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
