import threading

from . import config


class Voice:
    def __init__(self):
        self.enabled = config.ENABLE_VOICE
        self.log = []
        self._lock = threading.Lock()
        try:
            import pyttsx3
            self._engine_factory = pyttsx3
        except Exception:
            self._engine_factory = None
            self.enabled = False

    def say(self, text: str, block: bool = False):
        with self._lock:
            self.log.append(text)
        if not self.enabled or self._engine_factory is None:
            return
        if block:
            self._speak(text)
        else:
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text: str):
        try:
            engine = self._engine_factory.init()
            engine.setProperty("rate", 180)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass

    def take_log(self) -> list:
        with self._lock:
            out = self.log
            self.log = []
            return out


_voice = None


def get_voice() -> Voice:
    global _voice
    if _voice is None:
        _voice = Voice()
    return _voice
