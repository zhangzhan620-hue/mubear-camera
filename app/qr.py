import io

import qrcode


def qr_png_bytes(url: str) -> bytes:
    img = qrcode.make(url, box_size=8, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
