import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config


class CosUploader:
    def __init__(self):
        self.client = None
        self.error = None
        if not (config.COS_SECRET_ID and config.COS_SECRET_KEY
                and config.COS_REGION and config.COS_BUCKET):
            self.error = "cos not configured"
            return
        try:
            from qcloud_cos import CosConfig, CosS3Client
            cfg = CosConfig(
                Region=config.COS_REGION,
                SecretId=config.COS_SECRET_ID,
                SecretKey=config.COS_SECRET_KEY,
            )
            self.client = CosS3Client(cfg)
        except Exception as e:
            self.error = str(e)

    @property
    def available(self) -> bool:
        return self.client is not None

    def upload_photo(self, local_path: Path, session_id: str) -> str:
        key = f"{config.COS_PREFIX}/{session_id}.jpg"
        self.client.upload_file(
            Bucket=config.COS_BUCKET,
            Key=key,
            LocalFilePath=str(local_path),
        )
        return self.presigned_url(key)

    def presigned_url(self, key: str) -> str:
        return self.client.get_presigned_download_url(
            Bucket=config.COS_BUCKET,
            Key=key,
            Expired=config.PRESIGN_EXPIRE_SECONDS,
        )

    def cleanup_expired(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=config.CLEANUP_AGE_SECONDS)
        deleted = 0
        marker = ""
        while True:
            resp = self.client.list_objects(
                Bucket=config.COS_BUCKET,
                Prefix=f"{config.COS_PREFIX}/",
                Marker=marker,
                MaxKeys=1000,
            )
            contents = resp.get("Contents", [])
            for obj in contents:
                last_modified = datetime.fromisoformat(
                    obj["LastModified"].replace("Z", "+00:00")
                )
                if last_modified < cutoff:
                    self.client.delete_object(Bucket=config.COS_BUCKET, Key=obj["Key"])
                    deleted += 1
            if resp.get("IsTruncated") == "true" and contents:
                marker = contents[-1]["Key"]
            else:
                break
        return deleted


_uploader = None
_cleanup_started = False


def get_uploader() -> CosUploader:
    global _uploader
    if _uploader is None:
        _uploader = CosUploader()
    return _uploader


def start_cleanup_loop():
    global _cleanup_started
    if _cleanup_started:
        return
    _cleanup_started = True

    def loop():
        while True:
            time.sleep(config.CLEANUP_INTERVAL_SECONDS)
            up = get_uploader()
            if up.available:
                try:
                    up.cleanup_expired()
                except Exception:
                    pass

    threading.Thread(target=loop, daemon=True).start()
