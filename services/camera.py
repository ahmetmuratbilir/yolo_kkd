# ============================================================
#  services/camera.py  –  Windows kamera açılışını sağlamlaştırma
# ============================================================
import sys
import time

import cv2

import config


_BACKENDS = {
    "ANY": 0,
    "DSHOW": cv2.CAP_DSHOW,
    "MSMF": cv2.CAP_MSMF,
}


def open_camera():
    """Kamerayı birkaç backend ile dener ve ilk düzgün okuyan bağlantıyı döner."""
    source = config.CAMERA_SOURCE
    backend_names = getattr(config, "CAMERA_BACKENDS", ["MSMF", "DSHOW", "ANY"])
    if sys.platform != "win32":
        backend_names = ["ANY"]

    for backend_name in backend_names:
        backend = _BACKENDS.get(backend_name.upper(), 0)
        print(f"[Camera] Kamera deneniyor: source={source}, backend={backend_name}")

        cap = cv2.VideoCapture(source, backend)
        _configure_camera(cap)

        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = _read_warm_frame(cap)
        if ok and frame is not None:
            print(f"[Camera] Kamera acildi: backend={backend_name}, frame={frame.shape[1]}x{frame.shape[0]}")
            return cap, backend_name

        cap.release()
        time.sleep(0.2)

    return None, None


def reopen_camera(current_cap=None):
    if current_cap is not None:
        current_cap.release()
    return open_camera()


def _configure_camera(cap) -> None:
    if not cap:
        return

    fourcc = getattr(config, "CAMERA_FOURCC", "")
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    width = getattr(config, "CAMERA_WIDTH", 0)
    height = getattr(config, "CAMERA_HEIGHT", 0)
    fps = getattr(config, "CAMERA_FPS", 0)

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)


def _read_warm_frame(cap, tries: int = 15):
    last_frame = None
    for _ in range(tries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            last_frame = frame
        time.sleep(0.03)

    return last_frame is not None, last_frame
