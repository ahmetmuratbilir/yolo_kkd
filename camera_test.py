import cv2


BACKENDS = {
    "DSHOW": cv2.CAP_DSHOW,
    "MSMF": cv2.CAP_MSMF,
    "ANY": 0,
}

FOURCCS = ["", "MJPG"]


def configure(cap, fourcc):
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)


def main():
    print("[camera_test] Kamera kombinasyonlari deneniyor...")
    for source in range(2):
        for backend_name, backend in BACKENDS.items():
            for fourcc in FOURCCS:
                cap = cv2.VideoCapture(source, backend)
                configure(cap, fourcc)

                ok = cap.isOpened()
                ret, frame = cap.read() if ok else (False, None)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok else 0
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok else 0
                fps = cap.get(cv2.CAP_PROP_FPS) if ok else 0
                cap.release()

                status = "OK" if ret and frame is not None and frame.size > 0 else "YOK"
                shape = f"{frame.shape[1]}x{frame.shape[0]}" if ret and frame is not None else "-"
                print(
                    f"source={source} backend={backend_name:5} fourcc={fourcc or 'NONE':4} "
                    f"opened={ok} read={status} prop={width}x{height}@{fps:.1f} frame={shape}"
                )


if __name__ == "__main__":
    main()
