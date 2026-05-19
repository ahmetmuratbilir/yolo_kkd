# ============================================================
#  services/drawing.py  –  OpenCV görsel katmanı
# ============================================================
import cv2
import config


# ── Sabitler ────────────────────────────────────────────────────────── #
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_SCALE      = config.FONT_SCALE
_THICK      = config.FONT_THICKNESS
_GREEN      = config.COLOR_SAFE
_RED        = config.COLOR_WARN
_YELLOW     = config.COLOR_BOX
_WHITE      = (255, 255, 255)
_BLACK      = (0, 0, 0)
_GRAY       = (150, 150, 150)

def draw_person_status(frame, person_result: dict) -> None:
    """
    Tek bir kişi için:
      - Bounding box çizer (yeşil=güvenli, kırmızı=uyarılı)
      - Sol üst köşeye ekipman özeti yazar
      - Kişi ID'sini gösterir
    """
    x1, y1, x2, y2 = map(int, person_result["box"])
    safe    = person_result["safe"]
    has_pending = any(person_result.get("pending_missing", {}).values())
    box_clr = _GREEN if safe else _RED
    if safe and has_pending:
        box_clr = _YELLOW

    # Kişi kutusu
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_clr, 2)

    # Üst bant (arka plan)
    banner_h = 20
    banner_top = max(0, y1 - banner_h)
    cv2.rectangle(frame, (x1, banner_top), (x2, y1), box_clr, -1)

    # Kişi ID + durum
    status_text = "GUVENLI" if safe else "UYARI"
    if safe and has_pending:
        status_text = "KONTROL"
    label = f"P{person_result['person_id']}  {status_text}"
    cv2.putText(frame, label, (x1 + 4, max(15, y1 - 5)),
                _FONT, _SCALE, _WHITE, _THICK, cv2.LINE_AA)

    # Uyarı mesajları kutunun altında
    if not safe:
        for i, warn in enumerate(person_result["warnings"]):
            y_pos = y2 + 18 + i * 18
            cv2.putText(frame, f"  {warn}", (x1, y_pos),
                        _FONT, 0.48, _RED, 1, cv2.LINE_AA)
    elif has_pending:
        pending_warnings = person_result.get("raw_warnings", [])
        for i, warn in enumerate(pending_warnings[:3]):
            y_pos = y2 + 18 + i * 18
            cv2.putText(frame, f"  Kontrol: {warn}", (x1, y_pos),
                        _FONT, 0.48, _YELLOW, 1, cv2.LINE_AA)

    # Sağ kenar ekipman özeti
    items = [
        ("Kask",   person_result["helmet"]),
        ("Yelek",  person_result["vest"]),
        ("Maske",  person_result["mask"]),
        ("Gozluk", person_result["glasses"]),
        ("Sol E.", person_result["left_glove"]),
        ("Sag E.", person_result["right_glove"]),
    ]
    required = person_result.get("required", {})
    confirmed_missing = person_result.get("confirmed_missing", {})
    pending_missing = person_result.get("pending_missing", {})

    frame_w = frame.shape[1]
    panel_w = 92
    panel_x = x2 + 6
    if panel_x + panel_w > frame_w:
        panel_x = max(2, x1 - panel_w)

    for i, (name, ok) in enumerate(items):
        key = ("helmet", "vest", "mask", "glasses", "left_glove", "right_glove")[i]
        if not required.get(key, True):
            clr = _GRAY
            state = "OPS"
        elif ok:
            clr = _GREEN
            state = "VAR"
        elif pending_missing.get(key):
            clr = _YELLOW
            state = "YOK"
        elif confirmed_missing.get(key, not ok):
            clr = _RED
            state = "YOK"
        else:
            clr = _RED
            state = "YOK"

        text  = f"{name}: {state}"
        y_pos = y1 + 16 + i * 18
        cv2.putText(frame, text, (panel_x, y_pos),
                    _FONT, 0.45, clr, 1, cv2.LINE_AA)


def draw_fps(frame, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                _FONT, 0.65, _YELLOW, 2, cv2.LINE_AA)


def draw_header(frame, total_persons: int, unsafe_count: int, debug_text: str = "") -> None:
    """Ekranın üst kısmına genel özet yazar."""
    h, w = frame.shape[:2]
    bar_h = 56 if debug_text else 36
    cv2.rectangle(frame, (0, 0), (w, bar_h), _BLACK, -1)
    summary = (
        f"Kisi: {total_persons}  |  "
        f"Uyarili: {unsafe_count}  |  "
        f"Guvenli: {total_persons - unsafe_count}"
    )
    cv2.putText(frame, summary, (12, 24),
                _FONT, 0.65, _WHITE, 1, cv2.LINE_AA)
    if debug_text:
        cv2.putText(frame, debug_text, (12, 46),
                    _FONT, 0.48, _YELLOW, 1, cv2.LINE_AA)
