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

# Durum ikonları (metin simgesi)
_ICONS = {
    True:  "OK ",
    False: "!! ",
}


def draw_person_status(frame, person_result: dict) -> None:
    """
    Tek bir kişi için:
      - Bounding box çizer (yeşil=güvenli, kırmızı=uyarılı)
      - Sol üst köşeye ekipman özeti yazar
      - Kişi ID'sini gösterir
    """
    x1, y1, x2, y2 = map(int, person_result["box"])
    safe    = person_result["safe"]
    box_clr = _GREEN if safe else _RED

    # Kişi kutusu
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_clr, 2)

    # Üst bant (arka plan)
    banner_h = 20
    cv2.rectangle(frame, (x1, y1 - banner_h), (x2, y1), box_clr, -1)

    # Kişi ID + durum
    label = f"P{person_result['person_id']}  {'GUVENLI' if safe else 'UYARI'}"
    cv2.putText(frame, label, (x1 + 4, y1 - 5),
                _FONT, _SCALE, _WHITE, _THICK, cv2.LINE_AA)

    # Uyarı mesajları kutunun altında
    if not safe:
        for i, warn in enumerate(person_result["warnings"]):
            y_pos = y2 + 18 + i * 18
            cv2.putText(frame, f"  {warn}", (x1, y_pos),
                        _FONT, 0.48, _RED, 1, cv2.LINE_AA)

    # Sağ kenar ekipman özeti
    items = [
        ("Kask",   person_result["helmet"]),
        ("Yelek",  person_result["vest"]),
        ("Maske",  person_result["mask"]),
        ("Sol E.", person_result["left_glove"]),
        ("Sag E.", person_result["right_glove"]),
    ]
    panel_x = x2 + 6
    for i, (name, ok) in enumerate(items):
        clr   = _GREEN if ok else _RED
        icon  = _ICONS[ok]
        text  = f"{icon}{name}"
        y_pos = y1 + 16 + i * 18
        cv2.putText(frame, text, (panel_x, y_pos),
                    _FONT, 0.45, clr, 1, cv2.LINE_AA)


def draw_fps(frame, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                _FONT, 0.65, _YELLOW, 2, cv2.LINE_AA)


def draw_header(frame, total_persons: int, unsafe_count: int) -> None:
    """Ekranın üst kısmına genel özet yazar."""
    h, w = frame.shape[:2]
    bar_h = 36
    cv2.rectangle(frame, (0, 0), (w, bar_h), _BLACK, -1)
    summary = (
        f"Kisi: {total_persons}  |  "
        f"Uyarili: {unsafe_count}  |  "
        f"Guvenli: {total_persons - unsafe_count}"
    )
    cv2.putText(frame, summary, (12, 24),
                _FONT, 0.65, _WHITE, 1, cv2.LINE_AA)
