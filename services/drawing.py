# ============================================================
#  services/drawing.py  –  ISG KKD Algilama Sistemi - Gorsellestirme
#  Modern, nükleer güvenlik temalı UI
# ============================================================
import cv2
import numpy as np
import config

# ── Renk Paleti (BGR) ──────────────────────────────────────────────────── #
_GREEN      = (50, 220, 80)       # Güvenli - canlı yeşil
_RED        = (40, 40, 220)       # Tehlike - kırmızı
_ORANGE     = (30, 140, 255)      # Uyarı - turuncu
_YELLOW     = (0, 210, 255)       # Sarı vurgu
_CYAN       = (220, 200, 0)       # Bilgi - cyan
_WHITE      = (255, 255, 255)
_BLACK      = (0, 0, 0)
_DARK       = (20, 20, 30)        # Koyu arka plan
_GRAY       = (140, 140, 140)
_OPS_GRAY   = (100, 100, 100)     # OPS (opsiyonel) gri
_SMOKE_RED  = (30, 30, 200)       # Sigara uyarısı
_PANEL_BG   = (30, 35, 45)        # Panel arka planı

_FONT       = cv2.FONT_HERSHEY_DUPLEX
_FONT_SM    = cv2.FONT_HERSHEY_SIMPLEX


def _overlay_rect(frame, x1, y1, x2, y2, color, alpha=0.6):
    """Yarı saydam dikdörtgen çizer."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _rounded_rect(frame, x1, y1, x2, y2, color, thickness=2, radius=8):
    """Köşe efektli dikdörtgen (simüle)."""
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y1 + thickness), color, -1)
    cv2.rectangle(frame, (x1 + radius, y2 - thickness), (x2 - radius, y2), color, -1)
    cv2.rectangle(frame, (x1, y1 + radius), (x1 + thickness, y2 - radius), color, -1)
    cv2.rectangle(frame, (x2 - thickness, y1 + radius), (x2, y2 - radius), color, -1)


def draw_person_status(frame, person_result: dict) -> None:
    """
    Tek bir kişi için görsel çizer:
    - Renkli bounding box (yeşil/sarı/kırmızı)
    - Kişi ID + durum bandı
    - Sağda ekipman durum paneli
    - Uyarı mesajları
    """
    x1, y1, x2, y2 = map(int, person_result["box"])
    safe     = person_result["safe"]
    smoking  = person_result.get("smoking", False)
    pending  = any(person_result.get("pending_missing", {}).values())

    # Renk belirle
    if smoking:
        box_clr = _SMOKE_RED
    elif not safe:
        box_clr = _RED
    elif pending:
        box_clr = _ORANGE
    else:
        box_clr = _GREEN

    # ── Kişi kutusu (çift çizgi efekti) ──────────────────────── #
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_clr, 2)
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (*box_clr[:2], max(0, box_clr[2]-60)), 1)

    # ── Üst etiket bandı ──────────────────────────────────────── #
    band_h = 24
    band_top = max(0, y1 - band_h)
    _overlay_rect(frame, x1, band_top, x2, y1, box_clr, alpha=0.85)

    if smoking:
        status_text = "SIGARA!"
    elif not safe:
        status_text = "UYARI"
    elif pending:
        status_text = "KONTROL"
    else:
        status_text = "GUVENLI"

    pid = person_result.get("person_id", "?")
    label = f"P{pid}  {status_text}"
    cv2.putText(frame, label, (x1 + 5, band_top + 17),
                _FONT, 0.48, _WHITE, 1, cv2.LINE_AA)

    # ── Sağ ekipman paneli ────────────────────────────────────── #
    items = [
        ("Kask",   "helmet",      person_result["helmet"]),
        ("Yelek",  "vest",        person_result["vest"]),
        ("Gozluk", "glasses",     person_result["glasses"]),
        ("Sol E.", "left_glove",  person_result["left_glove"]),
        ("Sag E.", "right_glove", person_result["right_glove"]),
        ("Maske",  "mask",        person_result["mask"]),
    ]
    required        = person_result.get("required", {})
    confirmed_miss  = person_result.get("confirmed_missing", {})
    pending_miss    = person_result.get("pending_missing", {})

    frame_w  = frame.shape[1]
    panel_w  = 90
    panel_x1 = x2 + 5
    if panel_x1 + panel_w > frame_w:
        panel_x1 = max(2, x1 - panel_w - 5)
    panel_x2 = panel_x1 + panel_w
    panel_h  = len(items) * 19 + 8
    panel_y1 = y1

    # Panel arka planı
    _overlay_rect(frame, panel_x1, panel_y1, panel_x2, panel_y1 + panel_h, _PANEL_BG, alpha=0.75)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y1 + panel_h), box_clr, 1)

    for i, (name, key, ok) in enumerate(items):
        req = required.get(key, True)
        y_pos = panel_y1 + 16 + i * 19

        if not req:
            clr   = _OPS_GRAY
            state = "OPS"
            dot   = (150, 150, 150)
        elif ok:
            clr   = _GREEN
            state = "VAR"
            dot   = _GREEN
        elif pending_miss.get(key):
            clr   = _ORANGE
            state = "YOK?"
            dot   = _ORANGE
        else:
            clr   = _RED
            state = "YOK"
            dot   = _RED

        # Renkli nokta göstergesi
        cv2.circle(frame, (panel_x1 + 7, y_pos - 4), 4, dot, -1)
        text = f"{name}:{state}"
        cv2.putText(frame, text, (panel_x1 + 15, y_pos),
                    _FONT_SM, 0.38, clr, 1, cv2.LINE_AA)

    # Sigara ikonu ekipman panelinde
    if smoking:
        smo_y = panel_y1 + panel_h + 14
        _overlay_rect(frame, panel_x1, panel_y1 + panel_h, panel_x2, smo_y + 2, _SMOKE_RED, alpha=0.85)
        cv2.putText(frame, "SIGARA!", (panel_x1 + 8, smo_y - 2),
                    _FONT_SM, 0.38, _WHITE, 1, cv2.LINE_AA)

    # ── Uyarı mesajları ───────────────────────────────────────── #
    if not safe:
        warnings = person_result.get("warnings", [])
        for i, warn in enumerate(warnings[:4]):
            wy = y2 + 16 + i * 16
            if wy > frame.shape[0] - 5:
                break
            # Arka plan şeridi
            tw = cv2.getTextSize(warn, _FONT_SM, 0.40, 1)[0][0]
            _overlay_rect(frame, x1, wy - 12, x1 + tw + 8, wy + 4, _DARK, alpha=0.6)
            clr = _SMOKE_RED if "SIGARA" in warn else _RED
            cv2.putText(frame, warn, (x1 + 4, wy),
                        _FONT_SM, 0.40, clr, 1, cv2.LINE_AA)
    elif pending:
        for i, warn in enumerate(person_result.get("raw_warnings", [])[:2]):
            wy = y2 + 16 + i * 16
            if wy > frame.shape[0] - 5:
                break
            cv2.putText(frame, f"? {warn}", (x1 + 4, wy),
                        _FONT_SM, 0.38, _ORANGE, 1, cv2.LINE_AA)


def draw_fps(frame, fps: float) -> None:
    """Sol alt köşeye FPS göstergesi."""
    h = frame.shape[0]
    _overlay_rect(frame, 4, h - 28, 110, h - 4, _DARK, alpha=0.7)
    clr = _GREEN if fps >= 20 else (_ORANGE if fps >= 10 else _RED)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10),
                _FONT_SM, 0.52, clr, 1, cv2.LINE_AA)


def draw_header(frame, total_persons: int, unsafe_count: int,
                smoking_count: int = 0, debug_text: str = "") -> None:
    """Üst bilgi çubuğu — nükleer güvenlik temalı."""
    h, w = frame.shape[:2]
    has_debug = bool(debug_text)
    bar_h = 62 if has_debug else 46

    # Arka plan (koyu gradyan efekti)
    _overlay_rect(frame, 0, 0, w, bar_h, _DARK, alpha=0.88)
    # Alt çizgi aksanı
    accent_clr = _RED if (unsafe_count > 0 or smoking_count > 0) else _GREEN
    cv2.rectangle(frame, (0, bar_h - 2), (w, bar_h), accent_clr, -1)

    # ── Sol: Logo / Başlık ────────────────────────────────────── #
    cv2.putText(frame, "NUKLEER REAKTOR", (10, 18),
                _FONT_SM, 0.52, _CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, "ISG SISTEMI", (10, 36),
                _FONT_SM, 0.48, _GRAY, 1, cv2.LINE_AA)

    # ── Orta: Kişi sayıları ───────────────────────────────────── #
    safe_count = total_persons - unsafe_count
    mid_x = w // 2 - 120

    # Toplam
    _draw_stat_box(frame, mid_x, 4, f"KISI: {total_persons}", _CYAN)
    # Güvenli
    _draw_stat_box(frame, mid_x + 110, 4, f"GUVENLI: {safe_count}", _GREEN)
    # Uyarılı
    warn_clr = _RED if unsafe_count > 0 else _GRAY
    _draw_stat_box(frame, mid_x + 230, 4, f"UYARILI: {unsafe_count}", warn_clr)

    # ── Sigara uyarısı ────────────────────────────────────────── #
    if smoking_count > 0:
        sig_x = w - 185
        _overlay_rect(frame, sig_x, 5, sig_x + 175, 40, _SMOKE_RED, alpha=0.80)
        cv2.putText(frame, f"!  SIGARA ALGILANDI: {smoking_count}", (sig_x + 6, 27),
                    _FONT_SM, 0.46, _WHITE, 1, cv2.LINE_AA)
    elif unsafe_count == 0 and total_persons > 0:
        cv2.putText(frame, "TUM PERSONEL GUVENLI", (w - 230, 28),
                    _FONT_SM, 0.48, _GREEN, 1, cv2.LINE_AA)

    # ── Debug ─────────────────────────────────────────────────── #
    if has_debug:
        cv2.putText(frame, debug_text, (8, bar_h - 6),
                    _FONT_SM, 0.38, _YELLOW, 1, cv2.LINE_AA)


def _draw_stat_box(frame, x: int, y: int, text: str, color) -> None:
    """Küçük istatistik kutusu çizer."""
    tw, th = cv2.getTextSize(text, _FONT_SM, 0.47, 1)[0]
    _overlay_rect(frame, x, y, x + tw + 12, y + th + 10, color, alpha=0.18)
    cv2.rectangle(frame, (x, y), (x + tw + 12, y + th + 10), color, 1)
    cv2.putText(frame, text, (x + 6, y + th + 4),
                _FONT_SM, 0.47, color, 1, cv2.LINE_AA)
