# ============================================================
#  services/status_smoother.py  –  Kısa süreli yanlış alarm filtresi
# ============================================================
from collections import deque

import config


EQUIPMENT_KEYS = ("helmet", "vest", "mask", "glasses", "left_glove", "right_glove")
WARNING_LABELS = {
    "helmet": "Kask eksik",
    "vest": "Yelek eksik",
    "mask": "Maske eksik",
    "glasses": "Gozluk eksik",
    "left_glove": "Sol eldiven eksik",
    "right_glove": "Sag eldiven eksik",
}


class StatusSmoother:
    """
    Aynı kişi için son birkaç kareyi tutar ve ihlali sadece tutarlıysa onaylar.
    """

    def __init__(self):
        self.enabled = getattr(config, "ENABLE_STATUS_SMOOTHING", True)
        
        self.history_size_helmet_vest = getattr(config, "STATUS_HISTORY_SIZE_HELMET_VEST", 5)
        self.confirm_frames_helmet_vest = getattr(config, "STATUS_CONFIRM_FRAMES_HELMET_VEST", 2)
        
        self.history_size_gloves_goggles = getattr(config, "STATUS_HISTORY_SIZE_GLOVES_GOGGLES", 7)
        self.confirm_frames_gloves_goggles = getattr(config, "STATUS_CONFIRM_FRAMES_GLOVES_GOGGLES", 2)
        
        self.missing_timeout = getattr(config, "STATUS_MISSING_TIMEOUT_FRAMES", 3)
        
        self.max_history = max(self.history_size_helmet_vest, self.history_size_gloves_goggles)
        
        self.max_idle_frames = getattr(config, "TRACK_MAX_MISSED", 12) + self.max_history
        self.frame_index = 0
        self.histories = {}
        self.last_seen = {}
        # Son pozitif tespitlerin kaynağını sakla (fallback_sources tutarlılığı için)
        self.last_positive_source = {}

    # ------------------------------------------------------------------ #
    def update(self, results: list[dict]) -> list[dict]:
        if not self.enabled:
            return [self._without_smoothing(result) for result in results]

        self.frame_index += 1
        smoothed = []

        for result in results:
            person_id = result["person_id"]
            self.last_seen[person_id] = self.frame_index

            history = self.histories.setdefault(
                person_id,
                {key: deque(maxlen=self.max_history) for key in EQUIPMENT_KEYS},
            )
            # Kişi başı son pozitif kaynağı sakla
            pos_src = self.last_positive_source.setdefault(person_id, {})

            updated = dict(result)
            updated["raw_warnings"] = list(result.get("warnings", []))
            updated["confirmed_missing"] = {}
            updated["pending_missing"] = {}
            updated["fallback_sources"] = dict(result.get("fallback_sources", {}))

            stable_warnings = []
            required = result.get("required", getattr(config, "REQUIRED_EQUIPMENTS", {}))

            for equipment in EQUIPMENT_KEYS:
                detected_now = bool(result.get(equipment))
                history[equipment].append(detected_now)

                # Bu frame'de pozitif tespit olduysa kaynağı güncelle
                src_now = result.get("fallback_sources", {}).get(equipment, "")
                if detected_now and src_now and "default_missing" not in src_now:
                    pos_src[equipment] = src_now

                is_required = required.get(equipment, True)
                stably_missing = is_required and self._is_stably_missing(equipment, history[equipment])

                # Smoothed durum: tarihe göre STABLY_MISSING ise YOK, değilse VAR ANCAK
                # sadece bu frame veya geçmişte gerçekten görüldüyse VAR
                recent = list(history[equipment])
                ever_seen = any(recent)   # geçmişte en az 1 pozitif var mı?

                if stably_missing:
                    smoothed_val = False
                elif ever_seen:
                    smoothed_val = True
                    # Kaynak: bu frame'de görülduyse doğrudan, yoksa geçmişteki son kaynağı yaz
                    if not detected_now and equipment in pos_src:
                        updated["fallback_sources"][equipment] = f"smoothed({pos_src[equipment]})"
                else:
                    smoothed_val = False   # Hiç görülmedi, default_missing kalmasın

                updated[equipment] = smoothed_val

                confirmed = is_required and not smoothed_val
                pending = is_required and not detected_now and not confirmed

                updated["confirmed_missing"][equipment] = confirmed
                updated["pending_missing"][equipment] = pending

                if confirmed:
                    stable_warnings.append(WARNING_LABELS[equipment])

            updated["warnings"] = stable_warnings
            updated["safe"] = len(stable_warnings) == 0
            smoothed.append(updated)

        self._prune_old_histories()
        return smoothed

    # ------------------------------------------------------------------ #
    def _is_stably_missing(self, equipment: str, history: deque) -> bool:
        if equipment in ("helmet", "vest"):
            history_size = self.history_size_helmet_vest
            confirm_frames = self.confirm_frames_helmet_vest
        else:
            history_size = self.history_size_gloves_goggles
            confirm_frames = self.confirm_frames_gloves_goggles
            
        recent_all = list(history)[-history_size:] if len(history) >= history_size else list(history)
        
        if len(recent_all) == 0:
            return True
            
        # Hızlı düşüş kuralı: Eğer son N karedir (genelde 3) hiç pozitif yoksa, anında YOK kabul et.
        recent_timeout = list(history)[-self.missing_timeout:]
        if len(recent_timeout) >= self.missing_timeout and sum(1 for d in recent_timeout if d) == 0:
            return True
            
        # Normal onay kuralı: Son 'history_size' karede en az 'confirm_frames' kadar varsa VAR.
        positives = sum(1 for d in recent_all if d)
        if positives >= confirm_frames:
            return False
            
        # Yoksa varsayılan olarak YOK kabul et. (Belirsiz döndürmeme kuralı)
        return True

    # ------------------------------------------------------------------ #
    def _without_smoothing(self, result: dict) -> dict:
        updated = dict(result)
        updated["raw_warnings"] = list(result.get("warnings", []))
        required = result.get("required", getattr(config, "REQUIRED_EQUIPMENTS", {}))

        confirmed_missing = {}
        pending_missing = {}
        for equipment in EQUIPMENT_KEYS:
            confirmed_missing[equipment] = required.get(equipment, True) and not result.get(equipment)
            pending_missing[equipment] = False

        updated["confirmed_missing"] = confirmed_missing
        updated["pending_missing"] = pending_missing
        return updated

    # ------------------------------------------------------------------ #
    def _prune_old_histories(self) -> None:
        expired_ids = [
            person_id
            for person_id, last_seen in self.last_seen.items()
            if self.frame_index - last_seen > self.max_idle_frames
        ]

        for person_id in expired_ids:
            self.histories.pop(person_id, None)
            self.last_seen.pop(person_id, None)
            self.last_positive_source.pop(person_id, None)
