# ============================================================
#  services/person_tracker.py  –  Basit kişi ID takibi
# ============================================================
import math

import config


def _center(box: list) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _area(box: list) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = _area([x1, y1, x2, y2])
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PersonTracker:
    """
    YOLO person kutularına kareler arasında sabit ID verir.
    Ağır bir takip bağımlılığı eklemeden IoU + merkez uzaklığı ile eşleştirir.
    """

    def __init__(self):
        self.enabled = getattr(config, "ENABLE_TRACKING", True)
        self.max_missed = getattr(config, "TRACK_MAX_MISSED", 12)
        self.min_iou = getattr(config, "TRACK_MIN_IOU", 0.05)
        self.max_distance = max(1.0, float(getattr(config, "TRACK_MAX_DISTANCE", 140)))
        self.next_id = 1
        self.tracks = {}

    # ------------------------------------------------------------------ #
    def update(self, persons: list[dict]) -> list[dict]:
        if not self.enabled:
            return [{**person, "track_id": idx + 1} for idx, person in enumerate(persons)]

        if not persons:
            self._age_unmatched_tracks(set())
            return []

        output = [dict(person) for person in persons]
        candidates = []

        for track_id, track in self.tracks.items():
            track_center = _center(track["box"])
            for det_idx, person in enumerate(persons):
                person_box = person["box"]
                iou = _iou(track["box"], person_box)
                dist = _distance(track_center, _center(person_box))

                if iou >= self.min_iou or dist <= self.max_distance:
                    distance_score = max(0.0, 1.0 - (dist / self.max_distance))
                    score = (2.0 * iou) + distance_score
                    candidates.append((score, track_id, det_idx))

        candidates.sort(reverse=True)
        matched_tracks = set()
        matched_detections = set()

        for _, track_id, det_idx in candidates:
            if track_id in matched_tracks or det_idx in matched_detections:
                continue

            matched_tracks.add(track_id)
            matched_detections.add(det_idx)
            output[det_idx]["track_id"] = track_id
            self.tracks[track_id] = {
                "box": persons[det_idx]["box"],
                "missed": 0,
            }

        self._age_unmatched_tracks(matched_tracks)

        for det_idx, person in enumerate(persons):
            if det_idx in matched_detections:
                continue

            track_id = self.next_id
            self.next_id += 1
            output[det_idx]["track_id"] = track_id
            self.tracks[track_id] = {
                "box": person["box"],
                "missed": 0,
            }

        return output

    # ------------------------------------------------------------------ #
    def _age_unmatched_tracks(self, matched_tracks: set[int]) -> None:
        expired = []
        for track_id, track in self.tracks.items():
            if track_id in matched_tracks:
                continue
            track["missed"] += 1
            if track["missed"] > self.max_missed:
                expired.append(track_id)

        for track_id in expired:
            self.tracks.pop(track_id, None)
