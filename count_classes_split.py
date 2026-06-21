"""
count_classes_split.py
======================
Merge oncesi ve sonrasi dosyalari ayirt ederek sinif sayimi yapar.

Kural:
  - Dosya adinda '__' varsa  → merge_gloves_to_combined.py tarafindan eklendi
  - '__' yoksa               → orijinal combined_ppe dataseti

Cikti:
  - Her sinif icin ayri ayri oncesi/sonrasi annotation sayisi
  - Merge edilen resim basina ortalama gloves kutu sayisi
  - 10'dan fazlaysa uyari (etiket patlamasi)
"""

from pathlib import Path
from collections import defaultdict

DATASET = Path("datasets/combined_ppe")
SPLIT   = "train"   # sadece train analiz edilir

CLASS_NAMES = {
    0: "person",
    1: "helmet_pos",
    2: "helmet_neg",
    3: "vest_pos",
    4: "vest_neg",
    5: "gloves_pos",
    6: "gloves_neg",
    7: "goggles_pos",
    8: "goggles_neg",
    9: "smoking",
}

GLOVES_IDS = {5, 6}

def main():
    lbl_dir = DATASET / SPLIT / "labels"
    if not lbl_dir.exists():
        print(f"[!] Klasor bulunamadi: {lbl_dir}")
        return

    all_files = [f for f in lbl_dir.iterdir() if f.suffix == ".txt"]
    print(f"Toplam etiket dosyasi ({SPLIT}): {len(all_files):,}")

    # oncesi / sonrasi ayir
    before = [f for f in all_files if "__" not in f.stem]
    after  = [f for f in all_files if "__" in f.stem]

    print(f"  Merge oncesi (orijinal): {len(before):,}")
    print(f"  Merge sonrasi (eklenen): {len(after):,}")
    print()

    # Sayim fonksiyonu
    def count(file_list):
        cls_counts  = defaultdict(int)
        gloves_boxes_per_file = []
        for f in file_list:
            try:
                lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            gloves_in_this = 0
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                c = int(parts[0])
                cls_counts[c] += 1
                if c in GLOVES_IDS:
                    gloves_in_this += 1
            if gloves_in_this > 0:
                gloves_boxes_per_file.append(gloves_in_this)
        return cls_counts, gloves_boxes_per_file

    print("Sayiliyor: merge oncesi...")
    cnt_before, gboxes_before = count(before)
    print("Sayiliyor: merge sonrasi...")
    cnt_after,  gboxes_after  = count(after)
    print()

    # --- Tablo ---
    print("=" * 65)
    print(f"  {'Sinif':<16} {'Oncesi':>10}  {'Sonrasi':>10}  {'Toplam':>10}")
    print("-" * 65)
    for cid in sorted(CLASS_NAMES.keys()):
        n_b = cnt_before[cid]
        n_a = cnt_after[cid]
        name = CLASS_NAMES[cid]
        star = " <--" if cid in GLOVES_IDS else ""
        print(f"  {cid} {name:<14} {n_b:>10,}  {n_a:>10,}  {n_b+n_a:>10,}{star}")
    print("=" * 65)
    total_b = sum(cnt_before.values())
    total_a = sum(cnt_after.values())
    print(f"  {'TOPLAM':<16} {total_b:>10,}  {total_a:>10,}  {total_b+total_a:>10,}")
    print()

    # --- Kritik satirlar ---
    gp_b = cnt_before[5]; gp_a = cnt_after[5]
    gn_b = cnt_before[6]; gn_a = cnt_after[6]
    print(f"gloves_pos -> merge oncesi: {gp_b:,} | merge sonrasi eklenen: {gp_a:,}")
    print(f"gloves_neg -> merge oncesi: {gn_b:,} | merge sonrasi eklenen: {gn_a:,}")
    print()

    # --- Merge edilen resim basina ortalama gloves kutu sayisi ---
    if gboxes_after:
        avg = sum(gboxes_after) / len(gboxes_after)
        print(f"Merge edilen resim basina ortalama gloves kutu sayisi: {avg:.1f}")
        if avg > 10:
            print("[UYARI] Ortalama >10 — etiket patlamasi olabilir, ornekleri kontrol et!")
        else:
            print("[OK] Ortalama makul (<= 10)")
    else:
        print("Merge sonrasi gloves kutusu bulunamadi.")

    print()
    # --- Orijinal datasetteki gloves orani ---
    total_orig = sum(cnt_before.values())
    gloves_orig = cnt_before[5] + cnt_before[6]
    pct_orig = gloves_orig / total_orig * 100 if total_orig else 0
    print(f"Orijinal datasette gloves orani: %{pct_orig:.1f}")

    total_new = total_b + total_a
    gloves_new = (gp_b + gp_a) + (gn_b + gn_a)
    pct_new = gloves_new / total_new * 100 if total_new else 0
    print(f"Merge sonrasi gloves orani    : %{pct_new:.1f}")

    print()
    # --- Senaryo tespiti ---
    print("-" * 65)
    if pct_orig > 8:
        print("SENARYO A: Orijinal dataset zaten gloves-zengin.")
        print("  Merge ek katki yapmis ama asil veri zaten vardi.")
        print("  Oversampling GEREKMEZ — train_v4_clean.py'yi direkt calistir.")
    else:
        print("SENARYO B: Orijinal dataset gloves-yoksuldu, merge kritikti.")
        if pct_new < 5:
            print("  Merge sonrasi hala az — oversample_gloves.py --factor 3 onerilir.")
        else:
            print("  Merge sonrasi yeterli — train_v4_clean.py'yi direkt calistir.")
    print("-" * 65)

if __name__ == "__main__":
    main()
