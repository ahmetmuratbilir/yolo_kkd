import json

with open("benchmark_output/results.json", encoding="utf-8") as f:
    data = json.load(f)

ekran = [d for d in data if "Ekran" in d["image"]]
pexels = [d for d in data if "Ekran" not in d["image"]]

print(f"=== TOPLAM: {len(data)} gorsel ===")
print(f"Senin gorsellerin (Ekran): {len(ekran)}")
print(f"Pexels gorseller: {len(pexels)}")
print()

print("=== SENIN GORSELLERININ ANALIZI ===")
for d in ekran:
    img = d["image"].split("\\")[-1].replace("Ekran goruntüsü ", "").strip()
    results = d["results"]
    if not results:
        print(f"[{img}] -> KISI BULUNAMADI!")
        continue
    for i, r in enumerate(results):
        helmet  = r.get("helmet", False)
        vest    = r.get("vest", False)
        lg      = r.get("left_glove", False)
        rg      = r.get("right_glove", False)
        glasses = r.get("glasses", False)
        safe    = r.get("safe", False)
        src     = r.get("fallback_sources", {})
        warnings = r.get("warnings", [])

        h_src   = src.get("helmet", "?")
        v_src   = src.get("vest", "?")
        lg_src  = src.get("left_glove", "?")
        rg_src  = src.get("right_glove", "?")
        gl_src  = src.get("glasses", "?")

        durum = "GUVENLI" if safe else "IHLAL"
        print(f"[{img}] Kisi {i+1} -> {durum}")
        print(f"  Kask   : {'VAR' if helmet else 'YOK'} ({h_src})")
        print(f"  Yelek  : {'VAR' if vest else 'YOK'} ({v_src})")
        print(f"  Sol El : {'VAR' if lg else 'YOK'} ({lg_src})")
        print(f"  Sag El : {'VAR' if rg else 'YOK'} ({rg_src})")
        print(f"  Gozluk : {'VAR' if glasses else 'YOK'} ({gl_src})")
        if warnings:
            print(f"  >> {', '.join(warnings)}")
        print()

# Ozet istatistik
print("=" * 60)
print("=== OZET ISTATISTIK (Senin gorsellerin) ===")
total_persons = sum(len(d["results"]) for d in ekran)
helmet_ok  = sum(1 for d in ekran for r in d["results"] if r.get("helmet"))
vest_ok    = sum(1 for d in ekran for r in d["results"] if r.get("vest"))
lg_ok      = sum(1 for d in ekran for r in d["results"] if r.get("left_glove"))
rg_ok      = sum(1 for d in ekran for r in d["results"] if r.get("right_glove"))
glasses_ok = sum(1 for d in ekran for r in d["results"] if r.get("glasses"))
safe_count = sum(1 for d in ekran for r in d["results"] if r.get("safe"))

print(f"Toplam kisi: {total_persons}")
print(f"Kask    VAR: {helmet_ok}/{total_persons}  ({100*helmet_ok/max(total_persons,1):.0f}%)")
print(f"Yelek   VAR: {vest_ok}/{total_persons}  ({100*vest_ok/max(total_persons,1):.0f}%)")
print(f"Sol El  VAR: {lg_ok}/{total_persons}  ({100*lg_ok/max(total_persons,1):.0f}%)")
print(f"Sag El  VAR: {rg_ok}/{total_persons}  ({100*rg_ok/max(total_persons,1):.0f}%)")
print(f"Gozluk  VAR: {glasses_ok}/{total_persons}  ({100*glasses_ok/max(total_persons,1):.0f}%)")
print(f"GUVENLI kisi: {safe_count}/{total_persons}")

# fallback kaynak dagilimi
print()
print("=== KARAR KAYNAKLARI ===")
from collections import Counter
for equip in ["helmet", "vest", "left_glove", "right_glove", "glasses"]:
    sources = [r.get("fallback_sources", {}).get(equip, "?") for d in ekran for r in d["results"]]
    c = Counter(sources)
    print(f"{equip}: {dict(c)}")
