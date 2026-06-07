@echo off
echo Sinif dagilimi sayiliyor (Super Hizli - Native C)...
echo ---------------------------------------------------
echo 0: person
findstr /b /c:"0 " datasets\combined_ppe\train\labels\*.txt | find /c /v ""

echo 1: helmet_pos
findstr /b /c:"1 " datasets\combined_ppe\train\labels\*.txt | find /c /v ""

echo 2: helmet_neg
findstr /b /c:"2 " datasets\combined_ppe\train\labels\*.txt | find /c /v ""

echo 3: vest_pos
findstr /b /c:"3 " datasets\combined_ppe\train\labels\*.txt | find /c /v ""

echo 4: vest_neg
findstr /b /c:"4 " datasets\combined_ppe\train\labels\*.txt | find /c /v ""

echo 5: gloves_pos
findstr /b /c:"5 " datasets\combined_ppe\train\labels\*.txt | find /c /v ""

echo 6: gloves_neg
findstr /b /c:"6 " datasets\combined_ppe\train\labels\*.txt | find /c /v ""

echo 7: goggles_pos
findstr /b /c:"7 " datasets\combined_ppe\train\labels\*.txt | find /c /v ""

echo 8: goggles_neg
findstr /b /c:"8 " datasets\combined_ppe\train\labels\*.txt | find /c /v ""
echo ---------------------------------------------------
echo Sayim tamamlandi!
