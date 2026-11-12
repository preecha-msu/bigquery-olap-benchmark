import csv, hashlib
from pathlib import Path
R = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(R/'reproducibility_v7'/'SHA256_MANIFEST_V7.csv', encoding='utf-8')))
for r in rows:
    h = hashlib.sha256((R/r['Relative_Path']).read_bytes()).hexdigest()
    assert h == r['SHA256'], f"Mismatch {r['Relative_Path']}"
print(f'[MANIFEST VERIFIED OK] {len(rows)}/{len(rows)} V7 SHA-256 hashes match 100%.')
