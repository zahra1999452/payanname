"""Generate a new vector figure with an explicit signed metric label."""
from pathlib import Path
from collections import defaultdict
import csv,math,html
ROOT=Path(__file__).resolve().parent
groups=defaultdict(list)
with (ROOT/'H2_degradation_summary.csv').open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):groups[row['model']].append(float(row['mean_absolute_f1_drop']))
models=['YOLOv8','RT-DETR','RTMDet'];values=[]
for model in models:
    assert len(groups[model])==6
    values.append(math.fsum(groups[model])/6)
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" viewBox="0 0 1200 700">','<rect width="1200" height="700" fill="white"/>','<g font-family="Arial,sans-serif" fill="#202020">','<text x="600" y="45" text-anchor="middle" font-size="26">H2: mean signed F1 drop across six corruptions</text>','<text x="600" y="78" text-anchor="middle" font-size="18">D_signed = F1_clean - F1_corrupted; negative means improvement</text>']
for k in range(6):
    y=560-k*88
    svg += [f'<path d="M160 {y}H1110" stroke="#dddddd"/>',f'<text x="145" y="{y+6}" text-anchor="end" font-size="18">{k*.02:.2f}</text>']
for i,(model,value) in enumerate(zip(models,values)):
    x=240+i*300;height=value/.1*440;y=560-height
    svg += [f'<rect x="{x}" y="{y:.4f}" width="180" height="{height:.4f}" fill="#2e74b5"/>',f'<text x="{x+90}" y="{y-14:.4f}" text-anchor="middle" font-size="22">{value:.6f}</text>',f'<text x="{x+90}" y="600" text-anchor="middle" font-size="22">{html.escape(model)}</text>']
svg += ['<text transform="translate(55 340) rotate(-90)" text-anchor="middle" font-size="22">Mean signed F1 drop</text>','<text x="600" y="654" text-anchor="middle" font-size="17">Stored-CSV reanalysis; 2462 images x 6 conditions per model; no model rerun</text>','</g></svg>']
(ROOT/'H2_mean_signed_f1_drop.svg').write_text('\n'.join(svg),encoding='utf8')
print(dict(zip(models,values)))
