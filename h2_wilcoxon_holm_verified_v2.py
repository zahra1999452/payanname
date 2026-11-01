# -*- coding: utf-8 -*-
"""H2: paired Wilcoxon/Holm on absolute F1 drop; CSV input, stdlib only.

Run: python h2_wilcoxon_holm_verified_v2.py per_image_degradation.csv --out out
Primary unit: image mean across six conditions. Pooled results are descriptive
sensitivity only: repeated conditions of one image are not independent.
Differences are rounded to 12 decimals before ranking (numerical ties).
Asymptotic two-sided p; wilcox zeros; tie correction; no continuity correction.
"""
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
import math
import platform
import statistics
from collections import defaultdict
from pathlib import Path

ALIASES = {'RT-DETR': 'RT-DETR-L', 'RTMDet': 'RTMDet-Tiny', 'YOLOv8': 'YOLOv8n'}
MODELS = ['RT-DETR-L', 'RTMDet-Tiny', 'YOLOv8n']
CONDITIONS = {'brightness', 'contrast', 'gaussian_noise', 'gaussian_blur',
              'motion_blur', 'jpeg_compression'}
DECIMALS = 12
VERSION = 'H2_absolute_F1_verified_v2_timestamped'

def mean(values):
    return math.fsum(values) / len(values)

def wilcoxon_pair(differences):
    d = [round(float(x), DECIMALS) for x in differences]
    if not d or not all(math.isfinite(x) for x in d):
        raise ValueError('Paired differences must be finite and nonempty')
    nz = [x for x in d if x != 0]
    n = len(nz)
    if n == 0:
        return dict(n=len(d), n_zero=len(d), n_nonzero=0, W=0.0, p=1.0,
                    r_rb=0.0, mean_diff=mean(d), median_diff=statistics.median(d),
                    status='all differences zero; no signed ranks')
    counts = defaultdict(int)
    for x in nz:
        counts[abs(x)] += 1
    ranks, next_rank, tie_sum = {}, 1, 0
    for value, count in sorted(counts.items()):
        ranks[value] = (next_rank + next_rank + count - 1) / 2
        next_rank += count
        tie_sum += count**3 - count
    r_plus = math.fsum(ranks[abs(x)] for x in nz if x > 0)
    r_minus = math.fsum(ranks[abs(x)] for x in nz if x < 0)
    variance = (n*(n+1)*(2*n+1) - tie_sum/2) / 24
    z = (r_plus - n*(n+1)/4) / math.sqrt(variance)
    p = math.erfc(abs(z) / math.sqrt(2))
    return dict(n=len(d), n_zero=len(d)-n, n_nonzero=n, W=min(r_plus,r_minus),
                p=p, r_rb=(r_plus-r_minus)/(r_plus+r_minus), mean_diff=mean(d),
                median_diff=statistics.median(d), status='asymptotic; wilcox')

def test_pairs(wide, level):
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        # Positive D_b-D_a: model a has less drop; positive r favours a.
        diff = [v[b]-v[a] for _, v in sorted(wide.items())]
        rows.append(dict(level=level, pair=f'{a} vs {b}',
                         **wilcoxon_pair(diff), n_dropped=0))
    previous = 0.0
    for rank, i in enumerate(sorted(range(len(rows)), key=lambda i: rows[i]['p'])):
        previous = max(previous, (len(rows)-rank)*rows[i]['p'])
        rows[i]['p_holm'] = min(1.0, previous)
    return rows

def load(path):
    grouped, pooled = defaultdict(list), defaultdict(dict)
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'model', 'image_id', 'condition'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing model/image_id/condition columns')
        use_f1 = {'clean_f1','corrupted_f1'}.issubset(reader.fieldnames)
        if not use_f1 and 'absolute_f1_drop' not in reader.fieldnames:
            raise ValueError('Need clean_f1+corrupted_f1 or absolute_f1_drop; no degradation fallback')
        for r in reader:
            m = ALIASES.get(r['model'], r['model'])
            i, c = r['image_id'], r['condition']
            if m not in MODELS or c not in CONDITIONS or not i:
                raise ValueError('Unexpected model, condition, or image ID')
            if m in pooled[i,c]:
                raise ValueError('Duplicate model/image/condition')
            if use_f1:
                a, b = float(r['clean_f1']), float(r['corrupted_f1'])
                if not (0 <= a <= 1 and 0 <= b <= 1):
                    raise ValueError('F1 outside [0,1]')
                drop = a-b
                if r.get('absolute_f1_drop') and abs(drop-float(r['absolute_f1_drop']))>1e-10:
                    raise ValueError('absolute_f1_drop disagrees with clean-corrupted')
            else:
                drop = float(r['absolute_f1_drop'])
            if not math.isfinite(drop) or not -1 <= drop <= 1:
                raise ValueError('Invalid absolute F1 drop')
            pooled[i,c][m] = drop
            grouped[m,i].append((c,drop))
    if not pooled or any(set(v)!=set(MODELS) for v in pooled.values()):
        raise ValueError('Missing paired models')
    wide = defaultdict(dict)
    for (m,i), values in grouped.items():
        if len(values)!=6 or {c for c,_ in values}!=CONDITIONS:
            raise ValueError('Each image/model must have all six conditions')
        wide[i][m] = mean([v for _,v in values])
    if any(set(v)!=set(MODELS) for v in wide.values()):
        raise ValueError('Missing per-image model pair')
    return wide, pooled

def main():
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv', type=Path)
    ap.add_argument('--out', type=Path, default=Path('h2_verified_output'))
    args = ap.parse_args()
    wide, pooled = load(args.csv)
    rows = test_pairs(wide,'per-image') + test_pairs(pooled,'pooled image×condition')
    args.out.mkdir(parents=True, exist_ok=True)
    fields = ['level','pair','n','n_zero','n_nonzero','W','p','p_holm',
              'r_rb','median_diff','mean_diff','n_dropped','status']
    with (args.out/'h2_inferential_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:format(v,'.17g') if isinstance(v,float) else v for k,v in row.items()})
    meta = dict(version=VERSION, python=platform.python_version(),
                started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                input_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                metric='clean_f1-corrupted_f1 or absolute_f1_drop; never degradation',
                paired_difference='D_second-D_first', decimals=DECIMALS,
                primary_n=len(wide), pooled_n=len(pooled),
                method='two-sided asymptotic normal; tie correction; wilcox; no continuity correction',
                zero_policy='round differences to 12 decimals; exclude zeros from ranks',
                p_adjustment='Holm separately within each three-pair family',
                limitation='image independence; pooled conditions are dependent')
    (args.out/'run_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    for row in rows:
        print(row['level'], row['pair'], 'n_zero=',row['n_zero'],
              'p_Holm=',format(row['p_holm'],'.8e'), 'r=',format(row['r_rb'],'.6f'))

if __name__ == '__main__':
    main()
