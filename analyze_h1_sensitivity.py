from pathlib import Path
from collections import defaultdict, Counter
import csv, json, hashlib, math, re

ROOT=Path(__file__).resolve().parent
WORK=ROOT
SOURCE=ROOT/'per_image_metrics.csv'
records=list(csv.DictReader(SOURCE.open(encoding='utf-8-sig', newline='')))
by_model=defaultdict(dict)
for r in records:
    model, image=r['model'],r['image_id']
    assert image not in by_model[model], (model,image)
    for k in ['tp','fp','fn','num_gt','num_predictions']:
        r[k]=int(float(r[k]))
    r['f1']=float(r['f1'])
    tp,fp,fn=r['tp'],r['fp'],r['fn']
    expected=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0
    assert abs(r['f1']-expected)<1e-12
    assert tp+fn==r['num_gt'] and tp+fp==r['num_predictions']
    by_model[model][image]=r
ids=set(next(iter(by_model.values())))
assert len(records)==7386 and len(ids)==2462
for model, data in by_model.items():
    assert set(data)==ids
for image in ids:
    assert len({data[image]['num_gt'] for data in by_model.values()})==1
mean=lambda a: math.fsum(a)/len(a)
results=[]
for model in ['YOLOv8','RT-DETR','RTMDet']:
    rows=list(by_model[model].values())
    positive=[r for r in rows if r['num_gt']>0]
    both=sum(r['num_gt']==0 and r['num_predictions']==0 for r in rows)
    results.append(dict(model=model,n_all=len(rows),n_gt_positive=len(positive),
        n_gt_empty=len(rows)-len(positive),n_both_empty=both,
        mean_f1_original=mean([r['f1'] for r in rows]),
        mean_f1_gt_positive=mean([r['f1'] for r in positive]),
        mean_f1_both_empty_zero=mean([0.0 if r['num_gt']==0 and r['num_predictions']==0 else r['f1'] for r in rows])))
groups=Counter()
unmatched=[]
for image in sorted(ids):
    match=re.fullmatch(r'(.+)_frame(\d+)',image)
    if match: groups[match.group(1)]+=1
    else: unmatched.append(image)
with (WORK/'h1_sensitivity_verified.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
report=dict(input_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    n_records=len(records),n_paired_images=len(ids),results=results,
    all_f1_checked_against_counts=True,gt_counts_identical_across_models=True,
    filename_proxy=dict(rule=r'^(.+)_frame(\d+)$',matched_images=sum(groups.values()),
       unmatched_images=len(unmatched),prefix_count=len(groups),
       multi_frame_prefix_count=sum(n>1 for n in groups.values()),
       min_group_size=min(groups.values()),max_group_size=max(groups.values()),
       group_size_counts=dict(sorted(Counter(groups.values()).items())),
       largest_groups=groups.most_common(12),unmatched_examples=unmatched[:12]),
    limitation='Filename prefixes are provisional source proxies only. No verified video/source mapping or train/validation manifests and labels are available. No cluster inference or split-independence claim is made.',
    sensitivity_inference='Descriptive only; original ranking compared. No new p-values were calculated.')
(WORK/'h1_sensitivity_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
