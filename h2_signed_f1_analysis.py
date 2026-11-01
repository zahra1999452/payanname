"""Run the professor's unchanged v2 algorithm; clarify its historical labels.

D_signed = clean_f1 - corrupted_f1. Negative values indicate improvement.
Historical absolute_f1_drop names do NOT mean abs(clean_f1-corrupted_f1).
"""
from pathlib import Path
import json
import sys
import h2_wilcoxon_holm_verified_v2 as canonical

if __name__ == '__main__':
    canonical.main()
    out = Path(sys.argv[sys.argv.index('--out')+1]) if '--out' in sys.argv else Path('h2_verified_output')
    p=out/'run_manifest.json'
    meta=json.loads(p.read_text(encoding='utf8'))
    meta.update(metric='D_signed = clean_f1 - corrupted_f1',
                legacy_metric_aliases=['absolute_f1_drop','D_abs','mean_absolute_f1_drop'],
                negative_value_interpretation='improvement; no mathematical absolute value',
                canonical_algorithm_modified=False)
    p.write_text(json.dumps(meta,indent=2),encoding='utf8')
