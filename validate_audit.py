"""Recheck syntax, unit contracts and stored-CSV reanalysis. No model execution."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'audit_validation';OUT.mkdir(exist_ok=True)
report={'started_utc':datetime.now(timezone.utc).isoformat(),
        'training_performed':False,'inference_performed':False,
        'integration_passed':False,'syntax':[],'steps':[],'imports':[]}
for path in sorted(ROOT.rglob('*.py')):
    try:
        compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
        report['syntax'].append({'path':str(path.relative_to(ROOT)),'status':'pass'})
    except Exception as error:report['syntax'].append({'path':str(path.relative_to(ROOT)),'status':'fail','error':repr(error)})
commands=[('unit_contracts',['test_audit_fixes.py']),
          ('H2_signed_figure',['make_signed_h2_plot.py']),
          ('H1_sensitivity',['analyze_h1_sensitivity.py']),
          ('H2_signed_reanalysis',['h2_signed_f1_analysis.py','per_image_degradation.csv','--out','audit_validation/h2'])]
for name,args in commands:
    result=subprocess.run([sys.executable,'-X','utf8','-B',*args],cwd=ROOT,capture_output=True,text=True,encoding='utf8',timeout=90)
    (OUT/f'{name}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
    report['steps'].append({'name':name,'exit_code':result.returncode,'log':f'audit_validation/{name}.log'})
for name in ['unified_trainer','train_rtdetr_l_6gb','train_rtmdet_tiny']:
    result=subprocess.run([sys.executable,'-X','utf8','-B','-c',f'import {name}'],cwd=ROOT,capture_output=True,text=True,encoding='utf8',timeout=30)
    (OUT/f'import_{name}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
    report['imports'].append({'module':name,'exit_code':result.returncode,'passed':result.returncode==0,'log':f'audit_validation/import_{name}.log'})
expected=hashlib.sha256((ROOT/'h2_inferential_results.csv').read_bytes()).hexdigest()
actual=hashlib.sha256((OUT/'h2/h2_inferential_results.csv').read_bytes()).hexdigest()
report['h2_six_rows_exact_match']=actual==expected
report['h2_output_sha256']=actual
report['completed_utc']=datetime.now(timezone.utc).isoformat()
report['static_and_csv_passed']=all(x['status']=='pass' for x in report['syntax']) and all(x['exit_code']==0 for x in report['steps']) and actual==expected
report['integration_passed']=all(x['passed'] for x in report['imports'])
(OUT/'VALIDATION_CURRENT.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['static_and_csv_passed'] else 1)
