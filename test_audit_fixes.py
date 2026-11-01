"""Unit/contract tests, not GPU, dependency integration or historical reproduction."""
import ast
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from collections import defaultdict
from unittest.mock import patch
import numpy as np
from audit_support import HISTORY_COLUMNS, read_training_history, json_native, discover_rtmdet_config

HERE = Path(__file__).resolve().parent

def csv_fixture(path):
    with path.open('w', newline='', encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['epoch']+list(HISTORY_COLUMNS.values()))
        writer.writeheader()
        for epoch in range(2):
            writer.writerow({'epoch':epoch,**{key:0.1+epoch/10 for key in HISTORY_COLUMNS.values()}})

class AuditFixes(unittest.TestCase):
    def test_csv_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'results.csv';csv_fixture(path)
            result=read_training_history(path)
            self.assertEqual(result['mAP50'],[0.1,0.2])
            path.write_text('epoch\n0\n',encoding='utf8')
            with self.assertRaises(ValueError):read_training_history(path)

    def test_json_numpy(self):
        value={'f1':np.array([0.25,0.5]),'p':np.float32(0.75)}
        self.assertEqual(json.loads(json.dumps(json_native(value),allow_nan=False)),{'f1':[0.25,0.5],'p':0.75})

    def test_config_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.py';path.write_text('model = {}',encoding='utf8')
            with patch.dict(os.environ,{'MMDET_RTMDET_CONFIG':str(path)}):
                self.assertEqual(discover_rtmdet_config(),path.resolve())
            with patch.dict(os.environ,{'MMDET_RTMDET_CONFIG':str(path)+'-absent'}):
                with self.assertRaises(FileNotFoundError):discover_rtmdet_config()

    def test_actual_trainer_method_csv_contract(self):
        # Compile the real class without importing optional CV/GPU packages.
        source=ast.parse((HERE/'unified_trainer.py').read_text(encoding='utf8'))
        cls=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name=='UnifiedTrainer')
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);path=tmp/'actual_run/results.csv';path.parent.mkdir();csv_fixture(path)
            class FakeYOLO:
                def __init__(self,weight):self.trainer=type('Trainer',(),{'csv':path})()
                def train(self,**kwargs):return object()  # deliberately no .history
            scope=dict(Path=Path,RESULTS_ROOT=tmp,TRAIN_CONFIG={},defaultdict=defaultdict,
                       YOLO=FakeYOLO,json=json,read_training_history=read_training_history,json_native=json_native)
            exec(compile(ast.Module(body=[cls],type_ignores=[]),'unified_trainer.py','exec'),scope)
            trainer=scope['UnifiedTrainer']('absent.yaml');trainer._plot_history=lambda:None
            trainer.train()
            actual=json.loads((trainer.base_dir/'logs/training_history.json').read_text())
            self.assertEqual(actual['recall'],[0.1,0.2])
            for name in ['rtdetr','rtmdet']:
                with self.assertRaises(NotImplementedError):scope['UnifiedTrainer']('absent.yaml',model_type=name)

    def test_rtdetr_active_ast(self):
        tree=ast.parse((HERE/'train_rtdetr_l_6gb.py').read_text(encoding='utf8'))
        self.assertTrue(any(isinstance(n,ast.Import) and any(a.name=='os' for a in n.names) for n in tree.body))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id=='YOLO' for n in calls))
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='RTDETR' for n in calls),3)
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr=='load' and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' for n in calls))

if __name__=='__main__':unittest.main(verbosity=2)
