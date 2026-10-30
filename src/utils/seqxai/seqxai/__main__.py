"""CLI:  python -m seqxai --setup my_setup.py --predictions val.csv --out xai_out [--stages a,b]

my_setup.py must define
    build_model()   -> ModularModel with weights loaded
    build_adapter() -> seqxai.DataAdapter
and may define
    CONFIG = dict(...)   # any seqxai.Config field; 'selection' may be a dict of SelectionRule fields
"""
import argparse
import importlib.util

from .pipeline import STAGES, Config, Pipeline
from .selection import SelectionRule


def main():
    ap = argparse.ArgumentParser(prog='seqxai')
    ap.add_argument('--setup', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out', default='xai_out')
    ap.add_argument('--stages', default=','.join(STAGES))
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('xai_setup', a.setup)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    kw = dict(getattr(m, 'CONFIG', {}))
    if isinstance(kw.get('selection'), dict):
        kw['selection'] = SelectionRule(**kw['selection'])
    kw['out_dir'] = a.out
    if a.device:
        kw['device'] = a.device
    cfg = Config(**kw)
    model = m.build_model()
    Pipeline(model, m.build_adapter(), a.predictions, cfg).run(a.stages.split(','))


if __name__ == '__main__':
    main()
