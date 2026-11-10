
import argparse
import importlib
from pathlib import Path
import yaml

def set_dotted(cfg: dict, key: str, value) -> None:
    *parents, last = key.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[last] = value



def parse_value(v: str): 
    for cast in (int, float):  # PyYAML reads '1e-4' as a string, so try numbers first
        try:
            return cast(v)
        except ValueError:
            pass
    return yaml.safe_load(v)

def load_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="overrides data.init_args.num_workers")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value, e.g. model.init_args.lr=1e-4")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for item in args.overrides:
        key, value = item.split("=", 1)
        set_dotted(cfg, key, parse_value(value))
    '''
    if args.num_workers is not None:
        cfg.setdefault("data", {}).setdefault("init_args", {})["num_workers"] = args.num_workers'''
    cfg['num_workers'] = args.num_workers
    return cfg
