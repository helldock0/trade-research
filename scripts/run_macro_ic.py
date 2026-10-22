"""Phase A macro IC check: macro pair bias vs forward FRED FX returns.

    python scripts/run_macro_ic.py                       # fetch what is missing, run, write reports
    python scripts/run_macro_ic.py --no-download         # cached data only
    python scripts/run_macro_ic.py --start 2011-01-01 --tag since2011
    python scripts/run_macro_ic.py --variant hikes --tag hikes   # one variant from the config
    python scripts/run_macro_ic.py --validate                    # baseline vs variants, dev vs holdout

Writes reports/macro_ic.json and reports/macro_ic.md (or macro_ic_<tag>.*).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ROOT as PROJECT_ROOT, load_strategy_config
from src.data.macro import download_macro, load_macro_panel
from src.research.macro_compare import variant_config
from src.research.macro_ic import (
    HORIZONS,
    fetch_fx,
    load_fx,
    macro_cache_dir,
    render_markdown,
    render_validation_markdown,
    run_macro_ic,
    validate_variants,
    write_macro_ic_report,
    write_validation_report,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Macro IC check (Phase A): does the pair bias predict FX returns?")
    parser.add_argument("--config", default="strategy_macro_swing.yaml")
    parser.add_argument("--start", default=None, help="first decision date (default: data.start in the config)")
    parser.add_argument("--end", default=None, help="last decision date (default: data.end in the config)")
    parser.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    parser.add_argument("--no-download", action="store_true", help="use cached macro and FX series only")
    parser.add_argument("--force-fx", action="store_true", help="re-download the FRED FX series")
    parser.add_argument("--tag", default=None, help="report name suffix")
    parser.add_argument("--variant", default=None, help="run one variant from the config's `variants:`")
    parser.add_argument("--validate", action="store_true", help="baseline vs variants on the config's dev / holdout windows")
    args = parser.parse_args(argv)

    base_cfg = load_strategy_config(args.config)
    cfg = variant_config(base_cfg, args.variant) if args.variant else base_cfg
    val = base_cfg.get("validation") or {}
    configs = {n: variant_config(base_cfg, n) for n in [val.get("baseline"), *val.get("variants", [])] if n} if args.validate else {}
    macro_cfg = cfg["macro"]
    cache_dir = macro_cache_dir(macro_cfg)
    pairs = list(dict.fromkeys(p.upper() for c in [cfg, *configs.values()] for p in c["universe"]))
    if not args.no_download:
        status = {"macro": download_macro(macro_cfg), "fx": fetch_fx(pairs, cache_dir, force=args.force_fx)}
        errors = {k: v for grp in status.values() for k, v in grp.items() if "error" in v}
        if errors:
            print(json.dumps(errors, indent=2), file=sys.stderr)

    panel = load_macro_panel(macro_cfg)
    fx = load_fx(pairs, cache_dir)
    if not panel or not fx:
        print("No cached macro or FX series; run without --no-download first.", file=sys.stderr)
        sys.exit(1)

    if args.validate:
        res = validate_variants(panel, configs, fx, val["baseline"], tuple(val["dev"]), tuple(val["holdout"]))
        paths = write_validation_report(res, PROJECT_ROOT / "reports")
        print(render_validation_markdown(res))
        print("Reports:")
        for k, p in paths.items():
            print(f"  {k}: {p}")
        return

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    result = run_macro_ic(panel, cfg, fx, horizons=horizons, start=args.start, end=args.end)
    paths = write_macro_ic_report(result, PROJECT_ROOT / "reports", tag=args.tag)
    print(render_markdown(result))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
