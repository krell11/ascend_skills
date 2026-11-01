from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .build import from_config_json, from_hf, from_ir
from .render.html import write_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model-arch-viz",
        description=(
            "Visualize model architecture and per-layer math from HF config "
            "or a paper-derived IR — without downloading weights."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_hf = sub.add_parser("from-hf", help="Build from Hugging Face config.json only")
    p_hf.add_argument("model_id", help="Hub id, local dir, or path to config.json")
    p_hf.add_argument(
        "--meta",
        action="store_true",
        help="Also fetch safetensors index / headers (no weight payloads)",
    )
    p_hf.add_argument("--revision", default=None)
    p_hf.add_argument("-o", "--output", required=True, help="Output HTML path")
    p_hf.add_argument(
        "--dump-ir",
        default=None,
        help="Optional path to write the architecture IR JSON",
    )

    p_ir = sub.add_parser("from-ir", help="Build from paper / hand-authored IR JSON")
    p_ir.add_argument("ir_path", help="Path to .arch.json")
    p_ir.add_argument("-o", "--output", required=True, help="Output HTML path")

    p_cfg = sub.add_parser("from-config", help="Build from a local config.json file")
    p_cfg.add_argument("config_path", help="Path to config.json")
    p_cfg.add_argument("-o", "--output", required=True, help="Output HTML path")
    p_cfg.add_argument("--dump-ir", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "from-hf":
        arch = from_hf(args.model_id, meta=args.meta, revision=args.revision)
        if args.dump_ir:
            Path(args.dump_ir).write_text(
                json.dumps(arch.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    elif args.cmd == "from-ir":
        arch = from_ir(args.ir_path)
    elif args.cmd == "from-config":
        arch = from_config_json(args.config_path)
        if args.dump_ir:
            Path(args.dump_ir).write_text(
                json.dumps(arch.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    else:
        parser.error(f"unknown command {args.cmd}")
        return 2

    out = write_html(arch, args.output, standalone=True)
    print(f"Wrote {out}  ({arch.name})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
