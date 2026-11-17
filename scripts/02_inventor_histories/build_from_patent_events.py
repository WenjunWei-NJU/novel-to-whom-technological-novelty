"""Run the separate event-based history implementation for data/schema validation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from study.build import build_all
from study.io import connect, load_config
from study.preflight import check_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"Required configuration file does not exist: {args.config}")
    config = load_config(args.config)
    report = check_inputs(config)
    if report["problems"]:
        parser.error(str(report["problems"]))
    args.output.mkdir(parents=True, exist_ok=False)
    with connect(args.output, config) as database:
        build_all(database, config, args.output)


if __name__ == "__main__":
    main()
