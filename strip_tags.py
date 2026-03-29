from pathlib import Path
import argparse

import yaml


def parse_args():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Strip preliminary tags from papers.yaml"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "papers.full.yaml",
        help="Input papers YAML file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "papers.yaml",
        help="Output papers YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for paper in data.get("papers", []):
        tags = paper.get("tags")
        if not tags:
            paper.pop("tags", None)
            continue

        cleaned = [t for t in tags if isinstance(t, str) and not t.endswith("?")]

        if cleaned:
            paper["tags"] = cleaned
        else:
            paper.pop("tags", None)

    with args.output.open("w",encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

if __name__ == '__main__':
    main()
