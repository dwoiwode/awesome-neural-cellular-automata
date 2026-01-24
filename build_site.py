#!/usr/bin/env python3
"""
Static site generator for papers collection.
Reads papers.yaml and generates index.html with embedded data.
"""

import argparse
import json
import datetime
from pathlib import Path

import yaml


def load_papers(yaml_path: Path):
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("papers", [])


def generate_html(papers, template_path: Path, output_path: Path):
    template = template_path.read_text(encoding="utf-8")

    papers_json = json.dumps(papers, ensure_ascii=False, indent=2)
    html = template.replace(
        "/* PAPERS_DATA_PLACEHOLDER */",
        f"const papersData = {papers_json};",
    )

    build_time = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    html = html.replace(
        "<!-- BUILD_TIMESTAMP -->",
        f"<!-- Built: {build_time} -->",
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"✓ Generated {output_path} with {len(papers)} papers")


def parse_args():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Generate static HTML page from papers.yaml"
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=script_dir / "papers.yaml",
        help="Path to papers YAML file",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=script_dir / "assets" / "template.html",
        help="Path to HTML template",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "index.html",
        help="Output HTML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.papers.exists():
        print(f"Error: {args.papers} not found")
        return 1

    if not args.template.exists():
        print(f"Error: {args.template} not found")
        return 1

    papers = load_papers(args.papers)
    generate_html(papers, args.template, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
