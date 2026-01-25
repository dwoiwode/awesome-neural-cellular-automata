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
from jinja2 import Environment, FileSystemLoader


def load_papers(yaml_path: Path):
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    papers = []
    for paper in data.get("papers", []):
        tags = []
        for tag in paper.get("tags", []):
            if isinstance(tag, str) and not tag.endswith("?"):
                tags.append(tag)
        if tags:
            paper["tags"] = tags
        else:
            paper.pop("tags", None)
        papers.append(paper)
    return papers

def paper_json_to_str(papers) -> str:
    s = "[\n"
    for paper in papers:
        s += f"{json.dumps(paper, ensure_ascii=False, separators=(",", ":"))},\n"
    return s + "]"

def generate_html(papers, template_path: Path, output_path: Path):
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader("assets"))  # Load from current directory
    template = env.get_template(template_path.name)

    data = {
        "papers_json": paper_json_to_str(papers),
        "date": datetime.datetime.now(datetime.UTC)
    }

    # Render template with data
    output = template.render(**data)

    # Save the output to a Markdown file
    output_path.write_text(
        f"<!-- This file was automatically created on {datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M:%S} UTC. Any manual changes will be lost! -->\n{output}",
        encoding="utf-8")

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
        default=script_dir / "assets" / "template.html.jinja",
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
