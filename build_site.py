#!/usr/bin/env python3
"""
Static site generator for papers collection.
Reads papers.yaml and generates index.html and stats.html with embedded data.
"""

import argparse
import datetime
import json
from collections import Counter, defaultdict
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

        # Add "code" tag for filtering when multiple code-sources are available
        for url in paper.get("urls", []):
            if isinstance(url, str) and url.startswith("code") and url != "code":
                tags.append("code")
                break
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


def calculate_statistics(papers):
    """Calculate all statistics for the papers collection."""
    # Sort papers by date for cumulative calculations
    sorted_papers = sorted(papers, key=lambda p: p['year'])

    # Publication Timeline Statistics
    papers_by_year = Counter()
    papers_by_quarter = Counter()
    papers_by_month = Counter()  # Changed to Counter for easier processing
    papers_by_month_heatmap = defaultdict(lambda: defaultdict(int))

    # Generate year boundaries for charts
    all_years = sorted(set(p['year'][:4] for p in papers))
    year_boundaries = []
    for year in all_years[1:]:  # Skip first year
        # First day of each year (except the first year in dataset)
        year_boundaries.append(f"{year}-01-01")

    # Fine-grained cumulative data
    cumulative_timeline = []
    all_authors_seen = set()

    for i, paper in enumerate(sorted_papers):
        year = paper['year'][:4]
        papers_by_year[year] += 1

        # Extract month if available (format: YYYY-MM-DD)
        full_date = paper['year'][:10] if len(paper['year']) >= 10 else f"{year}-01-01"

        if len(paper['year']) >= 7:
            month = paper['year'][5:7]
            year_month = f"{year}-{month}"
            quarter = f"{year}-Q{(int(month) - 1) // 3 + 1}"
            papers_by_quarter[quarter] += 1
            papers_by_month[year_month] += 1
            papers_by_month_heatmap[year][int(month)] += 1

        # Track unique authors for this paper
        for author in paper['authors']:
            author_name = f"{author[0]} {author[1]}"
            all_authors_seen.add(author_name)

        # Add to cumulative timeline
        cumulative_timeline.append({
            'date': full_date,
            'paper_count': i + 1,
            'author_count': len(all_authors_seen)
        })

    # Tag Statistics
    tag_counter = Counter()
    special_tags_counter = {'code': 0, 'arxiv': 0, 'project_page': 0}

    for paper in papers:
        # Count special tags from URLs
        if paper.get('urls'):
            for key in special_tags_counter:
                if paper['urls'].get(key):
                    special_tags_counter[key] += 1

        # Count regular tags
        if paper.get('tags'):
            for tag in paper['tags']:
                if tag not in special_tags_counter.keys():
                    tag_counter[tag] += 1

    # Resource Availability
    resource_stats = {
        'code': special_tags_counter['code'],
        'arxiv': special_tags_counter['arxiv'],
        'project_page': special_tags_counter['project_page'],
        'paper': sum(1 for p in papers if p.get('urls', {}).get('paper')),
    }

    # Papers with at least one resource
    papers_with_resources = sum(1 for p in papers
                                if p.get('urls') and any(p['urls'].values()))
    resource_stats['none'] = len(papers) - papers_with_resources

    # Author Statistics
    author_counter = Counter()
    first_author_counter = Counter()

    for paper in papers:
        for i, author in enumerate(paper['authors']):
            author_name = f"{author[0]} {author[1]}"
            author_counter[author_name] += 1

            if i == 0:  # First author
                first_author_counter[author_name] += 1

    # General Statistics
    total_authors = sum(len(p['authors']) for p in papers)
    avg_coauthors = total_authors / len(papers) if papers else 0

    # Abstract lengths
    abstract_lengths_words = [len(p['abstract'].split()) for p in papers if p.get('abstract')]
    avg_abstract_length_words = sum(abstract_lengths_words) / len(
        abstract_lengths_words) if abstract_lengths_words else 0

    abstract_lengths_chars = [len(p['abstract']) for p in papers if p.get('abstract')]
    avg_abstract_length_chars = sum(abstract_lengths_chars) / len(
        abstract_lengths_chars) if abstract_lengths_chars else 0

    # Prepare heatmap data for months (years as rows, months as columns)
    month_heatmap = []
    all_years = sorted(papers_by_month_heatmap.keys())
    for year in all_years:
        row = {'year': year, 'months': []}
        for month in range(1, 13):
            row['months'].append({
                'month': month,
                'count': papers_by_month_heatmap[year].get(month, 0)
            })
        month_heatmap.append(row)

    current_date = datetime.datetime.now(datetime.UTC)

    # Resource percentages
    total = len(papers)
    resource_percentages = {
        'code': round(resource_stats['code'] / total * 100, 1) if total > 0 else 0,
        'arxiv': round(resource_stats['arxiv'] / total * 100, 1) if total > 0 else 0,
        'project_page': round(resource_stats['project_page'] / total * 100, 1) if total > 0 else 0,
        'paper': round(resource_stats['paper'] / total * 100, 1) if total > 0 else 0,
    }

    # Add to return statement:
    return {
        # Timeline
        'papers_by_year': dict(sorted(papers_by_year.items())),
        'papers_by_quarter': dict(sorted(papers_by_quarter.items())),
        'papers_by_month': dict(sorted(papers_by_month.items())),
        'cumulative_timeline': cumulative_timeline,  # Fine-grained cumulative data
        'month_heatmap': month_heatmap,
        'year_boundaries': year_boundaries,

        # Tags
        'top_tags': tag_counter.most_common(20),
        'special_tags': special_tags_counter,

        # Resources
        'resource_stats': resource_stats,

        # Authors
        'top_authors': author_counter.most_common(20),

        # General
        'total_papers': len(papers),
        'total_unique_authors': len(author_counter),
        'avg_coauthors': round(avg_coauthors, 2),
        'avg_abstract_length_words': round(avg_abstract_length_words, 1),
        'avg_abstract_length_chars': round(avg_abstract_length_chars, 1),
        'resource_percentages': resource_percentages,
    }


def get_heatmap_color(count):
    if count == 0:
        return 'var(--bg-tertiary)'
    intensity = min(count / 5, 1)  # Max at 5 papers
    r = round(16 + (239 - 16) * (1 - intensity))
    g = round(174 + (108 - 174) * (1 - intensity))
    b = round(72 + (50 - 72) * (1 - intensity))
    return f'rgb({r}, {g}, {b})'


def generate_html(papers, template_dir: str, output_folder: Path):
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_dir))

    now = datetime.datetime.now(datetime.UTC)
    statistics = calculate_statistics(papers)
    print(f"✓ Calculated statistics:\n"
          f"    > {statistics['total_papers']} papers\n"
          f"    > {statistics['total_unique_authors']} unique authors")

    data = {
        "papers_json": paper_json_to_str(papers),
        "date": now,
        "stats": statistics,
        "get_heatmap_color": get_heatmap_color
    }

    # Create output folder
    output_folder.mkdir(exist_ok=True, parents=True)
    print(f"✓ Created output directory: {output_folder}")

    # Index rendering
    index_template = env.get_template("index.html.jinja")
    output_index = index_template.render(**data)
    (output_folder / "index.html").write_text(
        f"<!-- This file was automatically created on {now :%Y-%m-%d %H:%M:%S} UTC. Any manual changes will be lost! -->\n{output_index}",
        encoding="utf-8"
    )
    print(f"✓ Generated index.html with {len(papers)} papers")

    # Stats rendering
    stats_template = env.get_template("stats.html.jinja")
    output_stats = stats_template.render(**data)
    (output_folder / "stats.html").write_text(
        f"<!-- This file was automatically created on {now :%Y-%m-%d %H:%M:%S} UTC. Any manual changes will be lost! -->\n{output_stats}",
        encoding="utf-8"
    )
    print(f"✓ Generated stats.html with {len(papers)} papers")


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
        default=script_dir / "assets",
        help="Path to HTML templates folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir,
        help="Output folder",
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
