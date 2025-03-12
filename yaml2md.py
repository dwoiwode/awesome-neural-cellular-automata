"""
Converts a yaml file to Markdown using Jinja2
"""
import datetime
import itertools
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from jinja2 import Environment, FileSystemLoader


def create_date_histograms(papers):
    dates = [datetime.datetime.strptime(paper["year"], "%Y-%m-%d") for paper in papers]

    # Prepare for yearly histogram
    years = [date.year for date in dates]
    year_counts = Counter(years)

    # Get a full list of years in range
    year_range = range(min(years), max(years) + 1)

    plt.figure(figsize=(10, 5))
    plt.bar(year_range, [year_counts.get(year, 0) for year in year_range], width=0.6)
    plt.title('Papers per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.xticks(year_range)
    plt.tight_layout()
    plt.savefig("assets/papers_per_year.png")

    # Prepare for quarterly histogram
    quarters = [(date.year, (date.month - 1) // 3 + 1) for date in dates]
    quarter_labels = [f"{year}-Q{quarter}" for year, quarter in quarters]
    quarter_counts = Counter(quarter_labels)

    # Determine full range of quarters
    first_year = min(years)
    last_year = max(years)
    now = datetime.datetime.now()
    current_year = now.year
    current_quarter = (now.month - 1) // 3 + 1
    full_quarters = [
        f"{year}-Q{q}"
        for year, q in itertools.product(range(first_year, last_year + 1), range(1, 5))
        if year < current_year or q <= current_quarter
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(full_quarters, [quarter_counts.get(quarter, 0) for quarter in full_quarters], width=0.6)
    plt.title('Publications per Quarter')
    plt.xlabel('Quarter')
    plt.ylabel('Number of Publications')
    plt.xticks(rotation=45)

    # Add vertical lines after each Q4
    for i in range(len(full_quarters)):
        if full_quarters[i].endswith('Q4'):
            plt.axvline(x=i + 0.5, color='gray', linestyle='--')

    plt.tight_layout()
    plt.savefig("assets/papers_per_quarter.png")


# Load YAML data
with Path("papers.yaml").open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

create_date_histograms(data["papers"])

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader("."))  # Load from current directory
template = env.get_template("Readme.md.jinja")

# Render template with data
output = template.render(**data)

# Save the output to a Markdown file
Path("Readme.md").write_text(
    f"<!-- This file was automatically created on {datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M:%S} UTC. Any manual changes will be lost! -->\n{output}",
    encoding="utf-8")
