"""
Converts a yaml file to Markdown using Jinja2
"""
import datetime
import itertools
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml
from jinja2 import Environment, FileSystemLoader


def create_date_histograms(papers):
    def do_plot():
        dates = [datetime.datetime.strptime(paper["year"], "%Y-%m-%d") for paper in papers]
        now = datetime.datetime.now()


        # Prepare for yearly histogram
        years = [date.year for date in dates]
        year_counts = Counter(years)

        # Get a full list of years in range
        first_year = min(years)
        last_year = max(years)
        years_range = range(first_year, last_year + 1)

        # Prepare for quarterly histogram
        quarters = [(date.year, (date.month - 1) // 3 + 1) for date in dates]
        quarter_labels = [f"{year}-Q{quarter}" for year, quarter in quarters]
        quarter_counts = Counter(quarter_labels)

        # Determine full range of quarters
        current_year = now.year
        current_quarter = (now.month - 1) // 3 + 1
        full_quarters = [
            f"{year}-Q{q}"
            for year, q in itertools.product(range(first_year, last_year + 1), range(1, 5))
            if year < current_year or q <= current_quarter
        ]

        # Plotting
        fig, ax_quarter = plt.subplots(figsize=(12, 4))

        year_position_map = {}
        tick_positions = []
        current_position = 0.5

        # Set the yearly bars and create a mapping for their positions
        for year in years_range:
            count = 0
            for q in range(1, 5):
                if f"{year}-Q{q}" in full_quarters:
                    count += 1
                    tick_positions.append(current_position)
                    current_position += 1
            year_position_map[year] = (tick_positions[-1] - (count / 2), count)

        # Yearly bars in the background
        ax_year = ax_quarter.twinx()
        for year, (position, width) in year_position_map.items():
            ax_year.bar(position, year_counts.get(year, 0), width=width, alpha=0.2, label='Yearly Counts', color=color_years,
                        align='center')

        # plt.figure(figsize=(10, 5))
        ax_quarter.bar(full_quarters, [quarter_counts.get(quarter, 0) for quarter in full_quarters], width=0.6, color=color_quarters)
        ax_quarter.set_title('Publications per Quarter',  fontweight='bold')
        ax_quarter.set_xlabel('Quarter')
        ax_quarter.set_ylabel('Number of Publications')

        ax_quarter.set_xticks(range(len(full_quarters)))
        ax_quarter.set_xticklabels(full_quarters, rotation=45)
        ax_quarter.tick_params(axis='y', labelcolor=color_quarters)
        ax_year.tick_params(axis='y', labelcolor=color_years)

        ax_year.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Add vertical lines after each Q4
        for i in range(len(full_quarters)):
            if full_quarters[i].endswith('Q4'):
                ax_quarter.axvline(x=i + 0.5, color=color_years, linestyle='--')


    # Light Mode Figure
    color_years = "#00509B"
    color_quarters = "#C8D317"

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "Arial",
        "axes.facecolor": "none",  # Transparent background
        "savefig.transparent": True,  # Transparent when saving
    })
    do_plot()

    plt.tight_layout()
    plt.savefig("assets/papers_per_quarter_light.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig("assets/papers_per_quarter_light.svg", bbox_inches='tight', transparent=True)

    # Dark Mode Figure
    color_years = "#99B9D8"
    color_quarters = "#C8D317"
    plt.rcParams.update({
        "axes.facecolor": "none",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
    })


    do_plot()
    plt.tight_layout()
    plt.savefig("assets/papers_per_quarter_dark.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig("assets/papers_per_quarter_dark.svg", bbox_inches='tight', transparent=True)


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
