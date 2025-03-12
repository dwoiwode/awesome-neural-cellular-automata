"""
Converts a yaml file to Markdown using Jinja2
"""
import datetime

import yaml
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

# Load YAML data
with Path("papers.yaml").open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader("."))  # Load from current directory
template = env.get_template("Readme.md.jinja")

# Render template with data
output = template.render(**data)

# Save the output to a Markdown file
Path("Readme.md").write_text(f"<!-- This file was automatically created on {datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M:%S} UTC. Any manual changes will be lost! -->\n{output}", encoding="utf-8")
