import yaml
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

# Load YAML data
with open("papers.modified.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader("."))  # Load from current directory
template = env.get_template("Readme.md.jinja")

# Render template with data
output = template.render(**data)

# Save the output to a Markdown file
Path("Readme.md").write_text(output, encoding="utf-8")
