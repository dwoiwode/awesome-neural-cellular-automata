"""
Converts a Zotero rdf structure to a automatically generated yaml. Also generates thumbnails for each entry.

Easiest way to add new papers to this list:
- Export selected entries from Zotero as rdf
- convert the rdf structure to yaml using this script
- Update `papers.yaml` using the content from `papers.auto.yaml`.
"""
import datetime
import re
import subprocess
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from xml.etree.ElementTree import Element

from dateutil import parser

import yaml


def unify_year(year: Optional[Element], year2:Optional[Element]) -> str:
    if year is None:
        year = year2
    if year is None:
        return "Unknown"
    year = year.text

    try:
        # Try to parse the date using dateutil's parser, which is very flexible
        dt = parser.parse(year, fuzzy=True)
        # Return in the desired format
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return "Unknown"


def parse_rdf(rdf_file:Path):
    rdf_file = Path(rdf_file)
    tree = ET.parse(rdf_file)
    root = tree.getroot()
    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'z': 'http://www.zotero.org/namespaces/export#',
        'foaf': 'http://xmlns.com/foaf/0.1/',
        'bib': 'http://purl.org/net/biblio#',
        'dcterms': 'http://purl.org/dc/terms/',
        'prism': 'http://prismstandard.org/namespaces/1.2/basic/',
        'link': 'http://purl.org/rss/1.0/modules/link/'
    }

    attachments =  {
        list(item.attrib.values())[0]: item
        for item in root.findall(".//z:Attachment", ns)
    }

    papers = []
    for item in root:
        if "Attachment" in item.tag:
            # We already got a list of attachments before
            continue
        title = item.find("dc:title", ns)
        year = item.find("dc:date", ns)
        year2 = item.find("dcterms:dateSubmitted", ns)
        abstract = item.find("dcterms:abstract", ns)

        authors = []
        authors_seq = item.find("bib:authors/rdf:Seq", ns)
        if authors_seq is not None:
            for author in authors_seq.findall("rdf:li/foaf:Person", ns):
                surname = author.find("foaf:surname", ns)
                given_name = author.find("foaf:givenName", ns)
                if surname is not None and given_name is not None:
                    authors.append([given_name.text, surname.text])

        # Find associated PDF attachment
        link_ref = item.find("link:link", ns)
        pdf_path = None
        if link_ref is not None:
            attachment = attachments.get(list(link_ref.attrib.values())[0])
            if attachment is not None:
                resource = attachment.find("rdf:resource", ns)
                if resource is not None:
                    pdf_path = rdf_file.parent / list(resource.attrib.values())[0]

        # Find paper urls
        urls = {  # TODO: Currently only done manually to improve overall quality
            "project_page": "",
            "paper": "",
            "code": "",
        }

        title_text = title.text if title is not None else "Unknown Title"
        year = unify_year(year, year2)
        id_text = generate_id(title_text, year, authors)

        papers.append({
            "id": id_text,
            "title": title_text,
            "year": year,
            "authors": authors if authors else [],
            "urls": urls,
            "thumbnail": create_thumbnail(pdf_path, id_text).as_posix(),
            "abstract": abstract.text if abstract is not None else "",
        })

    return papers



def generate_id(title, year, authors):
    id_ =  year + re.sub(r'[^a-zA-Z0-9]', '', title.lower().replace(" ", "")[:10])
    if len(authors) > 0:
        id_ += "_" + authors[0][1].lower()
    return id_


def create_thumbnail(pdf_path: Path, paper_id) -> Path:
    output_thumb = Path("assets/thumbnails/") / f"{paper_id}.jpg"
    if not output_thumb.exists():
        if pdf_path is not None:
            try:
                output_thumb.parent.mkdir(parents=True, exist_ok=True)

                subprocess.run([
                    "magick", "convert", "-background","white","-flatten", f"{pdf_path}[0]", "-resize", "300x", str(output_thumb)
                ], check=True)
                return output_thumb
            except Exception as e:
                warnings.warn(f"Converting {pdf_path} to thumbnail failed: {e}")

        output_thumb.write_bytes(Path("assets/thumbnail_placeholder.jpg").read_bytes())

    return output_thumb


def export_yaml(papers, output_file="papers.auto.yaml"):
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump({
            "date": f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}",
            "papers":papers,
        }, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )


def main():
    path = r"C:\Users\woiwode\Desktop\tmp\Exportierte Einträge\Exportierte Einträge.rdf"
    rdf_file = Path(path)
    papers = parse_rdf(rdf_file)
    print(f"Found {len(papers)} papers in {rdf_file}")
    papers = sorted(papers, key=lambda p: p["year"])
    export_yaml(papers)
    print(f"Exported {len(papers)} papers to papers.auto.yaml")


if __name__ == "__main__":
    main()
