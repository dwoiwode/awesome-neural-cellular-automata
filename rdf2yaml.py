import datetime
import re
import subprocess
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
            continue
        title = item.find("dc:title", ns)
        year = item.find("dc:date", ns)
        year2 = item.find("dcterms:dateSubmitted", ns)
        abstract = item.find("dcterms:abstract", ns)

        authors = []
        authors_seq = item.find("bib:authors/rdf:Seq", ns)
        if authors_seq:
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
        urls = {
            "paper": "",
            "code": "",
            "project_page": ""
        }

        title_text = title.text if title is not None else "Unknown Title"
        print(year, year2, end=" -> ")
        year = unify_year(year, year2)
        print(year)
        id_text = generate_id(title_text, year, authors)

        papers.append({
            "id": id_text,
            "title": title_text,
            "year": year,
            "authors": authors if authors else [],
            "urls": urls,
            "thumbnail": create_thumbnail(pdf_path, id_text).as_posix(),
            "abstract": abstract.text if abstract is not None else "",
            # "paper": pdf_path.relative_to(Path(__file__).parent).as_posix() if pdf_path else None,
        })

    return papers



def generate_id(title, year, authors):
    id_ =  year + re.sub(r'[^a-zA-Z0-9]', '', title.lower().replace(" ", "")[:10])
    if len(authors) > 0:
        id_ += "_" + authors[0][1].lower()
    return id_


def create_thumbnail(pdf_path, paper_id) -> Path:
    output_thumb = Path("assets/thumbnails/") / f"{paper_id}.jpg"
    if not output_thumb.exists():
        if pdf_path is not None:
            output_thumb.parent.mkdir(parents=True, exist_ok=True)

            subprocess.run([
                "magick", "convert", "-background","white","-flatten", f"{pdf_path}[0]", "-resize", "300x400", str(output_thumb)
            ], check=True)

        else:
            output_thumb.write_bytes(Path("placeholder_thumbnail.jpg").read_bytes())

    return output_thumb


def export_yaml(papers, output_file="papers.auto.yaml"):
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump({
            "date": f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}",
            "papers":papers,
        }, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )


def main():
    path = r"C:\Users\woiwode\Desktop\tmp\NCA_rdf\NCA_rdf.rdf"
    rdf_file = Path(path)
    papers = parse_rdf(rdf_file)
    print(f"Found {len(papers)} papers in {rdf_file}")
    papers = sorted(papers, key=lambda p: p["year"])
    export_yaml(papers)
    print(f"Exported {len(papers)} papers to papers.auto.yaml")


if __name__ == "__main__":
    main()
