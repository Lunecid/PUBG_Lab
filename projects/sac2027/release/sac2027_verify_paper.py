#!/usr/bin/env python3
"""Verify the completed v14 source/PDF and trace its experimental inputs."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import hashlib
import json
import re
from pathlib import Path
from pypdf import PdfReader

REPO = REPO_ROOT
PAPER = REPO / "paper/sac2027/v14_review"
BASE = REPO / "reports/sac2027_patch_v14"


def main():
    checks = {}

    def check(name, ok, detail=None):
        checks[name] = dict(passed=bool(ok), detail=detail)
        assert ok, (name, detail)

    check("all_experiments_verified", (BASE / "final_verification.json").is_file())
    source = (PAPER / "main.tex").read_text()
    for item in re.findall(r"\\input\{([^}]+)\}", source):
        source = source.replace(r"\input{" + item + "}", (PAPER / (item + ".tex")).read_text())
    check("no_pending_result_marker", "TUNING_HISTORY_POPULATION_GRAPH_RESULTS" not in source)
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", source)
    check("unique_labels", len(labels) == len(set(labels)))
    check("crossrefs_resolve", set(refs) <= set(labels))
    abstract = source.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
    check("abstract_100_to_200_words", 100 <= len(abstract.split()) <= 200, len(abstract.split()))
    keywords = re.search(r"\\keywords\{([^}]+)\}", source).group(1).split(",")
    check("at_most_five_keywords", len(keywords) <= 5, len(keywords))
    citations = []
    for group in re.findall(r"\\citation\{([^}]+)\}", (PAPER / "main.aux").read_text()):
        for key in group.split(","):
            if key not in citations:
                citations.append(key)
    bibliography = re.findall(
        r"\\bibitem\[.*?\]%\s*\{([^}]+)\}", (PAPER / "main.bbl").read_text(), re.S
    )
    check(
        "first_citation_order",
        citations == bibliography,
        dict(first_appearance=citations, bibliography=bibliography),
    )
    old = (REPO / "paper/sac2027/v13_v12_science_widom/main.tex").read_text()
    original_keys = {
        x for g in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", old) for x in g.split(",")
    }
    check("original_references_retained", original_keys <= set(citations))
    for name, digest in json.loads((PAPER / "qa/v13_source_manifest.json").read_text()).items():
        check(
            "original_v13_preserved_" + name,
            hashlib.sha256(
                (REPO / "paper/sac2027/v13_v12_science_widom" / name).read_bytes()
            ).hexdigest()
            == digest,
        )
    log = (PAPER / "main.log").read_text()
    check(
        "no_undefined_or_duplicate_refs",
        not re.search(
            r"(?:Citation|Reference).*undefined|undefined references|multiply defined", log
        ),
    )
    check("no_horizontal_overflow", r"Overfull \hbox" not in log)
    reader = PdfReader(PAPER / "main.pdf")
    check("page_limit", len(reader.pages) <= 10, len(reader.pages))
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    check("no_replacement_glyph", "\ufffd" not in text)
    check("anonymous_pdf", not any(x in text for x in ("lunecid", "/Users/", "TUNING_HISTORY_")))
    check("anonymous_metadata", not reader.metadata.get("/Author", ""))
    destinations = reader.named_destinations
    missing = []
    n_links = 0
    for page in reader.pages:
        annotations = page.get("/Annots", [])
        if hasattr(annotations, "get_object"):
            annotations = annotations.get_object()
        for ann in annotations:
            ann = ann.get_object()
            act = ann.get("/A")
            act = act.get_object() if act else None
            dest = act.get("/D") if act and act.get("/S") == "/GoTo" else ann.get("/Dest")
            if isinstance(dest, str):
                n_links += 1
                if dest not in destinations:
                    missing.append(dest)
    check("pdf_internal_links", not missing, dict(count=n_links, missing=missing))
    report = dict(
        checks=checks,
        pages=len(reader.pages),
        references=len(citations),
        tables=len(re.findall(r"\\begin\{table\*?\}", source)),
        figures=len(re.findall(r"\\begin\{figure\*?\}", source)),
        pdf_sha256=hashlib.sha256((PAPER / "main.pdf").read_bytes()).hexdigest(),
        vertical_warnings=re.findall(r"Overfull \\vbox[^\n]*", log),
    )
    (PAPER / "qa/final_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (PAPER / "qa/citation_order.json").write_text(
        json.dumps(checks["first_citation_order"], indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
