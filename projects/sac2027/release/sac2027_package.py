#!/usr/bin/env python3
"""Package only a verified paper; rebuild the standalone source before delivery."""

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
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from pypdf import PdfReader

REPO = REPO_ROOT
PAPER = REPO / "paper/sac2027/v14_review"
OUT = REPO / "output/pdf"
NAME = "SAC2027_MLA_v14_patch_graph_2026-10-08"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    qa = json.loads((PAPER / "qa/final_validation.json").read_text())
    assert all(v["passed"] for v in qa["checks"].values())
    assert qa["pdf_sha256"] == sha(PAPER / "main.pdf")
    assert (REPO / "reports/sac2027_patch_v14/final_verification.json").exists()
    files = [
        PAPER / n
        for n in (
            "main.tex",
            "refs.bib",
            "main.bbl",
            "acmart.cls",
            "ACM-Reference-Format-citation-order.bst",
            "ACM-Reference-Format.bst",
            "TEMPLATE_README.txt",
            "README_KO.md",
            "CHANGES_KO.md",
            "REVIEW_RESPONSE_KO.md",
            "RESULTS_KO.md",
            "qa/final_validation.json",
            "qa/citation_order.json",
        )
    ]
    files += sorted((PAPER / "tables").glob("*.tex"))
    files += [
        PAPER / "figures" / n
        for n in ("fig_pipeline_clean.pdf", "fig_rq1_effects.pdf", "fig_contrast.pdf")
    ]
    files += sorted((PAPER / "tables").glob("*.csv"))
    files += sorted((PAPER / "figures").glob("*.csv"))
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sac2027-v14-source-") as temp:
        stage = Path(temp)
        for source in files:
            target = stage / source.relative_to(PAPER)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        original = {str(p.relative_to(stage)): sha(p) for p in stage.rglob("*") if p.is_file()}
        build = subprocess.run(
            ["/opt/homebrew/bin/tectonic", "-C", "--keep-logs", "--keep-intermediates", "main.tex"],
            cwd=stage,
            capture_output=True,
            text=True,
        )
        (PAPER / "qa/package_build.log").write_text(build.stdout + "\n" + build.stderr)
        assert build.returncode == 0, build.stderr[-3000:]
        for name, digest in original.items():
            assert sha(stage / name) == digest, ("source changed during build", name)
        src = PdfReader(PAPER / "main.pdf")
        rebuilt = PdfReader(stage / "main.pdf")
        assert len(src.pages) == len(rebuilt.pages) <= 10
        same_text = all(
            a.extract_text() == b.extract_text() for a, b in zip(src.pages, rebuilt.pages)
        )
        assert same_text
        same_pages = all(
            a.get_contents().get_data() == b.get_contents().get_data()
            for a, b in zip(src.pages, rebuilt.pages)
        )
        assert same_pages
        assert "Overfull \\hbox" not in (stage / "main.log").read_text()
        rebuilt_sha = sha(stage / "main.pdf")
        shutil.copy2(PAPER / "main.pdf", OUT / (NAME + ".pdf"))
        archive = OUT / "SAC2027_MLA_v14_source_2026-10-08.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for source in files:
                z.write(source, str(source.relative_to(PAPER)))
            z.write(PAPER / "main.pdf", NAME + ".pdf")
            z.writestr("SOURCE_MANIFEST.json", json.dumps(original, indent=2) + "\n")
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name, digest in original.items():
                assert hashlib.sha256(z.read(name)).hexdigest() == digest
        result = dict(
            pages=len(src.pages),
            standalone_build=True,
            text_identical=same_text,
            page_content_identical=same_pages,
            rebuilt_pdf_sha256=rebuilt_sha,
            original_pdf_sha256=sha(PAPER / "main.pdf"),
            pdf_bytes_identical=rebuilt_sha == sha(PAPER / "main.pdf"),
            zip_sha256=sha(archive),
            source_files=len(files),
            source_manifest=original,
        )
        (PAPER / "qa/package_validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "source_manifest"}, indent=2))
        print(OUT / (NAME + ".pdf"))
        print(archive)


if __name__ == "__main__":
    main()
