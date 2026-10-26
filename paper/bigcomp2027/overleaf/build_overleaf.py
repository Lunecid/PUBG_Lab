#!/usr/bin/env python3
"""Build one Overleaf project (zip) holding both versions of the BigComp 2027 paper.

    python3 paper/bigcomp2027/overleaf/build_overleaf.py [--stamp 2026-10-08]

Output: ``paper/bigcomp2027/overleaf/bigcomp2027_overleaf_<stamp>.zip``.  Unpack it to check, or upload it with Overleaf: New Project > Upload Project.

Project layout (everything at the zip root)::

    main.tex       English manuscript (default main document)
    main_ko.tex    Korean manuscript (revisions in blue)
    sections_ko/   the six Korean section files that main_ko.tex inputs
    figures/       the three PDF figures
    refs.bib       references for both versions
    IEEEtran.cls, IEEEtran.bst, latexmkrc

Both versions need XeLaTeX (fontspec, xetexko).  The masters name macOS fonts; here every font has a fallback chain
(Times New Roman > TeX Gyre Termes by name > by file name > Liberation Serif; AppleMyungjo > Noto Serif CJK KR > NanumMyeongjo > UnBatang;
Apple SD Gothic Neo > Noto Sans CJK KR > NanumGothic > UnDotum), so the project also compiles where the macOS fonts are missing.
Only the preamble and the file paths differ from the masters; the text is copied unchanged.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EN = REPO / "paper/bigcomp2027/en_v12/BIGCOMP2027_EN_v12_2026-10-07.tex"
KO = REPO / "docs/paper_writing/review/BIGCOMP2027_MANUSCRIPT_KO_IEEE_v12_2026-10-07.tex"
SEC = REPO / "docs/paper_writing"
BIB = SEC / "BIGCOMP2027_REFS_2026-10-07.bib"
FIGDIR = REPO / "paper/bigcomp2027/figures"
FIGURES = ("fig_pipeline_clean.pdf", "fig_rq1_effects.pdf", "fig_contrast.pdf")
CLS_DIR = REPO / "paper/bigcomp2027/en_v12"
KO_SECTIONS = (  # master name -> project name
    ("BIGCOMP2027_ABSTRACT_KO_v3_2026-10-07", "abstract"),
    ("BIGCOMP2027_INTRO_KO_v9_2026-10-07", "intro"),
    ("BIGCOMP2027_SEC2_RELATED_KO_v7_2026-10-07", "related"),
    ("BIGCOMP2027_SEC3_METHOD_KO_v12_2026-10-07", "method"),
    ("BIGCOMP2027_SEC4_RESULTS_KO_v7_2026-10-07", "results"),
    ("BIGCOMP2027_SEC5_DISCUSSION_KO_v3_2026-10-07", "discussion"),
)
MAGIC = "% !TEX program = xelatex\n"
LATEXMKRC = (
    "# Both documents need XeLaTeX (fontspec, xetexko). If Overleaf ignores this file, set Menu > Compiler to XeLaTeX.\n"
    "$pdflatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';\n"
)
LATIN_FONT = (
    "\\IfFontExistsTF{Times New Roman}{\\setmainfont{Times New Roman}}{%\n"
    "  \\IfFontExistsTF{TeX Gyre Termes}{\\setmainfont{TeX Gyre Termes}}{%\n"
    "    \\IfFontExistsTF{texgyretermes-regular.otf}{%\n"
    "      \\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]}{%\n"
    "      \\IfFontExistsTF{Liberation Serif}{\\setmainfont{Liberation Serif}}{}}}}\n"
)
HANGUL_FONTS = (
    "\\IfFontExistsTF{AppleMyungjo}{\\setmainhangulfont{AppleMyungjo}[AutoFakeBold=2.5]}{%\n"
    "  \\IfFontExistsTF{Noto Serif CJK KR}{\\setmainhangulfont{Noto Serif CJK KR}}{%\n"
    "    \\IfFontExistsTF{NanumMyeongjo}{\\setmainhangulfont{NanumMyeongjo}}{%\n"
    "      \\IfFontExistsTF{UnBatang}{\\setmainhangulfont{UnBatang}}{}}}}\n"
    "\\IfFontExistsTF{Apple SD Gothic Neo}{\\setsanshangulfont{Apple SD Gothic Neo}}{%\n"
    "  \\IfFontExistsTF{Noto Sans CJK KR}{\\setsanshangulfont{Noto Sans CJK KR}}{%\n"
    "    \\IfFontExistsTF{NanumGothic}{\\setsanshangulfont{NanumGothic}}{%\n"
    "      \\IfFontExistsTF{UnDotum}{\\setsanshangulfont{UnDotum}}{}}}}\n"
)


def edit(text: str, pairs: list[tuple[str, str]], what: str) -> str:
    for a, b in pairs:
        n = text.count(a)
        if n != 1:
            raise SystemExit(f"{what}: expected exactly one match, found {n}: {a[:80]!r}")
        text = text.replace(a, b)
    return text


def build(out_zip: Path) -> list[str]:
    english = edit(EN.read_text(encoding="utf-8"), [
        ("\\usepackage{fontspec}\n\\setmainfont{Times New Roman}\n",
         "% Times New Roman if installed, else TeX Gyre Termes or Liberation Serif (metric-compatible Times clones).\n"
         "\\usepackage{fontspec}\n" + LATIN_FONT),
        ("\\graphicspath{{../figures/}}", "\\graphicspath{{figures/}}"),
        ("\\bibliography{../../../docs/paper_writing/BIGCOMP2027_REFS_2026-10-07}", "\\bibliography{refs}"),
    ], "English master")
    korean = edit(KO.read_text(encoding="utf-8"), [
        ("\\usepackage{fontspec}\n\\usepackage{xetexko}\n\\setmainfont{Times New Roman}\n"
         "\\setmainhangulfont{AppleMyungjo}[AutoFakeBold=2.5]\n\\setsanshangulfont{Apple SD Gothic Neo}\n",
         "\\usepackage{fontspec}\n\\usepackage{xetexko}\n" + LATIN_FONT + HANGUL_FONTS),
        ("\\graphicspath{{../../../paper/bigcomp2027/figures/}}", "\\graphicspath{{figures/}}"),
        ("\\definecolor{revblue}{HTML}{1F4FD8}\n",
         "% Revised text is blue. For an all-black copy change 1F4FD8 to 000000.\n\\definecolor{revblue}{HTML}{1F4FD8}\n"),
        ("\\bibliography{../BIGCOMP2027_REFS_2026-10-07}", "\\bibliography{refs}"),
    ] + [(f"\\input{{../{src}.tex}}", f"\\input{{sections_ko/{dst}}}") for src, dst in KO_SECTIONS], "Korean master")

    files: dict[str, bytes] = {
        "main.tex": (MAGIC + english).encode("utf-8"),
        "main_ko.tex": (MAGIC + korean).encode("utf-8"),
        "refs.bib": BIB.read_bytes(),
        "IEEEtran.cls": (CLS_DIR / "IEEEtran.cls").read_bytes(),
        "IEEEtran.bst": (CLS_DIR / "IEEEtran.bst").read_bytes(),
        "latexmkrc": LATEXMKRC.encode("utf-8"),
    }
    for src, dst in KO_SECTIONS:
        body = (SEC / f"{src}.tex").read_text(encoding="utf-8")
        if "\\input{" in body or "\\include{" in body:
            raise SystemExit(f"{src}: nested \\input / \\include is not handled")
        files[f"sections_ko/{dst}.tex"] = body.encode("utf-8")
    for f in FIGURES:
        files[f"figures/{f}"] = (FIGDIR / f).read_bytes()

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, files[name])
    return sorted(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="2026-10-08")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    out = a.out_dir / f"bigcomp2027_overleaf_{a.stamp}.zip"
    names = build(out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {len(names)} files)")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
