"""Verify manuscript preservation and PDF integrity; this does not rerun experiments."""
from pathlib import Path
import hashlib, json, re
from collections import Counter
from pypdf import PdfReader

release = Path(__file__).resolve().parents[1]
manifest = json.loads((release / 'qa/source_manifest.json').read_text())
old = Path(manifest['inputs'][0]['path']).read_text()
new = (release / 'main.tex').read_text()
checks = {}
def check(name, condition, details=None):
    checks[name] = {'passed': bool(condition), 'details': details}
    assert condition, (name, details)
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
check('original_inputs_unchanged', all(digest(i['path']) == i['sha256'] for i in manifest['inputs']))
check('bibliography_source_identical', (release/'refs.bib').read_bytes() == Path(manifest['inputs'][2]['path']).read_bytes())
for i in manifest['inputs'][3:]:
    check('figure_identical_' + Path(i['path']).stem, digest(release/'figures'/Path(i['path']).name) == i['sha256'])
eq = lambda s: re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', s, re.S)
check('three_equations_exact', eq(old) == eq(new) and len(eq(new)) == 3)
def cited(s):
    return set(','.join(re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}', s)).split(','))
check('all_32_citation_keys_preserved', cited(old) == cited(new) and len(cited(new)) == 32)
def tables(s):
    out = {}
    for block in re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', s, re.S):
        out[re.search(r'\\label\{([^}]+)\}', block).group(1)] = block
    return out
old_tables, new_tables = tables(old), tables(new)
check('five_tables_preserved', old_tables.keys() == new_tables.keys() and len(new_tables) == 5)
for label, block in old_tables.items():
    # All data rows, including signs, precision, bolding and categorical cells.
    # Only metric header rows were redesigned in the performance table.
    rows = [line.strip() for line in block.splitlines() if ' & ' in line and line.rstrip().endswith('\\\\')]
    for row in rows:
        if row.startswith((' & ', 'Design &')) or row.startswith('&'):
            continue
        check('row_' + label + '_' + row.split('&')[0].strip(), row in new_tables[label])
section = lambda s: s[s.index('\\section{Results}'):s.index('\\section{Discussion}')]
decimals = lambda s: set(re.findall(r'(?<!\d)\d+\.\d+(?!\d)', s))
check('result_decimal_values_preserved', decimals(section(old)) == decimals(section(new)))
def result_math(s):
    s = re.sub(r'\\begin\{(?:table|figure)\*?\}.*?\\end\{(?:table|figure)\*?\}', '', section(s), flags=re.S)
    return Counter(re.findall(r'\$([^$]+)\$', s))
check('all_result_inline_math_occurrences_preserved', not (result_math(old) - result_math(new)))

labels = re.findall(r'\\label\{([^}]+)\}', new)
refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', new)
check('unique_labels', len(labels) == len(set(labels)))
check('all_crossrefs_resolve', set(refs) <= set(labels))
check('cutoff_score_at_a', "$s_{m,i,k}$ is team $i$'s score at the cutoff $a_{m,k}$" in new)
check('three_relation_definitions_preserved', 'proximity (dropped for pairs that fought), combat, and shared zone pressure (two teams within 200\\,m, both outside or within 150\\,m of the zone edge)' in new)
for name, phrase in {
    'ai_team_robustness': 'lost significance in ranked matches',
    'reverse_order_check': '$+0.034$ $[0.029, 0.040]$',
    'patch_holdout': '$0.0021$ $[0.0008, 0.0035]$',
    'within_phase_patch_holdout': '$0.0039$ $[0.0003, 0.0075]$',
    'original_task_check': 'Both adaptations retain predictive value on their original tasks.',
}.items():
    check(name, phrase in new)
log = (release/'main.log').read_text()
check('no_undefined_citations_or_refs', not re.search(r'(?:Citation|Reference).*undefined|undefined references|multiply defined', log))
check('no_horizontal_overflow', 'Overfull \\hbox' not in log)
reader = PdfReader(release/'main.pdf')
check('pdf_page_count', len(reader.pages) == 9)
text = '\n'.join(p.extract_text() or '' for p in reader.pages)
check('no_replacement_glyphs', '\ufffd' not in text)
check('pdf_reference_count', len(re.findall(r'\\bibitem', (release/'main.bbl').read_text())) == 32)
names = reader.named_destinations
link_count = 0
missing = []
for page in reader.pages:
    for a in page.get('/Annots', []):
        a = a.get_object()
        action = a.get('/A')
        if action:
            action = action.get_object()
        destination = a.get('/Dest')
        if action and action.get('/S') == '/GoTo':
            destination = action.get('/D')
        if isinstance(destination, str):
            link_count += 1
            if destination not in names:
                missing.append(destination)
check('internal_pdf_links_resolve', not missing, {'checked': link_count, 'missing': missing})
report = {
    'scope': 'Manuscript/source/PDF preservation and rendering validation only; no experiment reruns or independent validation of original results.',
    'checks': checks,
    'summary': {'passed': len(checks), 'failed': 0, 'pages': len(reader.pages), 'tables': 5, 'figures': 3, 'equations': 3, 'cited_references': 32},
    'warnings': re.findall(r'Overfull \\vbox[^\n]*', log),
}
(release/'qa/validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report['summary']))
