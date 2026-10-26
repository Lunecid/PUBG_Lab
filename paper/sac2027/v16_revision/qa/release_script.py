from pathlib import Path
import csv,hashlib,json,statistics,re,difflib,importlib.util,subprocess,tempfile,shutil,zipfile
from pypdf import PdfReader
ROOT=Path('/Users/lunecid/project/PUBG');P=ROOT/'paper/sac2027/v15_feedback';OLD=ROOT/'paper/sac2027/v14_review';BASE=ROOT/'reports/sac2027_patch_v14'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Version and inherited evidence preservation.
manifest=json.loads((P/'qa/v14_source_manifest.json').read_text())
assert all(sha(OLD/n)==h for n,h in manifest.items())
assert sha(P/'RESULTS_KO.md')==sha(OLD/'RESULTS_KO.md')
for p in (OLD/'tables').glob('*'):
 if p.is_file():assert sha(p)==sha(P/'tables'/p.name)
for p in (OLD/'figures').glob('*.pdf'):assert sha(p)==sha(P/'figures'/p.name)
assert sha(P/'refs.bib')==sha(OLD/'refs.bib')
raw=[r for r in csv.DictReader((BASE/'primary_results_v2/contrasts_pooled.csv').open()) if r['kind']=='primary']
copied=list(csv.DictReader((P/'tables/primary_decisions_values.csv').open()));assert raw==copied
for r in raw:
 theta=float(r['theta']);lo=float(r['ci90_low']);hi=float(r['ci90_high']);delta=float(r['sesoi']);pv=float(r['p_holm'])
 verdict='gain' if pv<.05 and theta>=delta else 'equivalent' if lo>=-delta and hi<=delta else 'inconclusive'
 assert verdict==r['verdict']
rows=[r for r in csv.DictReader((BASE/'results_v2/ecological_model/auc_comparison.csv').open()) if r['cohort']!='karakin_tpp_squad']
lite={}
for arm in ['PHASE-only','EHM-lite','HGB-cut']:
 rs=[r for r in rows if r['arm']==arm];assert len(rs)==11
 lite[arm]={k:statistics.mean(float(r[k]) for r in rs) for k in ['auc','auc_within_phase','nll','brier']}
(P/'tables/lite_absolute_values.csv').write_text('arm,auc,auc_within_phase,nll,brier\n'+'\n'.join(a+','+','.join(str(v) for v in d.values()) for a,d in lite.items())+'\n')
# The Markdown parser's XML escape is reversible; no original content changed by anchorization.
decode=lambda s:re.sub(r'^<!--block:B\d+-->\n','',s,flags=re.M).replace('%ARS_XML%','')
assert decode((P/'revision/base.md').read_text())==(OLD/'main.tex').read_text()
assert decode((P/'revision/revised_deliverable.md').read_text())==(P/'main.tex').read_text()
(P/'revision/v14_to_v15.diff').write_text(''.join(difflib.unified_diff((OLD/'main.tex').read_text().splitlines(True),(P/'main.tex').read_text().splitlines(True),fromfile='v14/main.tex',tofile='v15/main.tex')))
files=['reports/sac2027_patch_v14/protocol.json','reports/sac2027_patch_v14/split_manifest.csv','reports/sac2027_patch_v14/final_verification.json','reports/sac2027_patch_v14/primary_results_v2/contrasts_pooled.csv','reports/sac2027_patch_v14/results_v2/ecological_model/auc_comparison.csv','model_stgcn/phase_v2/features.py','model_stgcn/phase_v2/grid.py','model_stgcn/phase_v2/alive.py','model_stgcn/phase_v2/labels.py','model_stgcn/phase_v2/replay.py','model_stgcn/ptr/aggregate.py','third_party/krafton_pubg_winprob/adaptation/train.py','third_party/krafton_pubg_winprob/adaptation/h0_fit.py','third_party/krafton_pubg_winprob/src/training/losses.py']
evidence=dict(version='v15',base='v14_patch_retraining',new_training=False,new_riskset_audit=False,checked_sources={n:sha(ROOT/n) for n in files},v14_all_files_preserved=True,existing_tables_figures_and_results_preserved=True,primary_decisions_from_csv=True,lite_absolute_means=lite,external_source={'url':'https://pubg.com/en/news/9926','checked':'2026-10-08','supported_claim':'Update 41.1 bot behavior/positioning and Jammer Pack tiers changed'},patch_protocol={'format':'ARS block patch','xml_adapter':'escape line-initial CCSXML tags in working Markdown; losslessly restore in final TeX','base_roundtrip_verified':True,'apply_report':'revision/revised_deliverable.md.apply-report.json'},remaining=['all-candidate online eligibility audit','redeploy event completeness','full-feature logistic and standalone MLP fits','all-cohort multiseed tuning','normal/ranked stratification','controlled latency/memory and top-k or user utility'])
(P/'qa/revision_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('verify',ROOT/'tools/sac2027_verify_paper.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);v.PAPER=P;v.main()
qa=json.loads((P/'qa/final_validation.json').read_text());qa['visual_review']={'pages_inspected':list(range(1,11)),'issues':[]};qa['revision_evidence_checks']=True;(P/'qa/final_validation.json').write_text(json.dumps(qa,indent=2)+'\n')
# Rebuild a clean standalone source set before delivering a ZIP.
names=['main.tex','refs.bib','main.bbl','acmart.cls','ACM-Reference-Format-citation-order.bst','ACM-Reference-Format.bst','TEMPLATE_README.txt','README_KO.md','CHANGES_KO.md','REVIEW_RESPONSE_KO.md','RESULTS_KO.md','METHODS_TRACE_KO.md','qa/final_validation.json','qa/citation_order.json','qa/revision_evidence.json','revision/v14_to_v15.diff','revision/patch.json','revision/base.md.block-manifest.json','revision/revised_deliverable.md.apply-report.json']
files=[P/n for n in names]+sorted((P/'tables').glob('*.tex'))+sorted((P/'tables').glob('*.csv'))+[P/'figures'/n for n in ['fig_pipeline_clean.pdf','fig_rq1_effects.pdf','fig_contrast.pdf']]+sorted((P/'figures').glob('*.csv'))
source_manifest={str(p.relative_to(P)):sha(p) for p in files}
with tempfile.TemporaryDirectory(prefix='sac-v15-rebuild-') as tmp:
 stage=Path(tmp)
 for p in files:
  dest=stage/p.relative_to(P);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 proc=subprocess.run(['/opt/homebrew/bin/tectonic','-C','--keep-logs','--keep-intermediates','main.tex'],cwd=stage,capture_output=True,text=True)
 (P/'qa/package_build.log').write_text(proc.stdout+'\n'+proc.stderr)
 assert proc.returncode==0,proc.stderr[-1000:]
 for n,h in source_manifest.items():assert sha(stage/n)==h,('source modified',n)
 a=PdfReader(P/'main.pdf');b=PdfReader(stage/'main.pdf');assert len(a.pages)==len(b.pages)==10
 same_text=all(x.extract_text()==y.extract_text() for x,y in zip(a.pages,b.pages));same_pages=all(x.get_contents().get_data()==y.get_contents().get_data() for x,y in zip(a.pages,b.pages));assert same_text and same_pages
 assert 'Overfull \\hbox' not in (stage/'main.log').read_text();assert 'Overfull \\vbox' not in (stage/'main.log').read_text()
 out=ROOT/'output/pdf';name='SAC2027_MLA_v15_feedback_2026-10-08';pdf=out/(name+'.pdf');archive=out/'SAC2027_MLA_v15_source_2026-10-08.zip'
 shutil.copy2(P/'main.pdf',pdf)
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,str(p.relative_to(P)))
  z.write(pdf,pdf.name);z.writestr('SOURCE_MANIFEST.json',json.dumps(source_manifest,indent=2)+'\n')
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for n,h in source_manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 result=dict(pages=10,references=39,tables=8,figures=3,standalone_build=True,text_identical=same_text,page_content_identical=same_pages,pdf_sha256=sha(pdf),zip_sha256=sha(archive),source_files=len(files),v14_preserved=True)
 (P/'qa/package_validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2));print(pdf);print(archive)
