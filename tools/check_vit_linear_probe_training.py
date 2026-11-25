"""Technical test of train_vit_linear_probe.py on small stratified subsets. No full training, no test image opened.

Runs --check-only, two identical short runs with feature cache (reproducibility), one short run without cache, and
checks cache against direct computation, head-only updates, saved files and every image file that was opened.
"""
import argparse,csv,json,shutil
from pathlib import Path
from common import ROOT
import PIL.Image,torch,torch_directml
from torch.utils.data import DataLoader,Subset
import train_vit_linear_probe as T
from pet_dataset import PetDataset
from vit_linear_probe import build_vit_linear_probe,load_linear_probe,set_probe_mode,vit_features

opened=[]; _open=PIL.Image.open
# Geöffnete Bilder mitschreiben, um Testbilder zu erkennen
def recording_open(fp,*a,**k):
    opened.append(str(fp)); return _open(fp,*a,**k)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    out=Path(a.out); shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    PIL.Image.open=recording_open
    T.main(['--check-only']); print('PASS --check-only',flush=True)
    # Zwei gleiche kurze Läufe müssen dasselbe ergeben
    common=['--limit-train','320','--limit-val','100','--epochs','2']
    run_a=T.main(common+['--output-dir',str(out/'lauf_a')]); run_b=T.main(common+['--output-dir',str(out/'lauf_b')])
    assert run_a['train_images']==320 and run_a['val_images']==100 and len(run_a['history'])==2
    keys=['train_loss','train_accuracy','val_loss','val_accuracy','val_macro_f1']
    same=[[r[k] for k in keys] for r in run_a['history']]==[[r[k] for k in keys] for r in run_b['history']]
    assert same; print('PASS zwei Laeufe mit gleichem Seed identisch, Merkmale jeweils neu berechnet',[round(r['val_macro_f1'],4) for r in run_a['history']],flush=True)
    run_r=T.main(common+['--output-dir',str(out/'lauf_a')])
    assert [[r[k] for k in keys] for r in run_r['history']]==[[r[k] for k in keys] for r in run_a['history']]
    print('PASS erneuter Lauf liest den Cache und ergibt dieselben Werte',flush=True)
    for name in ('best_head.pt','last_checkpoint.pt','training_log.csv','history.json','run_config.json'): assert (out/'lauf_a'/name).exists(),name
    last=torch.load(out/'lauf_a/last_checkpoint.pt',weights_only=True); state=last['optimizer']['state']
    assert len(last['optimizer']['param_groups'][0]['params'])==2 and sorted(v['exp_avg'].shape for v in state.values())==sorted([torch.Size([10]),torch.Size([10,768])])
    config=json.loads((out/'lauf_a/run_config.json').read_text(encoding='utf-8')); assert config['trainable_parameters']==7690 and config['selection']=='best validation macro-F1'
    print('PASS Dateien gespeichert; Optimizer verwaltet nur Kopfgewichte 10x768 und Bias 10',flush=True)

    # Cache mit der direkten Berechnung vergleichen
    torch.manual_seed(0); model,pre=build_vit_linear_probe(); device=torch_directml.device(); model=model.to(device); set_probe_mode(model)
    cache=sorted((out/'lauf_a/feature_cache').glob('train_*.pt')); stored=torch.load(cache[0],weights_only=True)
    data=PetDataset(T.DEFAULT_DATASET+'/train.csv',transform=pre); idx=T.stratified_subset(data,320,42)[:32]
    images,labels=next(iter(DataLoader(Subset(data,idx),batch_size=32)))
    with torch.no_grad(): direct=vit_features(model,images.to(device)).cpu(); full=model(images.to(device)).cpu(); split=model.heads.head(vit_features(model,images.to(device))).cpu()
    diff_cache=float((stored['features'][:32]-direct).abs().max()); diff_split=float((full-split).abs().max())
    assert torch.equal(stored['labels'][:32],labels) and diff_cache<1e-4 and diff_split<1e-4
    print('PASS Cache gegen direkte Berechnung, maximale Abweichung',diff_cache,'| Merkmale+Kopf gegen vollen Forward',diff_split,flush=True)

    # Nur der Kopf darf sich geändert haben
    torch.manual_seed(42); fresh,_=build_vit_linear_probe(); best=torch.load(out/'lauf_a/best_head.pt',weights_only=True)
    moved=float((best['head']['weight']-fresh.heads.head.weight).abs().max()); assert moved>0
    loaded,_,_=load_linear_probe(out/'lauf_a/best_head.pt')
    assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in loaded.state_dict().items() if not k.startswith('heads.'))
    print('PASS Kopf hat sich veraendert (max',round(moved,4),'), Backbone im Lauf und beim Laden unveraendert',flush=True)

    run_c=T.main(['--features','online','--limit-train','64','--limit-val','30','--epochs','1','--output-dir',str(out/'lauf_online')])
    assert run_c['history'][0]['train_loss']>0; print('PASS Modus ohne Cache laeuft',flush=True)

    # Es darf kein Testbild geöffnet worden sein
    with open(ROOT/'data/final_splits/test.csv',encoding='utf-8-sig',newline='') as h: test={str(Path(r['path'])).lower() for r in csv.DictReader(h)}
    seen={str(Path(p)).lower() for p in opened}; assert not seen&test
    print('PASS kein Oxford-Testbild geoeffnet; geoeffnete Bilddateien',len(seen),flush=True)
    (out/'test_ergebnis.json').write_text(json.dumps(dict(reproduzierbar=same,cache_abweichung=diff_cache,split_abweichung=diff_split,kopf_aenderung=moved,
        geoeffnete_bilder=len(seen),testbilder_geoeffnet=0,lauf_a=[{k:r[k] for k in keys} for r in run_a['history']],lauf_online={k:run_c['history'][0][k] for k in keys}),indent=2,ensure_ascii=False),encoding='utf-8')
    print('ALLES BESTANDEN',flush=True)

if __name__=='__main__': main()
