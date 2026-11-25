"""Technical test of train_resnet18_adaptation.py on small stratified subsets: short runs, interruption and resume in
separate processes, frozen mode, speed and GPU memory (Windows counters of the child process), no Oxford test image opened. No full training.
"""
import argparse,csv,json,re,subprocess,sys
from pathlib import Path
from common import ROOT
import torch

# Höchsten gemessenen Speicherverbrauch auslesen
def peaks_of(s):
    s.kill(); rows=[l.split() for l in s.stdout.read().splitlines() if re.fullmatch(r'[0-9.]+ [0-9.]+',l.strip())]
    return dict(dediziert_spitze_gb=round(max(float(r[0]) for r in rows)/1e9,2),gemeinsam_spitze_gb=round(max(float(r[1]) for r in rows)/1e9,2),messpunkte=len(rows)) if rows else None

# Einzelner Testlauf in einem eigenen Prozess
def child(argv,record,memory):
    import os,time,warnings; warnings.simplefilter('always')
    s=None
    if memory:  # der GPU-Zähler existiert erst, wenn DirectML gestartet ist
        import torch_directml; torch.zeros(1).to(torch_directml.device()).item(); s=sampler(os.getpid()); time.sleep(3)
    # Geöffnete Bilder mitschreiben, um Testbilder zu erkennen
    import PIL.Image
    opened=[]; original=PIL.Image.open
    def recording(fp,*a,**k): opened.append(str(fp)); return original(fp,*a,**k)
    PIL.Image.open=recording
    import train_resnet18_adaptation as T
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); result=T.main(argv)
    peaks=peaks_of(s) if s else None
    fallbacks=sorted({str(w.message)[:120] for w in caught if 'DML' in str(w.message) or 'fall back' in str(w.message)})
    Path(record).write_text(json.dumps(dict(opened=opened,fallbacks=fallbacks,speicher=peaks,result=result is not None),ensure_ascii=False),encoding='utf-8')

# GPU-Speicher über Windows messen
def sampler(pid):
    script=("while($true){try{$c=(Get-Counter @('\\GPU Process Memory(pid_%d_*)\\Dedicated Usage','\\GPU Process Memory(pid_%d_*)\\Shared Usage') -ErrorAction Stop).CounterSamples;"
            "$d=($c|?{$_.Path -like '*dedicated*'}|Measure-Object CookedValue -Sum).Sum;$s=($c|?{$_.Path -like '*shared*'}|Measure-Object CookedValue -Sum).Sum;"
            "[Console]::WriteLine(\"$d $s\")}catch{[Console]::WriteLine('x')}}")%(pid,pid)
    return subprocess.Popen(['powershell.exe','-NoProfile','-Command',script],stdout=subprocess.PIPE,text=True)

# Testlauf starten und Ausgabe lesen
def run(argv,record,log,memory=False):
    with open(log,'w',encoding='utf-8') as h:
        code=subprocess.run([sys.executable,'-X','utf8',__file__,'--child',record,'--memory' if memory else '--no-memory','--',*argv],stdout=h,stderr=subprocess.STDOUT,cwd=ROOT).returncode
    text=Path(log).read_text(encoding='utf-8'); assert code==0,text[-1500:]
    return text,json.loads(Path(record).read_text(encoding='utf-8'))['speicher']

def main():
    if '--child' in sys.argv:
        i=sys.argv.index('--child'); return child(sys.argv[sys.argv.index('--')+1:],sys.argv[i+1],sys.argv[i+2]=='--memory')
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    common=['--limit-train','96','--limit-val','64','--epochs','2','--log-every','1']
    records=[]; R={}
    def go(name,argv,memory=False):
        record=str(out/f'{name}_dateien.json'); text,peaks=run(argv,record,out/f'{name}.log',memory); records.append(record); return text,peaks
    for mode in ('finetune','frozen'):
        text,_=go(f'check_only_{mode}',['--check-only','--mode',mode]); assert 'CHECK OK' in text
    print('PASS --check-only (finetune und frozen): zehn Klassen in fester Reihenfolge, Gradienten genau bei den trainierbaren Parametern',flush=True)

    # Ein Lauf ohne und einer mit Unterbrechung müssen gleich enden
    go('lauf_U',common+['--output-dir',str(out/'lauf_U')])
    text,_=go('lauf_I',common+['--output-dir',str(out/'lauf_IR'),'--stop-after-epoch','1']); assert 'endet nach Epoche 1' in text
    ck=torch.load(out/'lauf_IR/last_checkpoint.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==1 and set(ck['random_states'])=={'python','numpy','torch','directml','loader'} and ck['optimizer']['state']
    print('PASS Unterbrechung nach Epoche 1; Checkpoint enthaelt Modell samt BatchNorm-Statistik, Optimizer, Epoche, Verlauf, Zufallszustaende',flush=True)
    text,_=go('lauf_R',common+['--output-dir',str(out/'lauf_IR'),'--resume','auto']); assert 'Fortgesetzt aus' in text
    u=json.loads((out/'lauf_U/history.json').read_text(encoding='utf-8')); r=json.loads((out/'lauf_IR/history.json').read_text(encoding='utf-8'))
    keys=['train_loss','train_accuracy','val_loss','val_accuracy','val_macro_f1']
    same_history=[[h[k] for k in keys] for h in u['history']]==[[h[k] for k in keys] for h in r['history']]
    cu=torch.load(out/'lauf_U/last_checkpoint.pt',map_location='cpu',weights_only=False); cr=torch.load(out/'lauf_IR/last_checkpoint.pt',map_location='cpu',weights_only=False)
    same_model=all(torch.equal(cu['model'][k],cr['model'][k]) for k in cu['model'])
    same_optimizer=all(torch.equal(cu['optimizer']['state'][i][n],cr['optimizer']['state'][i][n]) for i in cu['optimizer']['state'] for n in ('exp_avg','exp_avg_sq'))
    R['wiederaufnahme']=dict(verlauf_identisch=same_history,modell_identisch=same_model,optimizer_identisch=same_optimizer,verlauf=[{k:h[k] for k in keys} for h in u['history']])
    if not same_history:
        R['wiederaufnahme']['verlauf_R']=[{k:h[k] for k in keys} for h in r['history']]
        R['wiederaufnahme']['groesste_gewichtsabweichung']=max((cu['model'][k].float()-cr['model'][k].float()).abs().max().item() for k in cu['model'])
    (out/'test_ergebnis.json').write_text(json.dumps(R,indent=2,ensure_ascii=False),encoding='utf-8')
    assert same_history and same_model and same_optimizer, R['wiederaufnahme']
    print('PASS Wiederaufnahme in neuem Prozess bitgleich zum ununterbrochenen Lauf: Verlauf, Modell, Optimizer',[[round(h[k],4) for k in keys] for h in u['history']],flush=True)

    # Beim Fine-Tuning müssen sich alle Gewichte ändern
    from train_resnet18_adaptation import build_resnet18,load_resnet18_checkpoint
    fresh,_,cfg=build_resnet18('finetune'); init=fresh.state_dict()
    params={n for n,_ in fresh.named_parameters()}
    changed=sum(not torch.equal(cu['model'][n],init[n]) for n in params)
    bn_changed=sum(not torch.equal(cu['model'][n],init[n]) for n in init if n.endswith('running_mean'))
    assert cfg['trainable_parameters']==11181642 and changed==len(params) and bn_changed==20
    best,_,bcfg,bck=load_resnet18_checkpoint(out/'lauf_U/best_resnet18.pt'); assert bcfg['mode']=='finetune' and bck['epoch']==u['best_epoch']
    for name in ('best_resnet18.pt','last_checkpoint.pt','training_log.csv','history.json','run_config.json'): assert (out/'lauf_U'/name).exists(),name
    print(f'PASS finetune: alle {len(params)} Parametertensoren (11.181.642 Parameter) veraendert, BatchNorm-Statistik in allen 20 Schichten aktualisiert; best_resnet18.pt ladbar',flush=True)

    # Im Modus frozen darf sich nur der Kopf ändern
    go('frozen',common+['--mode','frozen','--output-dir',str(out/'frozen')])
    cf=torch.load(out/'frozen/last_checkpoint.pt',map_location='cpu',weights_only=False); fz,_,fcfg=build_resnet18('frozen'); fi=fz.state_dict()
    diff={n for n in fi if not torch.equal(cf['model'][n],fi[n])}
    assert diff=={'fc.weight','fc.bias'} and fcfg['trainable_parameters']==5130
    print('PASS frozen: nur fc.weight und fc.bias (5.130 Parameter) veraendert; Backbone und BatchNorm-Statistik unveraendert',flush=True)

    conf=json.loads((out/'lauf_U/run_config.json').read_text(encoding='utf-8'))
    R['konfiguration']={k:conf[k] for k in ('learning_rate','head_learning_rate','batch_size','seed','weight_decay','optimizer','selection','batchnorm','preprocessing')}
    R['parameter']=dict(finetune=cfg['trainable_parameters'],frozen=fcfg['trainable_parameters'],gesamt=cfg['parameters'])

    # Tempo und Speicher messen
    for workers in (0,2):
        _,peaks=go(f'tempo_w{workers}',['--limit-train','512','--limit-val','128','--epochs','1','--log-every','0','--num-workers',str(workers),'--output-dir',str(out/f'tempo_w{workers}')],memory=True)
        h=json.loads((out/f'tempo_w{workers}/history.json').read_text(encoding='utf-8'))
        R[f'tempo_workers_{workers}']=dict(sekunden_512_plus_128_val=round(h['trained_seconds'],1),geschaetzte_epoche_minuten=round(h['trained_seconds']/640*(12768+1233)/60,1),speicher=peaks)
        print('INFO Tempo und Speicher, Batch 32,',workers,'Ladeprozesse',R[f'tempo_workers_{workers}'],flush=True)

    # Es darf kein Testbild geöffnet worden sein
    with open(ROOT/'data/final_splits/test.csv',encoding='utf-8-sig',newline='') as h: test={str(Path(r['path'])).lower() for r in csv.DictReader(h)}
    seen=set(); fallbacks=set()
    for rec in records:
        d=json.loads(Path(rec).read_text(encoding='utf-8')); fallbacks|=set(d['fallbacks'])
        for p in d['opened']: seen.add(str(Path(p)).lower())
    assert not seen&test; R['geoeffnete_bilder_hauptprozess']=len(seen); R['testbilder_geoeffnet']=0; R['directml_cpu_rueckfaelle']=sorted(fallbacks)
    print('PASS kein Oxford-Testbild geoeffnet; geoeffnete Bilddateien',len(seen),'| DirectML-Rueckfaelle:',sorted(fallbacks) or 'keine',flush=True)
    (out/'test_ergebnis.json').write_text(json.dumps(R,indent=2,ensure_ascii=False),encoding='utf-8')
    print('ALLES BESTANDEN',flush=True)

if __name__=='__main__': main()
