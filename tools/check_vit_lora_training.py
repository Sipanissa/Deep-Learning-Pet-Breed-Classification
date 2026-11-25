"""Technical test of train_vit_lora.py on small stratified subsets: short runs, interruption and resume in separate
processes, only LoRA and head change, no Oxford test image opened. No full training.
"""
import argparse,csv,json,subprocess,sys
from pathlib import Path
from common import ROOT
import torch

# Einzelner Testlauf in einem eigenen Prozess
def child(argv,record):
    import PIL.Image
    opened=[]; original=PIL.Image.open
    def recording(fp,*a,**k): opened.append(str(fp)); return original(fp,*a,**k)
    PIL.Image.open=recording
    import train_vit_lora as T
    result=T.main(argv)
    Path(record).write_text(json.dumps(dict(opened=opened,result=result is not None),ensure_ascii=False),encoding='utf-8')

# Testlauf starten und Ausgabe lesen
def run(argv,record,log):
    with open(log,'w',encoding='utf-8') as h:
        p=subprocess.run([sys.executable,'-X','utf8',__file__,'--child',record,'--',*argv],stdout=h,stderr=subprocess.STDOUT,cwd=ROOT)
    text=Path(log).read_text(encoding='utf-8'); assert p.returncode==0,text[-1500:]
    return text

def main():
    if '--child' in sys.argv:
        i=sys.argv.index('--child'); return child(sys.argv[sys.argv.index('--')+1:],sys.argv[i+1])
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    common=['--rank','8','--limit-train','96','--limit-val','64','--epochs','2','--log-every','1']
    records=[]; R={}
    def go(name,argv):
        record=str(out/f'{name}_dateien.json'); text=run(argv,record,out/f'{name}.log'); records.append(record); return text
    text=go('check_only',['--check-only']); assert 'CHECK OK' in text; print('PASS --check-only',flush=True)
    # Ein Lauf ohne und einer mit Unterbrechung müssen gleich enden
    go('lauf_U',common+['--output-dir',str(out/'lauf_U')])
    text=go('lauf_I',common+['--output-dir',str(out/'lauf_IR'),'--stop-after-epoch','1']); assert 'endet nach Epoche 1' in text
    ck=torch.load(out/'lauf_IR/last_checkpoint.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==1 and set(ck['random_states'])=={'python','numpy','torch','directml','loader'} and ck['optimizer']['state']
    print('PASS Unterbrechung nach Epoche 1; Checkpoint enthaelt Gewichte, Optimizer, Epoche, Verlauf, Zufallszustaende',sorted(ck['random_states']),flush=True)
    text=go('lauf_R',common+['--output-dir',str(out/'lauf_IR'),'--resume','auto']); assert 'Fortgesetzt aus' in text
    u=json.loads((out/'lauf_U/history.json').read_text(encoding='utf-8')); r=json.loads((out/'lauf_IR/history.json').read_text(encoding='utf-8'))
    keys=['train_loss','train_accuracy','val_loss','val_accuracy','val_macro_f1']
    same_history=[[h[k] for k in keys] for h in u['history']]==[[h[k] for k in keys] for h in r['history']]
    cu=torch.load(out/'lauf_U/last_checkpoint.pt',map_location='cpu',weights_only=False); cr=torch.load(out/'lauf_IR/last_checkpoint.pt',map_location='cpu',weights_only=False)
    same_weights=all(torch.equal(cu['trainable'][k],cr['trainable'][k]) for k in cu['trainable'])
    same_optimizer=all(torch.equal(cu['optimizer']['state'][i][n],cr['optimizer']['state'][i][n]) for i in cu['optimizer']['state'] for n in ('exp_avg','exp_avg_sq'))
    assert same_history and same_weights and same_optimizer
    print('PASS Wiederaufnahme in neuem Prozess bitgleich zum ununterbrochenen Lauf: Verlauf, Gewichte, Optimizer',[[round(h[k],4) for k in keys] for h in u['history']],flush=True)
    R['wiederaufnahme']=dict(verlauf_identisch=same_history,gewichte_identisch=same_weights,optimizer_identisch=same_optimizer,verlauf=[{k:h[k] for k in keys} for h in u['history']])

    # Nur LoRA und Kopf dürfen sich geändert haben
    from vit_lora import build_vit_lora,load_lora_checkpoint
    torch.manual_seed(42); fresh,_,cfg=build_vit_lora(8,16.0,0.1)
    init={n:p.detach() for n,p in fresh.named_parameters() if p.requires_grad}
    assert set(cu['trainable'])==set(init) and len(init)==50 and cfg['trainable_parameters']==302602
    changed=sum(not torch.equal(cu['trainable'][n],init[n]) for n in init); assert changed==50
    best,_,bcfg,bck=load_lora_checkpoint(out/'lauf_U/best_lora.pt')
    assert all(torch.equal(p,fresh.state_dict()[n]) for n,p in best.state_dict().items() if n not in init)
    print('PASS nur 50 LoRA- und Kopftensoren (302.602 Parameter) trainierbar und veraendert; eingefrorene Gewichte im besten Modell gleich dem Ausgangsmodell',flush=True)
    for name in ('best_lora.pt','last_checkpoint.pt','training_log.csv','history.json','run_config.json'): assert (out/'lauf_U'/name).exists(),name
    conf=json.loads((out/'lauf_U/run_config.json').read_text(encoding='utf-8')); R['konfiguration']={k:conf[k] for k in ('learning_rate','head_learning_rate','batch_size','seed','weight_decay','optimizer','selection')}|{'alpha':conf['lora']['alpha'],'dropout':conf['lora']['dropout'],'rank':conf['lora']['rank'],'trainable':conf['lora']['trainable_parameters']}
    print('PASS gespeichert: best_lora.pt, last_checkpoint.pt, training_log.csv, history.json, run_config.json',R['konfiguration'],flush=True)

    # Ladetempo mit 0 und 2 Ladeprozessen messen
    for workers in (0,2):
        go(f'tempo_w{workers}',['--rank','8','--limit-train','256','--limit-val','32','--epochs','1','--log-every','0','--num-workers',str(workers),'--output-dir',str(out/f'tempo_w{workers}')])
        h=json.loads((out/f'tempo_w{workers}/history.json').read_text(encoding='utf-8'))
        R[f'tempo_workers_{workers}']=dict(sekunden_256_bilder_plus_32_val=round(h['trained_seconds'],1),geschaetzte_epoche_minuten=round(h['trained_seconds']/288*(12768+1233)/60,1))
        print('INFO Tempo',workers,'Ladeprozesse',R[f'tempo_workers_{workers}'],flush=True)

    # Es darf kein Testbild geöffnet worden sein
    with open(ROOT/'data/final_splits/test.csv',encoding='utf-8-sig',newline='') as h: test={str(Path(r['path'])).lower() for r in csv.DictReader(h)}
    seen=set()
    for rec in records:
        for p in json.loads(Path(rec).read_text(encoding='utf-8'))['opened']: seen.add(str(Path(p)).lower())
    assert not seen&test; R['geoeffnete_bilder_hauptprozess']=len(seen); R['testbilder_geoeffnet']=0
    print('PASS kein Oxford-Testbild geoeffnet; geoeffnete Bilddateien',len(seen),'(Laeufe mit Ladeprozessen zaehlen nur den Hauptprozess)',flush=True)
    (out/'test_ergebnis.json').write_text(json.dumps(R,indent=2,ensure_ascii=False),encoding='utf-8')
    print('ALLES BESTANDEN',flush=True)

if __name__=='__main__': main()
