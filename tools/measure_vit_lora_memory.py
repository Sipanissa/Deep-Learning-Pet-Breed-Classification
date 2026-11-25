"""Measures GPU memory and speed of LoRA training steps for one batch size in a fresh process. No training run.

Uses stratified training images only, takes a few optimizer steps on a throwaway model and saves nothing but --out.
The Windows counters report this process's dedicated GPU memory and shared (system) memory used by the GPU.
"""
import argparse,json,math,os,re,subprocess,time
from pathlib import Path
from common import ROOT
import torch,torch.nn as nn,torch_directml
from torch.utils.data import DataLoader,Subset
from pet_dataset import PetDataset
from train_vit_linear_probe import DEFAULT_DATASET,stratified_subset
from vit_lora import AdamWNoLerp,build_vit_lora

# GPU-Speicher über Windows messen
def sampler():
    pid=os.getpid()
    script=("while($true){try{$c=(Get-Counter @('\\GPU Process Memory(pid_%d_*)\\Dedicated Usage','\\GPU Process Memory(pid_%d_*)\\Shared Usage') -ErrorAction Stop).CounterSamples;"
            "$d=($c|?{$_.Path -like '*dedicated*'}|Measure-Object CookedValue -Sum).Sum;$s=($c|?{$_.Path -like '*shared*'}|Measure-Object CookedValue -Sum).Sum;"
            "[Console]::WriteLine(\"$d $s\")}catch{[Console]::WriteLine('x')}}")%(pid,pid)
    return subprocess.Popen(['powershell.exe','-NoProfile','-Command',script],stdout=subprocess.PIPE,text=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--batch',type=int,required=True); ap.add_argument('--rank',type=int,default=8); ap.add_argument('--steps',type=int,default=6); ap.add_argument('--out',required=True); a=ap.parse_args()
    # Modell und einen Batch vorbereiten
    device=torch_directml.device(); proc=sampler(); time.sleep(3)
    torch.manual_seed(42); model,pre,config=build_vit_lora(a.rank); model=model.to(device); model.train()
    optimizer=AdamWNoLerp([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=0.01); criterion=nn.CrossEntropyLoss()
    data=PetDataset(DEFAULT_DATASET+'/train.csv',transform=pre); x,y=next(iter(DataLoader(Subset(data,stratified_subset(data,a.batch,42)),batch_size=a.batch)))
    result=dict(batch=a.batch,rank=a.rank,trainable_parameters=config['trainable_parameters'])
    try:
        times=[];losses=[]
        # Einige Trainingsschritte messen
        for _ in range(a.steps):
            s=time.time(); optimizer.zero_grad(); loss=criterion(model(x.to(device)),y.to(device)); loss.backward(); optimizer.step(); losses.append(float(loss.item())); times.append(time.time()-s)
        assert all(math.isfinite(l) for l in losses)
        model.eval(); s=time.time()
        with torch.no_grad(): float(model(x.to(device)).sum().item())
        val_time=time.time()-s
        # Die ersten zwei Schritte dienen nur zum Aufwärmen
        step=sum(times[2:])/len(times[2:])
        result.update(status='stabil',losses=[round(l,4) for l in losses],sekunden_pro_schritt=round(step,2),bilder_pro_sekunde=round(a.batch/step,1),
                      geschaetzte_trainingsepoche_minuten=round(12768/a.batch*step/60,1),validierung_sekunden_pro_batch=round(val_time,2))
    except RuntimeError as e:
        result.update(status='Fehler',meldung=(e.args[0] if e.args else str(e))[:300])
    except UnicodeDecodeError as e:
        result.update(status='Fehler',meldung=e.object.decode('cp1252')[:300])
    # Messung beenden und Ergebnis speichern
    time.sleep(3); proc.kill(); rows=[l.split() for l in proc.stdout.read().splitlines() if re.fullmatch(r'[0-9.]+ [0-9.]+',l.strip())]
    result.update(dediziert_spitze_gb=round(max(float(r[0]) for r in rows)/1e9,2) if rows else None,gemeinsam_spitze_gb=round(max(float(r[1]) for r in rows)/1e9,2) if rows else None,messpunkte=len(rows))
    Path(a.out).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
