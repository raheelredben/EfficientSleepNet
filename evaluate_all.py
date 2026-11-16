"""Aggregate held-out subject-fold predictions and export evaluation metrics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

import config as cfg
from dataset_loader import SleepEEGDataset,discover_sleep_edf_records
from efficientsleepnet_model import efficientsleepnet
from utils import ensure_dir


def per_class_metrics_from_cm(cm,class_names=cfg.CLASS_NAMES):
    total=cm.sum()
    rows=[]
    for i,name in enumerate(class_names):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp;fn=cm[i,:].sum()-tp;tn=total-tp-fp-fn
        p=tp/(tp+fp) if tp+fp else 0.0
        r=tp/(tp+fn) if tp+fn else 0.0
        spec=tn/(tn+fp) if tn+fp else 0.0
        f=2*p*r/(p+r) if p+r else 0.0
        rows.append({'class':name,'precision':p,'recall':r,'specificity':spec,
                     'f1':f,'gmean':np.sqrt(r*spec),'support':int(cm[i,:].sum())})
    return pd.DataFrame(rows)


def metrics(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred,labels=list(range(cfg.NUM_CLASSES)))
    pc=per_class_metrics_from_cm(cm)
    return {'accuracy':float(np.mean(np.asarray(y_true)==np.asarray(y_pred))),
            'macro_f1':float(f1_score(y_true,y_pred,average='macro',
                                     labels=list(range(cfg.NUM_CLASSES)),zero_division=0)),
            'kappa':float(cohen_kappa_score(y_true,y_pred)),
            'macro_gmean':float(pc['gmean'].mean())},cm,pc


def evaluate_all_folds():
    ensure_dir(cfg.RESULTS_DIR)
    root=Path(cfg.RESULTS_DIR)
    manifest_path=root/'split_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Training-generated split_manifest.json absent. Train first.')
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('architecture') != cfg.ARCHITECTURE_SPEC:
        raise ValueError('Architecture mode differs from training split manifest')
    records=discover_sleep_edf_records(cfg.DATA_DIR)
    recorded_map={r.record_id:r.participant_id for r in records}
    saved_map={r['record_id']:r['participant_id'] for r in manifest['records']}
    if recorded_map!=saved_map:
        raise ValueError('Dataset records/subject mapping changed since training')
    yall=[];pall=[];rows=[];csvrows=[];tested_subjects=set()
    for i,split in enumerate(manifest['folds'],1):
        path=Path(cfg.MODEL_DIR)/f'efficientsleepnet_fold{i}_best.pt'
        if not path.exists():
            raise FileNotFoundError(f'Missing fold checkpoint {path}; refusing partial 5-fold result')
        # torch >= 2.6 defaults to weights_only=True; our dict is primitive+tensor.
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        if ckpt.get('architecture') != cfg.ARCHITECTURE_SPEC:
            raise ValueError(f'Fold{i} architecture mismatch')
        for part in ['train','val','test']:
            if ckpt[f'{part}_subjects']!=split[part]:
                raise ValueError(f'Fold{i} checkpoint/manifest mismatch: {part}')
        model=efficientsleepnet().to(cfg.DEVICE)
        model.load_state_dict(ckpt['model_state'],strict=True)
        model.eval();yt=[];yp=[]
        for record in records:
            if record.participant_id not in split['test']:
                continue
            ds=SleepEEGDataset(record.edf_path,record.hyp_path)
            loader=DataLoader(ds,batch_size=cfg.BATCH_SIZE,shuffle=False)
            with torch.inference_mode():
                preds=[]
                for x,_ in loader:
                    preds.extend(model(x.to(cfg.DEVICE)).argmax(dim=-1).cpu().tolist())
            actual=ds.y.tolist()
            yt.extend(actual);yp.extend(preds)
            csvrows.extend({'fold':i,'participant_id':record.participant_id,
                            'record_id':record.record_id,'epoch':j,'true':int(a),
                            'pred':int(b)} for j,(a,b) in enumerate(zip(actual,preds)))
        if not yt:raise ValueError(f'Fold {i}: zero held-out test epochs')
        if tested_subjects.intersection(split['test']):
            raise ValueError('Participant was included in test of multiple folds')
        tested_subjects.update(split['test'])
        vals,_,_=metrics(yt,yp)
        rows.append({'fold':i,'n_test_epochs':len(yt),**vals})
        print(f'Fold {i}: {vals}')
        yall+=yt;pall+=yp
    if tested_subjects != set(recorded_map.values()):
        raise ValueError('Not all participants evaluated exactly once')
    allvals,cm,pc=metrics(yall,pall)
    print(f'Pooled held-out 5-fold results: {allvals}')
    pd.DataFrame(rows).to_csv(root/'fold_summary_metrics.csv',index=False)
    pd.DataFrame([allvals]).to_csv(root/'overall_metrics.csv',index=False)
    pc.to_csv(root/'per_class_metrics_combined.csv',index=False)
    pd.DataFrame(csvrows).to_csv(root/'heldout_predictions.csv',index=False)
    pd.DataFrame(cm,index=cfg.CLASS_NAMES,columns=cfg.CLASS_NAMES).to_csv(root/'confusion_matrix_counts.csv')
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(cm)
    ax.set_xticks(range(cfg.NUM_CLASSES),cfg.CLASS_NAMES)
    ax.set_yticks(range(cfg.NUM_CLASSES),cfg.CLASS_NAMES)
    ax.set(xlabel='Predicted',ylabel='True',title='Combined held-out 5-fold CV')
    for a in range(cfg.NUM_CLASSES):
        for b in range(cfg.NUM_CLASSES):
            ax.text(b,a,str(cm[a,b]),ha='center',va='center')
    fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(root/'conf_matrix_5fold.png',dpi=200);plt.close(fig)
    return allvals

if __name__=='__main__':evaluate_all_folds()
