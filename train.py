"""Sleep-EDF subject-independent 5-fold training (3 train, 1 val, 1 test).

Held-out test participants are not used during training or model selection.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

import config as cfg
from dataset_loader import SleepEEGDataset,discover_sleep_edf_records
from efficientsleepnet_model import efficientsleepnet
from utils import ensure_dir, seed_everything, count_parameters


class ClassAwareLoss(nn.Module):
    """Weighted cross-entropy: mean(weight[target] * -log p[target]).

    This differs from torch.nn.CrossEntropyLoss(weight=...), which divides
    by the total weight OF THE OBSERVED BATCH instead of by sample count N.
    """
    def __init__(self,class_counts,alpha=cfg.CLASS_AWARE_ALPHA,
                 beta=cfg.CLASS_AWARE_BETA,gamma=cfg.CLASS_AWARE_GAMMA,
                 rho=cfg.CLASS_AWARE_RHO):
        super().__init__()
        counts=torch.as_tensor(class_counts,dtype=torch.float32)
        if counts.ndim!=1 or counts.numel()!=cfg.NUM_CLASSES:
            raise ValueError('class_counts must contain five class frequencies')
        if torch.any(counts <= 0):
            raise ValueError('Training fold lacks a sleep stage. Class-aware logarithmic weights undefined')
        zeta=torch.tensor([beta,gamma,beta,alpha,beta],dtype=torch.float32) / cfg.NUM_CLASSES
        freq_factor=torch.maximum(torch.ones_like(counts),torch.log(rho*counts.sum()/counts))
        self.register_buffer('weights',zeta*freq_factor)

    def forward(self,logits,targets):
        unreduced=nn.functional.cross_entropy(logits,targets,reduction='none')
        return (unreduced*self.weights[targets]).mean()


def make_subject_folds(subjects,n_folds=cfg.N_FOLDS,seed=cfg.RANDOM_STATE):
    """Unique PARTICIPANTS, not EEG nights; 3:1:1 subject-fold rotation."""
    unique=sorted(set(subjects))
    if len(unique)<n_folds:
        raise ValueError(f'{len(unique)} unique participants cannot make {n_folds} folds')
    if n_folds!=5:
        raise ValueError('This implementation uses five folds (3:1:1)')
    kf=KFold(n_splits=n_folds,shuffle=True,random_state=seed)
    idx_groups=[np.asarray(test) for _,test in kf.split(unique)]
    results=[]
    for i in range(n_folds):
        test={unique[k] for k in idx_groups[i]}
        val={unique[k] for k in idx_groups[(i+1)%n_folds]}
        train=set(unique)-val-test
        assert not (train&val or val&test or train&test)
        assert train|val|test==set(unique)
        results.append({'train':sorted(train),'val':sorted(val),'test':sorted(test)})
    return results


def _make_loader(records,partition,shuffle=False):
    ds=[]
    for record in records:
        if record.participant_id in partition:
            ds.append(SleepEEGDataset(record.edf_path,record.hyp_path,verbose=True))
    if not ds:
        raise RuntimeError(f'No usable EEG for participant set {sorted(partition)}')
    union=ConcatDataset(ds)
    loader=DataLoader(union,batch_size=cfg.BATCH_SIZE,shuffle=shuffle,
                      num_workers=0,pin_memory=(cfg.DEVICE=='cuda'))
    return loader,ds


def _epoch(model,loader,loss_fn,optimizer=None):
    is_train=optimizer is not None
    model.train(is_train)
    tot_n=0; tot_loss=0.0; yt=[];yp=[]
    ctx=torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x,y in loader:
            x=x.to(cfg.DEVICE,non_blocking=True)
            y=y.to(cfg.DEVICE,non_blocking=True)
            if is_train: optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss=loss_fn(logits,y)
            if is_train:
                loss.backward()
                if cfg.GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.GRAD_CLIP_NORM)
                optimizer.step()
            tot_n+=len(y);tot_loss+=loss.detach().item()*len(y)
            yt.extend(y.cpu().tolist());yp.extend(logits.argmax(-1).detach().cpu().tolist())
    return {'loss':tot_loss/tot_n, 'acc':accuracy_score(yt,yp),
            'macro_f1':f1_score(yt,yp,labels=list(range(cfg.NUM_CLASSES)),
                               average='macro',zero_division=0)}


def _optimizer(model):
    if cfg.OPTIMIZER.lower()!='rmsprop':
        raise ValueError('Only RMSprop is configured; update _optimizer for another optimizer')
    return torch.optim.RMSprop(model.parameters(),lr=cfg.LEARNING_RATE,
                               weight_decay=cfg.WEIGHT_DECAY,momentum=cfg.MOMENTUM)


def _paths():
    ensure_dir(cfg.MODEL_DIR);ensure_dir(cfg.RESULTS_DIR)
    return Path(cfg.MODEL_DIR),Path(cfg.RESULTS_DIR)


def train():
    seed_everything(cfg.RANDOM_STATE)
    model_dir,results_dir=_paths()
    records=discover_sleep_edf_records(cfg.DATA_DIR)
    folds=make_subject_folds([r.participant_id for r in records])
    manifest={'dataset':cfg.DATASET,'seed':cfg.RANDOM_STATE,
              'records':[vars(r) for r in records], 'folds':folds,
              'architecture':cfg.ARCHITECTURE_SPEC}
    (results_dir/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(f'{len(records)} recordings, {len(set(r.participant_id for r in records))} participants')
    for i,split in enumerate(folds,1):
        print(f'\nFold {i}/5: {len(split["train"])} train, '
              f'{len(split["val"])} val, {len(split["test"])} held-out participants')
        train_loader,train_sets=_make_loader(records,set(split['train']),shuffle=True)
        val_loader,_=_make_loader(records,set(split['val']))
        class_counts=torch.bincount(torch.cat([ds.y for ds in train_sets]),
                                    minlength=cfg.NUM_CLASSES)
        loss_fn=ClassAwareLoss(class_counts).to(cfg.DEVICE)
        print('Class counts:',class_counts.tolist())
        print('Class-aware loss weights:',loss_fn.weights.detach().cpu().tolist())
        model=efficientsleepnet().to(cfg.DEVICE)
        print('Model parameters:',count_parameters(model))
        optimizer=_optimizer(model)
        scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',
                    factor=0.5,patience=8) if cfg.USE_LR_SCHEDULER else None)
        best=(-float('inf'),float('inf'));best_epoch=-1;bad=0;history=[]
        savepath=model_dir/f'efficientsleepnet_fold{i}_best.pt'
        for epoch in range(1,cfg.EPOCHS+1):
            tr=_epoch(model,train_loader,loss_fn,optimizer)
            va=_epoch(model,val_loader,loss_fn)
            if scheduler: scheduler.step(va['loss'])
            history.append({'epoch':epoch,'train':tr,'val':va})
            print(f'Fold {i} Epoch {epoch:03d}: train loss {tr["loss"]:.4f}, '
                  f'val loss {va["loss"]:.4f}, val MF1 {va["macro_f1"]:.4f}')
            # Choose checkpoint by validation macro-F1, using loss as a tiebreaker.
            key=(va['macro_f1'],-va['loss'])
            if key > (best[0],-best[1]):
                best=(va['macro_f1'],va['loss']);best_epoch=epoch;bad=0
                torch.save({'model_state':model.state_dict(),'fold':i,'epoch':epoch,
                            'val_macro_f1':va['macro_f1'],'val_loss':va['loss'],
                            'train_class_counts':class_counts.tolist(),
                            'train_subjects':split['train'],'val_subjects':split['val'],
                            'test_subjects':split['test'],
                            'architecture':cfg.ARCHITECTURE_SPEC},savepath)
            else:
                bad+=1
                if cfg.EARLY_STOP_PATIENCE and bad>=cfg.EARLY_STOP_PATIENCE:
                    print(f'Early stopped at epoch {epoch}')
                    break
        (results_dir/f'fold{i}_history.json').write_text(
            json.dumps(history,indent=2),encoding='utf8')
        print(f'[BEST] Fold {i}: epoch {best_epoch} val MF1 {best[0]:.4f}; '
              'test not touched during model selection')
    print('Training complete. Run python evaluate_all.py for held-out test metrics.')


if __name__=='__main__':train()
