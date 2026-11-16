"""Strict, single-channel Sleep-EDF loader.

- Sleep-Cassette PSG EDF and matching hypnogram EDF.
- Subject identity grouping: 'SC4001E0' and 'SC4002E0' are recordings
  of participant 'SC400' and MUST stay in the same CV fold.
- Expand annotations into exact nonoverlapping 30-second windows and never
  synthesize labels across unmapped/movement annotations.
- Resample to 100 Hz; epoch-wise z-scoring.

ISRUC and SHHS have different annotations/files, and are intentionally NOT
claimed supported by this Sleep-EDF-specific loader.
"""
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List

import mne
import numpy as np
import torch
from torch.utils.data import Dataset
from config import (DATA_DIR, EEG_CHANNEL_BY_DATASET, DATASET, EPOCH_DURATION,
                    INPUT_LENGTH, SAMPLE_RATE, TRIM_WAKE_MINUTES,
                    SLEEP_EDF_SUBJECT_IDS, REQUIRE_COMPLETE_COHORT)
from utils import zscore

STAGE_MAP = {
    'sleep stage w': 0, 'w': 0, 'wake': 0,
    'sleep stage 1': 1, 'sleep stage n1': 1, 'n1': 1, '1': 1,
    'sleep stage 2': 2, 'sleep stage n2': 2, 'n2': 2, '2': 2,
    'sleep stage 3': 3, 'sleep stage n3': 3, 'n3': 3, '3': 3,
    'sleep stage 4': 3, 'sleep stage n4': 3, 'n4': 3, '4': 3,
    'sleep stage r': 4, 'sleep stage rem': 4, 'rem': 4, 'r': 4,
}


@dataclass(frozen=True)
class SleepRecord:
    record_id: str
    participant_id: str
    edf_path: str
    hyp_path: str


def canonical_stage(name):
    return STAGE_MAP.get(str(name).strip().lower())


def participant_id_from_record(record_id: str) -> str:
    """Sleep-EDF SC NN[N] first 5 chars are a PERSON, next digit is night.

    Dataset naming documented at Sleep-EDF PhysioNet; e.g. SC4001E0 and
    SC4002E0 are nights for the same individual SC400. Reject unknown IDs
    instead of accidentally performing recording-wise CV.
    """
    match = re.fullmatch(r'(SC\d{3})[12][A-Za-z]\d', record_id, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized Sleep-Cassette record ID: {record_id!r}; "
                         "subject identity cannot be verified")
    return match.group(1).upper()


def _match_hypnogram(record_id: str, hyp_paths: List[Path]) -> Path:
    """PSG SC4001E0 and hypnogram SC4001EC match on first six chars."""
    matches = [p for p in hyp_paths if p.name[:6].upper() == record_id[:6].upper()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly 1 hypnogram for {record_id}, found "
                         f"{len(matches)}: {[p.name for p in matches]}")
    return matches[0]


def discover_sleep_edf_records(data_dir=DATA_DIR):
    if DATASET != 'sleep_edf':
        raise NotImplementedError('This loader supports Sleep-EDF Sleep-Cassette only. '
                                  'ISRUC/SHHS formats require dedicated adapters.')
    root=Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Sleep-EDF directory does not exist: {root}")
    psg=sorted(p for p in root.iterdir() if p.name.lower().endswith('-psg.edf'))
    hyps=sorted(p for p in root.iterdir() if 'hypnogram' in p.name.lower() and p.suffix.lower()=='.edf')
    if not psg or not hyps:
        raise FileNotFoundError(f"Expected *-PSG.edf and *Hypnogram.edf in {root}")
    records=[]
    for p in psg:
        rid=p.name[:-len('-PSG.edf')]
        records.append(SleepRecord(rid,participant_id_from_record(rid),str(p),
                                   str(_match_hypnogram(rid,hyps))))
    cohort=set(SLEEP_EDF_SUBJECT_IDS)
    records=[r for r in records if r.participant_id in cohort]
    found={r.participant_id for r in records}
    missing=cohort-found
    if REQUIRE_COMPLETE_COHORT and missing:
        raise ValueError(f'Missing {len(missing)} Sleep-EDF20 participants: {sorted(missing)}')
    if len({r.record_id for r in records})!=len(records):
        raise ValueError('Duplicate Sleep-EDF recording IDs found.')
    return records


def discover_sleep_edf_files(data_dir=DATA_DIR):
    """Compatibility function; use discover_sleep_edf_records for subject CV."""
    records=discover_sleep_edf_records(data_dir)
    return ([Path(r.edf_path).name for r in records],
            [Path(r.hyp_path).name for r in records],
            [r.participant_id for r in records])


def find_hyp_file(record_id,hyp_files):
    paths=[Path(h) for h in hyp_files]
    return _match_hypnogram(record_id,paths).name


def _segments_for_annotations(annotations, n_samples, sfreq):
    """Return stage/onset pairs for fully labeled epochs only.

    Never use events_from_annotations directly: that creates one event at
    the start of a multi-epoch annotation and loses the remaining epochs.
    """
    steps=[]
    for ann in annotations:
        stage=canonical_stage(ann['description'])
        if stage is None:
            continue
        onset=float(ann['onset'])
        duration=float(ann['duration'])
        # EDF hypnogram stages normally align to 30-second boundaries.
        # Only complete annotated 30s epochs can become supervised examples.
        count=int(np.floor((duration + 1e-5) / EPOCH_DURATION))
        for i in range(count):
            a=onset + i*EPOCH_DURATION
            b=a + EPOCH_DURATION
            if a < -1e-5 or b>n_samples/sfreq+1e-4:
                continue
            start=int(round(a*sfreq))
            end=start+INPUT_LENGTH
            if start<0 or end>n_samples:
                continue
            steps.append((start,end,stage))
    steps.sort(key=lambda a:a[0])
    for i in range(1,len(steps)):
        if steps[i][0] < steps[i-1][1]:
            raise ValueError('Overlapping hypnogram segments; review original annotations')
    if not steps:
        raise ValueError('No complete annotated 30-second epochs')
    if TRIM_WAKE_MINUTES is not None:
        nonwake=[i for i,(_,_,label) in enumerate(steps) if label!=0]
        if nonwake:
            pad=int(round(TRIM_WAKE_MINUTES*60/EPOCH_DURATION))
            steps=steps[max(0,nonwake[0]-pad):min(len(steps),nonwake[-1]+pad+1)]
    return steps


class SleepEEGDataset(Dataset):
    def __init__(self,edf_path,hyp_path,verbose=False):
        # Annotations external to PSG are the source of truth.
        annotations=mne.read_annotations(str(hyp_path))
        raw=mne.io.read_raw_edf(str(edf_path),preload=False,verbose='ERROR')
        selected=EEG_CHANNEL_BY_DATASET['sleep_edf']
        choices=[c for c in raw.ch_names if c.lower().replace(' ','').removeprefix('eeg')
                 == selected.lower().replace(' ','')]
        if len(choices)!=1:
            raise ValueError(f"Expected EEG channel {selected!r} in {edf_path}; "
                             f"available channels={raw.ch_names}. No silent fallback allowed.")
        raw.pick([choices[0]])
        raw.load_data(verbose='ERROR')
        if abs(float(raw.info['sfreq'])-SAMPLE_RATE)>1e-6:
            raw.resample(SAMPLE_RATE, npad='auto',verbose='ERROR')
        sig=raw.get_data(picks=[choices[0]])[0].astype(np.float32,copy=False)
        slices=_segments_for_annotations(annotations,len(sig),SAMPLE_RATE)
        # Z-score individual 30s epochs rather than the full night.
        x=np.stack([zscore(sig[start:end]) for start,end,_ in slices])[:,None,:]
        y=np.asarray([label for _,_,label in slices],dtype=np.int64)
        self.X=torch.from_numpy(x)
        self.y=torch.from_numpy(y)
        self.channel=choices[0]
        self.record_id=Path(edf_path).name.split('-')[0]
        if verbose:
            vals,cts=np.unique(y,return_counts=True)
            print(f'[LOAD] {self.record_id}: {tuple(self.X.shape)}, EEG={self.channel}, '
                  f'labels={dict(zip(vals.tolist(),cts.tolist()))}')

    def __len__(self): return len(self.y)
    def __getitem__(self,idx): return self.X[idx],self.y[idx]
