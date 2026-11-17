"""Run: python -m unittest discover -s tests -v"""
import unittest
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import mne
import numpy as np
import torch
from torch.nn import functional as F
from dataset_loader import (participant_id_from_record, _segments_for_annotations,
                            canonical_stage, SleepEEGDataset)
from efficientsleepnet_model import efficientsleepnet
from train import ClassAwareLoss,make_subject_folds
from utils import count_parameters

class EfficientSleepNetTests(unittest.TestCase):
    def test_model_shapes_and_parameter_counts(self):
        model=efficientsleepnet().eval()
        with torch.inference_mode():
            x=torch.randn(1,1,3000)
            a=model.mrcnn.short_branch(x);b=model.mrcnn.long_branch(x)
            self.assertEqual(tuple(a.shape),(1,128,187))
            self.assertEqual(tuple(b.shape),(1,128,187))
            self.assertEqual(tuple(model.mrcnn(x).shape),(1,256,187))
            self.assertEqual(tuple(model(x).shape),(1,5))
            self.assertEqual(model.cara.attn.inner,255)
            self.assertEqual(model.cara.attn.heads,5)
        # Compact grouped-convolution model output dimensions.
        self.assertEqual(count_parameters(model),2507445)
        # Dense-convolution alternative remains available.
        self.assertEqual(count_parameters(efficientsleepnet(compact=False)),6177461)
        self.assertEqual(model.birnn.num_layers, 2)
        self.assertEqual(model.birnn.hidden_size, 128)
        self.assertEqual(len(model.decoder.cells), 2)
        self.assertEqual(model.decoder.hidden_size, 128)
        self.assertEqual(model.mrcnn.short_branch.conv3.conv.conv.groups, 2)
        self.assertEqual(model.mrcnn.long_branch.conv3.conv[0].conv.groups, 8)
        self.assertEqual(model.mrcnn.long_branch.conv3.conv[1].kernel_size, (1,))

    def test_compact_training_step(self):
        model = efficientsleepnet().train()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
        x = torch.randn(2, 1, 3000)
        y = torch.tensor([0, 1])
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(model.mrcnn.long_branch.conv3.conv[1].weight.grad).all().item())
        self.assertIsNotNone(model.decoder.cells[1].weight_ih.grad)
        self.assertTrue(torch.isfinite(model.decoder.cells[1].weight_ih.grad).all().item())
        optimizer.step()

    def test_attentive_decoder_teacher_forcing_and_attention(self):
        model = efficientsleepnet().train()
        x = torch.randn(2,1,3000)
        targets = torch.tensor([[0,1,2],[3,4,0]],dtype=torch.long)
        logits,weights = model(x,decode_steps=3,targets=targets,
                               teacher_forcing_ratio=1.0,return_attention=True)
        self.assertEqual(tuple(logits.shape),(2,3,5))
        self.assertEqual(tuple(weights.shape),(2,3,187))
        torch.testing.assert_close(weights.sum(-1),torch.ones(2,3),atol=1e-5,rtol=1e-5)
        # Default one-step decoder remains compatible with training/evaluation.
        single,alpha = model(x,return_attention=True)
        self.assertEqual(tuple(single.shape),(2,5))
        self.assertEqual(tuple(alpha.shape),(2,187))
        with self.assertRaises(ValueError):
            model(x,decode_steps=3,targets=torch.zeros(2,2,dtype=torch.long))
        with self.assertRaises(ValueError):
            model(x,decode_steps=0)

    def test_long_sequence_output_shape(self):
        model=efficientsleepnet().eval()
        with torch.inference_mode():
            x=torch.randn(1,1,30000)
            self.assertEqual(tuple(model.mrcnn(x).shape),(1,256,1875))
            self.assertEqual(tuple(model(x).shape),(1,5))

    def test_class_aware_loss_formula(self):
        count=torch.tensor([8285,2804,17799,5703,7717])
        loss=ClassAwareLoss(count)
        zeta=torch.tensor([1.5,2.,1.5,1.,1.5])/5.
        expected=zeta*torch.maximum(torch.ones(5),torch.log(1.2*count.sum()/count))
        torch.testing.assert_close(loss.weights,expected)
        logits=torch.tensor([[2.,0.,1.,-.5,1.],[0.,2.,0.,1.,0.]])
        labels=torch.tensor([0,1])
        expected_scalar=(F.cross_entropy(logits,labels,reduction='none')*expected[labels]).mean()
        torch.testing.assert_close(loss(logits,labels),expected_scalar)

    def test_strict_subject_independence(self):
        self.assertEqual(participant_id_from_record('SC4001E0'),'SC400')
        self.assertEqual(participant_id_from_record('SC4002E0'),'SC400')
        persons=[f'SC4{i:02d}' for i in range(20)]
        folds=make_subject_folds(persons)
        self.assertEqual(len(folds),5)
        seen=[]
        for f in folds:
            self.assertEqual(len(f['train']),12)
            self.assertEqual(len(f['val']),4)
            self.assertEqual(len(f['test']),4)
            self.assertFalse(set(f['train'])&set(f['test']))
            self.assertFalse(set(f['train'])&set(f['val']))
            self.assertFalse(set(f['val'])&set(f['test']))
            seen.extend(f['test'])
        self.assertCountEqual(seen,persons)
        with self.assertRaises(ValueError): participant_id_from_record('fake_record')

    def test_loader_one_channel_and_full_annotations(self):
        # Mock MNE EDF I/O to verify the actual loader path without user EDF files.
        sfreq=100
        times=np.arange(18000)/sfreq
        eeg=(5*np.sin(2*np.pi*2*times)).astype(np.float64)
        other=(7*np.cos(2*np.pi*6*times)).astype(np.float64)
        raw=mne.io.RawArray(np.vstack([eeg,other]),
            mne.create_info(['EEG Fpz-Cz','Pz-Oz'],sfreq,['eeg','eeg']),verbose='ERROR')
        annotations=mne.Annotations(onset=[0.,90.,120.],duration=[90.,30.,60.],
                        description=['Sleep stage W','Movement time','Sleep stage R'])
        with patch('dataset_loader.mne.io.read_raw_edf',return_value=raw), \
             patch('dataset_loader.mne.read_annotations',return_value=annotations):
            ds=SleepEEGDataset('SC4001E0-PSG.edf','SC4001EC-Hypnogram.edf')
        self.assertEqual(tuple(ds.X.shape),(5,1,3000))
        self.assertEqual(ds.y.tolist(),[0,0,0,4,4])
        self.assertTrue(torch.allclose(ds.X.mean(dim=-1),torch.zeros(5,1),atol=1e-5))
        self.assertTrue(torch.allclose(ds.X.std(dim=-1,unbiased=False),
                          torch.ones(5,1),atol=1e-4))

    def test_expand_annotations_with_exclusions(self):
        ann=mne.Annotations(onset=[0,90,120],duration=[90,30,60],
                            description=['Sleep stage W','Movement time','Sleep stage 4'])
        segments=_segments_for_annotations(ann,18000,100)
        self.assertEqual(len(segments),5)
        self.assertEqual([z[2] for z in segments],[0,0,0,3,3])
        self.assertEqual([z[0] for z in segments],[0,3000,6000,12000,15000])
        self.assertEqual(canonical_stage('Sleep stage ?'),None)
        self.assertEqual(canonical_stage('Sleep stage R'),4)

if __name__=='__main__':unittest.main()
