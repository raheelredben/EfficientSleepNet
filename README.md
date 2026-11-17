# EfficientSleepNet

Compact PyTorch implementation for **five-class sleep-stage classification from single-channel EEG**.

The default model uses multi-resolution 1D convolutions, cross-aware recalibration attention (CARA), a **two-layer bidirectional LSTM encoder and two-layer attention-based LSTM decoder**, and a class-aware cross-entropy objective. It accepts a **30-second EEG segment sampled at 100 Hz** (`[B, 1, 3000]`) and returns five stage logits (`[B, 5]`): Wake, N1, N2, N3, and REM.

## Model architecture

- **MRCNN:** parallel short- and long-kernel branches; convolution channel widths `32 → 64 → 128` in each branch and kernels of `50` and `400` samples.
- **Compact convolutions:** short-branch Conv3 uses `groups=2`, long-branch Conv2 uses `groups=4`, long-branch Conv3 uses `groups=8`, and a `1×1` convolution mixes channels after long-branch Conv3.
- **CARA:** five-head temporal attention, 255-dimensional internal projection, depthwise refinements, dual-pooling gating, and residual connection.
- **Encoder:** two-layer bidirectional LSTM, 128 hidden units per direction.
- **Decoder:** two stacked unidirectional LSTM cells, 128 hidden units each; additive attention across encoder time steps, a start token, and autoregressive stage-token feedback. Decoder hidden/cell states are initialized from both encoder directions using shared trainable projection layers.
- **Model size:** **2,507,445 trainable parameters** (~9.57 MiB of float32 weights). This is an architectural count, not a trained checkpoint.

The compact convolution grouping and shared decoder projections keep the encoder–decoder network close to 2.5 million parameters. The decoder and encoder are both implemented in `efficientsleepnet_model.py`. Set `COMPACT_CONVOLUTIONS=False` in `config.py` to use a fully dense convolution alternative (6,177,461 trainable parameters). Changing this setting creates separate checkpoint/result directories.

## Setup

Python 3.10+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The included loader supports the Sleep-EDF Sleep-Cassette PSG/hypnogram EDF naming convention. Download the data separately, then place `*-PSG.edf` and matching `*-Hypnogram.edf` files in `data/`.

Default cohort: Sleep-Cassette participant IDs **SC400–SC419** (20 participants). To use another cohort, update `SLEEP_EDF_SUBJECT_IDS` in `config.py`. The loader is configured to require all selected participants; set `REQUIRE_COMPLETE_COHORT=False` only if you intentionally want to use a partial cohort. EEG recordings and labels are not bundled.

Preprocessing selects the Fpz-Cz EEG derivation, resamples to 100 Hz, excludes unknown/movement stages, maps N3/N4 together, creates nonoverlapping 30-second epochs and applies epoch-wise z-score normalization. Different datasets need dataset-specific loaders.

## Run

```bash
# Test model, dataset handling and training utilities (synthetic data only)
python -m unittest discover -s tests -v

# Train with subject-independent five-fold cross-validation
python train.py

# Evaluate held-out test folds after training
python evaluate_all.py
```

For the default `L=1` case, the decoder produces exactly one prediction using a start token. No ground-truth label is used as an input at that step, so `train.py` and `evaluate_all.py` do not require changes. For experimental multi-step decoding, call `model(x, decode_steps=3)` for `[B,3,5]` logits. During training only, `model(x, decode_steps=3, targets=target_labels, teacher_forcing_ratio=1.0)` enables shifted teacher forcing after the first output. **The included EEG loader/training scripts only train single-epoch (`L=1`) classification**; multi-step decoding is not a validated sleep-stage sequence training pipeline. The decoder has a fixed number of output steps and does not produce an EOS token.

Configuration is centralized in `config.py`. Default training uses RMSprop, learning rate `1e-4`, momentum `0.9`, weight decay `5e-4`, batch size `32`, up to `100` epochs, and five subject-level folds with a `3:1:1` train/validation/test rotation. The best checkpoint is chosen by validation macro-F1 (validation loss breaks ties).

Training saves checkpoints under `trained_models/<architecture>/` and fold histories, split manifests, and evaluation outputs under `results/<architecture>/`. Both paths are excluded from Git tracking.

## Notes

- **L=1** refers to one 30-second EEG segment, not the number of LSTM layers.
- Checkpoints from the previous decoder-free release are incompatible with the new encoder–decoder architecture. Train new checkpoints before evaluation.
- An untrained network's parameter count does **not** establish clinical performance. No real dataset training results or pretrained weights are included.
- Test files verify shapes, parameter counts, loss, subject splits, and synthetic-data execution.
- Only Sleep-EDF loading is implemented here; the other dataset channel names in `config.py` are informational and do not enable additional loaders.

## Contact

- **Raheel Aslam** — `raheel.mark88@gmail.com`

## License

No license has been selected for this repository. Add a `LICENSE` file if you intend to grant reuse permissions.
