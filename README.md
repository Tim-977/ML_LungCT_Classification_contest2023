# Lung CT Classification: SimpleConvNet vs. AlexNet, With and Without Segmentation Masks


Three-class classification of lung CT scans (**Healthy / Infected / COVID-19**) using two CNN architectures, plus an experiment in mask-guided training using a self-trained UNet segmentation network. Implemented in PyTorch.

## Data

- 27,000 labeled training images (`train_answers.csv`, 3 classes), grayscale, PNG.
- Two preprocessing pipelines depending on target architecture:
  - **256×256**, grayscale, no normalization, used for SimpleConvNet.
  - **224×224**, grayscale, `Normalize(mean=0.5, std=0.5)`, used for AlexNet.
- Splits: 80/20 for SimpleConvNet (21,600 / 5,400), 90/10 for AlexNet (24,300 / 2,700). Different splits were used per-model run rather than a fixed held-out set, worth fixing if this is reused (see Limitations).

## Architectures

| Model | Parameters | Input | Role |
|---|---|---|---|
| `SimpleConvNet` | 419,931 | 1×256×256 | Baseline: 4 conv blocks + 3 FC layers |
| `AlexNet` (modified) | 14,256,323 | 1×224×224 | Classic AlexNet feature extractor, FC head expanded to 5 layers (512-unit) before the final 3-way output |
| `UNet` | 31,036,481 | 1×256×256 → 1×256×256 | Trained separately to segment lung tissue; used only to *generate masks*, not for classification |
| `HybridNet` (AE + AlexNet) |, |, | Attempted, removed from the final notebook, training diverged (exploding gradients) before it produced usable results |

The UNet is trained once (10 epochs, MSE loss against ground-truth lung masks) and then used to auto-generate masks for the full train/test set, which are cached to disk (`train_lung_masks_generated/`, `test_lung_masks_generated/`) and reused by the mask-augmented classifiers.

## Training setup

- Optimizer: Adam, default LR, no scheduler.
- Loss: `CrossEntropyLoss` (classification), `MSELoss` (UNet segmentation).
- Batch size 128 (classifiers), 16 (UNet).
- No data augmentation beyond resize/grayscale/normalize.
- No gradient clipping anywhere in the pipeline, relevant to the failure below.

## Results

Final-epoch metrics on the held-out test split (macro-averaged F1/Recall/Precision):

| Model | Epochs | Test F1 | Test Recall | Test Precision | Test Loss |
|---|---|---|---|---|---|
| SimpleConvNet | 20 | 0.854 | 0.854 | 0.854 | 0.00335 |
| **AlexNet** | 20 | **0.897** | **0.897** | **0.899** | **0.00289** |
| SimpleConvNet + masks | 20 | 0.763 | 0.762 | 0.765 | 0.00455 |
| AlexNet + masks |, | 0.173 | 0.333 | 0.118 | 0.00895 |

Plain AlexNet is the best-performing model. Both mask-augmented variants perform *worse* than their unmasked counterparts, and the AlexNet+masks run is not a "worse result", it's a collapsed model: recall pinned at exactly 0.333 (= 1/3 of a balanced 3-class problem) means the network converged to predicting a single class for every input, from epoch 2 onward. The run was manually interrupted (`KeyboardInterrupt`, epoch 0, batch 24/190) on re-execution, i.e. this was a known dead end that wasn't worth re-training to completion.

## Why masking broke AlexNet

`LungsWithMasksDataset` computes `image = image * mask` (element-wise, in `[0, 1]`) *before* the transform pipeline runs. For the AlexNet pipeline that transform includes `Normalize(mean=0.5, std=0.5)`, which maps every masked-out (zero) pixel to exactly **-1**. Depending on the UNet mask's coverage, that can be 40–60% of the frame collapsing to a uniform, saturated negative value feeding directly into AlexNet's first conv layer (11×11 kernel, stride 4), a much larger, more structured perturbation than in the SimpleConvNet pipeline, where the same masking is applied *without* the normalization step and pixels stay at 0.

This is a plausible, code-verifiable explanation for the collapse (large blocks of a extreme, constant value in a deep ReLU network with no gradient clipping and no batchnorm in the classifier head), not a confirmed root cause, no gradient-norm logging exists in the notebook to verify it directly. The original notebook's own conclusion ("gradients either vanish or explode, not sure why") reflects that this wasn't debugged further.

## Limitations

- No fixed random seed, splits and results aren't reproducible run-to-run.
- Train/test split ratio differs between models (80/20 vs. 90/10), so numbers aren't perfectly comparable across the two base architectures.
- No gradient clipping, no LR scheduling, no early stopping, all three would be a reasonable first response to the masked-AlexNet collapse.
- UNet mask quality is not evaluated quantitatively (no IoU/Dice score against ground truth), mask quality is a candidate cause for the degradation and was never isolated as a variable.
- `HybridNet` was abandoned rather than fixed; no diagnosis of its divergence is in the notebook.

## What I'd try next

1. Normalize masked images with mean/std computed *after* masking, or apply the mask post-normalization instead of pre-normalization, so masked-out pixels are 0 rather than -1.
2. Add gradient clipping (`clip_grad_norm_`) before re-attempting AlexNet+masks.
3. Score the UNet's generated masks against ground truth (Dice/IoU) before trusting them as an input transform for classification.
4. Fix the train/test split (single seed, same ratio) across all four experiments for a valid comparison.

## Repository contents

```
Lung_Classification_EN.ipynb  , this notebook, English comments
Lung_Classification_RU.ipynb  , same notebook, Russian comments
notebook_original.ipynb       , unmodified source
create_notebooks.py           , generates the EN/RU variants from the original
requirements.txt
LICENSE
```

`data/` (train/test images, masks, CSV labels) is not included, see `.gitignore`. Point `csv_file`/`root_dir`/`mask_dir` in the dataset classes at your own copy to run it.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook Lung_Classification_EN.ipynb
```

CUDA GPU strongly recommended, UNet alone is ~31M parameters at 256×256 resolution.

## References

- Krizhevsky, Sutskever, Hinton, *ImageNet Classification with Deep Convolutional Neural Networks* (2012)
- Ronneberger, Fischer, Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015)

