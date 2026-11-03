# Brain tumour MRI classification (EfficientNetB0)

Four-class classifier (glioma, meningioma, pituitary tumour, no tumour) on 2D MRI slices, using transfer learning from ImageNet EfficientNetB0, with a Grad-CAM heatmap and a web demo.

> **Research prototype. Not a medical device. Not for clinical use.** Trained and tested only on a public dataset; no patient data is included in this repository.

**Live demo:** _Hugging Face Space link to add_

## Results (held-out Testing folder, 1,600 images, 400 per class)

| Metric | Value |
|---|---|
| Accuracy | 86.9% |
| Macro F1 | 0.865 |
| Macro AUC (one-vs-rest) | 0.975 |

| Class | Recall | Precision | AUC |
|---|---|---|---|
| Glioma | 70.3% | 93.1% | 0.943 |
| Meningioma | 79.8% | 81.6% | 0.960 |
| No tumour | 98.3% | 93.6% | 0.998 |
| Pituitary | 99.3% | 81.5% | 0.998 |

Main confusions: gliomas predicted as meningioma (67 of 400) and meningiomas predicted as pituitary (60 of 400). See `results/metrics.json` and the figures in `results/`.

## Method

- Data: Kaggle `masoudnickparvar/brain-tumor-mri-dataset` (Training 5,600 and Testing 1,600 slices, combining figshare, SARTAJ and Br35H sources). 15% of Training was held out for validation.
- Model: EfficientNetB0 (ImageNet) with light augmentation. Phase 1: head only (Adam 1e-3, 10 epochs). Phase 2: last 60 layers unfrozen with BatchNorm frozen (Adam 1e-5, 15 epochs).
- Explainability: Grad-CAM.

## Limitations (read before quoting the numbers)

- Classification is per 2D slice, not per patient or volume. Near-duplicate slices from the same patient may appear in both Training and Testing, so the scores can be optimistic.
- The dataset mixes sources with different scanners and protocols and there is no external validation.
- Glioma recall is only 70%: a quarter of gliomas are missed or mislabelled. This model would not be usable for screening.
- This is a new project (not the earlier YOLO-based work); the numbers above are from this notebook only.
- Grad-CAM shows where the network looked, not that it found a lesion.

## Run it

Train (Google Colab, free T4 GPU): open `brain_tumor_mri_training.ipynb` and run all cells; it saves `brain_tumor_effnetb0.keras` and `metrics.json`.

Demo locally:

```bash
cd demo
pip install -r requirements.txt
# put brain_tumor_effnetb0.keras in this folder
python app.py
```

## Author

Abdullah Ghani, data and AI engineering student at ECE Paris.
