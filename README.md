# NeuroVision AI — Source Code

This package contains Python source code only. It does not include FER-2013 images,
trained model weights, Haar Cascade XML, SQLite records, or a virtual environment.

## Required files to run the dashboard

Place these files in the project:
- `data/raw/fer2013/train/<emotion>/` and `data/raw/fer2013/test/<emotion>/`
  (only needed for training/evaluation)
- `models/best_emotion_model.pth` (required for dashboard predictions)
- `models/haarcascade_frontalface_default.xml` (required for face detection)

## Install

Use Python 3.12 in a fresh virtual environment:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run dashboard

```powershell
python -m streamlit run dashboard/emotion_dashboard.py
```

## Train / evaluate

The training scripts expect the FER-2013 folder structure described above.

```powershell
python training/train_emotion.py
python training/evaluate.py
```

## Standalone webcam

```powershell
python -m src.camera
```

## Important

The model loader expects a PyTorch checkpoint containing `model_state_dict` and
`class_names`, compatible with MobileNetV3-Small. Confirm compatibility with
your own saved checkpoint before using this reconstructed source package.
