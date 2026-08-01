# 🔪 Knife Detection using Deep Learning (PyTorch & TIMM)

![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)
![TIMM](https://img.shields.io/badge/TIMM-PyTorch%20Image%20Models-blue.svg)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-green.svg)
![Coursework](https://img.shields.io/badge/Coursework-EEEM066%20FML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

A modular PyTorch deep learning framework for **knife detection and image classification** utilizing transfer learning with `timm` backbones (defaulting to **EfficientNet-B0**). 

The project features customizable data augmentation pipelines, custom dataset loaders, custom learning rate schedulers, flexible optimization choices (Adam, SGD, RMSprop), mixed-precision training (`torch.cuda.amp`), and evaluation metrics.

---

## 📌 Features

- **Backbone Architecture**: Powered by `timm` (PyTorch Image Models), defaulting to pre-trained `tf_efficientnet_b0` fine-tuned over multi-class weapon image targets.
- **Custom PyTorch Dataset Loader**: Optimized `knifeDataset` (`data.py`) handling multi-label and single-label CSV annotations with RGB image transformations.
- **Data Augmentation**: Supports Random Rotation, Color Jittering (brightness, contrast, saturation, hue), Random Vertical/Horizontal Flips, and Random Erasing (`--random-erase`).
- **Flexible Optimization & Schedulers**: Configurable choices via `src/optimizers.py` (Adam, SGD with Nesterov, RMSprop) and `src/lr_schedulers.py` (StepLR, CosineAnnealing, MultiStepLR).
- **Mixed Precision Training**: Uses `torch.cuda.amp.autocast` and `GradScaler` for reduced GPU memory footprint and faster iteration times.

---

## 📂 Repository Structure

```text
knife-detection-deep-learning/
├── README.md                           # Comprehensive documentation & execution guide
├── requirements.txt                    # Python library requirements (PyTorch, TIMM, OpenCV)
├── .gitignore                          # Excludes raw images, saved weights, & caches
├── Training.py                         # Main training execution loop
├── Testing.py                          # Evaluation & inference script
├── args.py                             # Command-line argument parser & hyperparameter configuration
├── data.py                             # Custom PyTorch Dataset definition
├── utils.py                            # Metrics evaluation, logging, & checkpointing helpers
├── train.sh                            # Shell script execution wrapper for model training
├── test.sh                             # Shell script execution wrapper for model testing
├── src/                                # Model training modules
│   ├── lr_schedulers.py                # Learning rate scheduler initializers
│   ├── optimizers.py                   # Optimizer initializers
│   └── transforms.py                  # PyTorch data augmentation & image preprocessing
└── dataset/                            # CSV annotation splits
    ├── classes.csv                     # Target class mappings
    ├── train.csv                       # Training dataset index
    ├── validation.csv                  # Validation dataset index
    └── test.csv                        # Testing dataset index
```

---

## ⚡ Installation & Environment Setup

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU recommended for training (supports CPU mode for testing)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/knife-detection-deep-learning.git
cd knife-detection-deep-learning
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Training the Model

You can launch training using the provided shell script or run `Training.py` directly:

```bash
# Using the convenience shell script
bash train.sh
```

Or execute via Python CLI with explicit parameters:

```bash
python Training.py \
  --dataset_location /path/to/knife_images \
  --train_datacsv dataset/train.csv \
  --val_datacsv dataset/validation.csv \
  --model_mode tf_efficientnet_b0 \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0003 \
  --optim adam \
  --brightness 0.2 \
  --contrast 0.2 \
  --random-erase
```

### 2. Evaluating / Testing the Model

Run testing against the test dataset split:

```bash
# Using shell script
bash test.sh
```

Or via Python CLI specifying the saved checkpoint:

```bash
python Testing.py \
  --dataset_location /path/to/knife_images \
  --test_datacsv dataset/test.csv \
  --model-path saved_checkpoint.pth \
  --evaluate-only
```

---

## ⚙️ Key Hyperparameter Options (`args.py`)

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--model_mode` | `str` | `tf_efficientnet_b0` | Model architecture backbone from `timm` |
| `--dataset_location` | `str` | *Required* | Path to the directory containing raw images |
| `--epochs` | `int` | `10` | Total number of training epochs |
| `--batch_size` | `int` | `32` | Batch size for DataLoader |
| `--learning_rate` | `float` | `0.0003` | Initial learning rate |
| `--optim` | `str` | `adam` | Optimization algorithm (`adam`, `sgd`, `rmsprop`) |
| `--resized_img_height` | `int` | `224` | Image resize target height |
| `--resized_img_width` | `int` | `224` | Image resize target width |

---

## 📁 Dataset Folder Requirements

> [!NOTE]
> Raw image datasets are excluded from this repository via `.gitignore` to keep the repository under GitHub size limits.
>
> 1. Download or organize the image dataset into a local folder (e.g., `./images/`).
> 2. Ensure image filenames match the entries in `dataset/train.csv`, `dataset/validation.csv`, and `dataset/test.csv`.

---

## 🎓 Academic Context

Developed for **EEEM066 Fundamentals of Machine Learning (FML)** coursework at the **University of Surrey**.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
