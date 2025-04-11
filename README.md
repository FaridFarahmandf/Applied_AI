# Applied AI PRoject


# Museum Image Classifier: Indoor vs. Outdoor

This repository contains the implementation of a binary image classification project to distinguish indoor and outdoor museum scenes using classical machine learning and custom Convolutional Neural Networks (CNNs). The project leverages the dataset and achieves state-of-the-art performance with a custom DeepCNN model, reaching over 97% accuracy.

## Project Overview

The project is divided into two phases:

1. **Phase 1: Classical Machine Learning**
   - Features: Average RGB values extracted from resized images.
   - Models: Decision Tree, Random Forest, Gradient Boosting, and a semi-supervised Decision Tree.
   - Performance: Solid baselines with Gradient Boosting achieving up to 90.1% ROC-AUC.

2. **Phase 2: Deep Learning**
   - Models: Three custom CNNs—SimpleCNN, DeepCNN, and LightCNN—trained on 128x128 normalized RGB images using PyTorch.
   - Performance: DeepCNN excels with 97.1% accuracy and 98.2% ROC-AUC, demonstrating the power of spatial feature learning.

### Key Features
- **Dataset**: Subset of MIT Places dataset (8,119 indoor, 4,221 outdoor museum images).
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Cohen's Kappa, and Matthews Correlation Coefficient.
- **Ablation Study**: Analyzes learning rate, network depth, and regularization impacts.
- **Environment**: Developed in a GPU-accelerated Google Colab environment.

## Repository Structure

```
Applied_AI/
├── AI_Project.ipynb          # Initial exploration and classical ML models
├── FINAL_Project.ipynb       # Final combined code for both phases
├── FINAL_Project_Phase2.ipynb # Deep learning phase (CNN training and evaluation)
├── Project.ipynb             # Early project code (classical ML focus)
├── README.md                 # This file
└── LICENSE                   # Project license (to be added)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/FaridFarahmand/Applied_AI.git
   cd Applied_AI
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Dataset**:
   - Download the MIT Places dataset (`m_museum_indoor` and `m_museum_outdoor` categories) from [http://places.csail.mit.edu/](http://places.csail.mit.edu/).
   - Place the images in the `data/` directory or update the dataset path in `src/dataset.py`.

## Usage

1. **Preprocessing**:
   - Run `notebooks/preprocessing.ipynb` to resize images to 128x128 and apply RGB normalization.

2. **Training Classical Models**:
   - Execute `notebooks/classical_models.ipynb` to train and evaluate Decision Tree, Random Forest, Gradient Boosting, and semi-supervised models.

3. **Training CNNs**:
   - Run `notebooks/cnn_training.ipynb` to train SimpleCNN, DeepCNN, or LightCNN.
   - Hyperparameters (e.g., learning rate=0.001, batch size=32) are configurable in the notebook.

4. **Evaluation**:
   - Use `notebooks/evaluation.ipynb` to compute metrics (Accuracy, F1-Score, ROC-AUC, etc.) and generate visualizations (confusion matrices, ROC curves).

Alternatively, run the pipeline from the command line:
```bash
python src/main.py --model deepcnn --data_path data/ --epochs 20
```

## Results

| Model            | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------|----------|-----------|--------|----------|---------|
| Decision Tree    | 84.7%    | 83.4%     | 83.2%  | 82.4%    | 86.3%   |
| Random Forest    | 87.7%    | 85.5%     | 85.0%  | 85.7%    | 89.2%   |
| Gradient Boosting| 88.8%    | 83.2%     | 86.1%  | 84.6%    | 90.1%   |
| SimpleCNN        | 94.3%    | 93.4%     | 93.1%  | 93.3%    | 96.3%   |
| LightCNN         | 92.3%    | 93.7%     | 91.0%  | 93.5%    | 96.0%   |
| **DeepCNN**      | **97.1%**| **96.7%** | **96.5%**| **96.6%**| **98.2%**|

- **DeepCNN** outperforms all models, excelling in complex scenes like open courtyards.
- **LightCNN** offers a lightweight alternative for resource-constrained environments.
- Classical models provide decent baselines but lag in generalization.

## Future Work
- Incorporate data augmentation to improve robustness.
- Explore transfer learning with lightweight pretrained models.
- Experiment with self-supervised learning for better feature extraction.

## Dependencies
- Python 3.8+
- PyTorch
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- See `requirements.txt` for full list.



## Acknowledgments
- Authors: Raymand Shojaie Aghabalaghe, Farid Farahmand, Amirmohammad Rezaeipour
- Institution: Concordia University
