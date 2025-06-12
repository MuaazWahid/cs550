
# Cauliflower Leaf Disease Classification Using Machine Learning

This project focuses on applying machine learning techniques to classify cauliflower leaf images as either **healthy**, **infected by black rot**, or **attacked by insects**.

Our goal is to support **precision agriculture** by enabling early detection of plant diseases by training AI/ML models to accurately predict whether a cauliflower leaf is infected or not, based on image data.

This can be applied in the real world today with **real-time disease monitoring**, **precision farming**, and **smartphone-based diagnoses for farmers**.

## Team Members

- Glenn Marvin Musoke
- Muaaz Wahid

## Dataset

- **Source**: Kaggle
- **Size**: ~5.3k images (~6.25 GB)
- **Categories**:
  - Healthy
  - Black Rot
  - Insect Hole

> Please note: the dataset came with the original images as well as processed images, where the subject was masked (the background was removed)

Each image was resized, cropped, and optionally grayscaled before being transformed into a 1D NumPy array for modeling.

## Problem Statement


## Preprocessing Steps

- Grayscale the image (optional)
- Crop image (center-focused, 500×500 px)
- Resize to standard dimensions (64×64)
- Flatten into 1D NumPy arrays
- Label encoding of disease types

## Machine Learning Models Used

| Model                   | Accuracy | Notes                                   |
|------------------------|----------|-----------------------------------------|
| SVM (Linear)           | 53%      | Fast but lower accuracy                 |
| SVM (RBF)              | 64%      | Improved recall, especially for NCD     |
| Decision Tree          | 51%      | Overfits with small depth               |
| Random Forest          | 68%      | Best performance overall                |
| Logistic Regression    | 57%      | Decent but less robust                  |
| XGBoost                | 64%      | Strong alternative to Random Forest     |

### Hyperparameter Tuning

Used `GridSearchCV` for tuning the `C` and `gamma` parameters in SVM:
- Best parameters: `C=10`, `gamma='scale'`
- Best CV accuracy: 62%

## Dimensionality Reduction

- Applied **PCA** (6 components) for visualization and potential performance improvement.

## Clustering

- Implemented **DBSCAN** to explore underlying cluster structure post-PCA.
- Visualized cluster distributions in 2D PCA space.

## Key Findings

- Random Forest and XGBoost had the best overall performance.
- Limited training samples for some categories (e.g., NCD) reduced precision.
- Processed grayscale images slightly improved model performance in some configurations.
- Feature importance visualizations provided insights into image region contributions.

## Project Files

| File | Description |
|------|-------------|
| `final_original_images.ipynb` | Image classification on original colored images |
| `final_processed_images.ipynb` | Image classification on masked/processed images |
| `final_gray_original_images.ipynb` | Grayscale image classification |
| `final_gray_processed_images.ipynb` | Grayscale + masked image classification |
| `classify-leaf-diseases.ipynb` | Drafting ideas and rough implementation of ml cauliflower image classification |

## Future Work
- Expand dataset size and diversity for underrepresented classes.
- Explore CNN-based deep learning models for improved image understanding.
- Develop a mobile-friendly API or interface for real-time farmer use.

## Acknowledgments
- Professor Yasin Ceran, who taught this class on Machine Learning/Business Intelligence @ San Francisco Bay University
- Kaggle for the dataset
