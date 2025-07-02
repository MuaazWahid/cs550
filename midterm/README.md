# Poultry Disease Classification

Using ML models to classify poultry images as either **coccidiosis**, **healthy**, **new castle disease**, or **salmonella**.

Supports husbandry by enabling early detection of poultry diseases utilizing ML models that can predict health status, based on image taken from a phone.

This can be applied in the real world today with **real-time disease monitoring** and **smartphone-based diagnoses for farmers**.

## Team Members
- Glenn Marvin Musoke
- Muaaz Wahid

## Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/kausthubkannan/poultry-diseases-detection/data)
- **Size**: ~6.8k images (~8.6 GB)
- **Categories**:
  - coccidiosis
  - healthy
  - new castle disease
  - salmonella

## Preprocessing Steps
- Grayscale the image (optional)
- Crop image (center-focused, 800×800 px)
- Flatten into 1D NumPy arrays
- Label encoding of disease types

## Key Findings

- Random Forest and XGBoost had the best overall performance.
- Limited training samples for some categories (e.g., NCD) reduced precision.
- Processed grayscale images slightly improved model performance in some configurations.
- Feature importance visualizations provided insights into image region contributions.

## Acknowledgments
- Professor Yasin Ceran: taught this class on Machine Learning/Business Intelligence @ San Francisco Bay University
