# Random Forest Classifier Implementation

##  Overview

This project implements a **Random Forest Classifier from scratch** using Python, demonstrating the core concepts of ensemble learning through the construction of Decision Trees and their aggregation for classification tasks. The implementation is applied to the Wine Quality dataset for predicting wine quality ratings.

##  Key Features

- **Custom Decision Tree Implementation**: Built from the ground up without relying on sklearn's tree implementation
- **Random Forest Ensemble**: Implements bagging and feature randomness for improved generalization
- **Entropy-based Splitting**: Uses information gain for optimal decision boundaries
- **Grid Search**: Includes hyperparameter tuning functionality
- **Visualization Support**: Integrates with matplotlib and seaborn for data analysis

## Dataset
**Wine Quality Dataset (Red Wine)**
- **Samples**: 1,599 red wine samples
- **Features**: 11 physicochemical properties
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target**: Quality rating (3-8 scale, treated as classification)
- **Source**: UCI Machine Learning Repository
## Decision Tree Components

1. **Node Structure**
   - Stores feature index and threshold for splitting
   - Contains references to left and right child nodes
   - Leaf nodes store class predictions

2. **Splitting Criteria**
   - Uses entropy to measure node impurity
   - Calculates information gain for feature selection
   - Implements minimum samples split and maximum depth constraints

3. **Tree Building Process**
   - Recursive partitioning of feature space
   - Stops when maximum depth or minimum impurity is reached
   - Handles both numerical features

##  Key Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `n_trees` | Number of trees in the forest | 100 |
| `max_depth` | Maximum depth of each tree | float('inf') |
| `min_sample_split` | Minimum samples required to split a node | 2 |
| `min_impurity` | Minimum impurity decrease required for split | 1e-7 |
| `max_feature` | Number of features to consider for best split | None (uses sqrt) |

##  Model Metrics
The notebook evaluates the model using:
- **Accuracy Score**: Overall correct predictions
- **Confusion Matrix**: Detailed breakdown of predictions vs actual
- **Classification Report**: Precision, recall, and F1-score per class

## Additional Notes
- This is an educational implementation focusing on understanding the algorithm
- For production use, consider optimized libraries like scikit-learn
- The implementation uses pure Python/NumPy for transparency
- Performance can be improved with vectorization and parallel processing
