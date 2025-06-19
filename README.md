#  Customer Segmentation with K-Means Clustering

A comprehensive Python solution for retail customer segmentation using machine learning techniques. This tool helps businesses understand their customer base by automatically grouping customers based on their purchasing behavior, demographics, and spending patterns.

##  Table of Contents

- [Features]
- [Installation]
- [Quick Start]
- [Data Format]
- [Usage Guide]
- [Visualizations]
- [Understanding Results]
- [Advanced Usage]
- [Troubleshooting]
- [License]

##  Features

###  **Intelligent Clustering**
- Automatic optimal cluster detection using Elbow Method and Silhouette Analysis
- Robust K-means implementation with multiple initializations
- Standardized data preprocessing for improved clustering performance

###  **Rich Visualizations**
- 2D scatter plots showing customer segments
- Age distribution analysis by cluster
- PCA visualization for high-dimensional data
- Interactive cluster characteristics heatmap

###  **Business Insights**
- Detailed statistical analysis for each customer segment
- Demographic and behavioral profiles
- Actionable insights for marketing strategies
- Export functionality for further analysis

###  **Easy to Use**
- Simple, intuitive API
- Comprehensive error handling
- Sample data generation for testing
- Flexible data input options

##  Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn warnings
```

### Download the Script

1. Download the `customer_segmentation.py` file
2. Place it in your working directory
3. You're ready to go!

##  Quick Start

### Using Sample Data (for testing)

```python
from customer_segmentation import CustomerSegmentation

# Initialize the tool
segmentation = CustomerSegmentation()

# Load sample data
segmentation.load_data(sample_data=True)

# Run complete analysis
segmentation.preprocess_data()
segmentation.find_optimal_clusters()
segmentation.perform_clustering()
segmentation.visualize_clusters()
segmentation.analyze_clusters()
```

### Using Your Own Data

```python
from customer_segmentation import CustomerSegmentation

# Initialize the tool
segmentation = CustomerSegmentation()

# Load your CSV file
segmentation.load_data('path/to/your/customer_data.csv')

# Run complete analysis
segmentation.preprocess_data()
segmentation.find_optimal_clusters()
segmentation.perform_clustering()
segmentation.visualize_clusters()
segmentation.analyze_clusters()

# Save results
segmentation.save_results('customer_segments_results.csv')
```

##  Data Format

Your CSV file should contain the following columns:

| Column Name | Data Type | Description | Required |
|-------------|-----------|-------------|----------|
| `CustomerID` | Integer/String | Unique identifier for each customer | ✅ |
| `Gender` | String | Customer gender (Male/Female) | ❌ |
| `Age` | Integer | Customer age in years | ✅ |
| `Annual Income (k$)` | Float | Annual income in thousands of dollars | ✅ |
| `Spending Score (1-100)` | Float | Spending score assigned by the mall | ✅ |

### Example Data Format:

```csv
CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
```

##  Usage Guide

### Step-by-Step Process

#### 1. **Initialize the Tool**
```python
segmentation = CustomerSegmentation()
```

#### 2. **Load Your Data**
```python
# Option A: Load from CSV file
segmentation.load_data('your_data.csv')

# Option B: Use sample data for testing
segmentation.load_data(sample_data=True)
```

#### 3. **Preprocess the Data**
```python
segmentation.preprocess_data()
```
This step:
- Handles missing values
- Encodes categorical variables (Gender)
- Standardizes numerical features
- Prepares data for clustering

#### 4. **Find Optimal Number of Clusters**
```python
segmentation.find_optimal_clusters(max_clusters=10)
```
Automatically determines the best number of customer segments using:
- **Elbow Method**: Finds the point where adding clusters doesn't significantly improve performance
- **Silhouette Analysis**: Measures how well-separated the clusters are

#### 5. **Perform Clustering**
```python
segmentation.perform_clustering()
```
Applies K-means algorithm with the optimal number of clusters.

#### 6. **Visualize Results**
```python
segmentation.visualize_clusters()
```
Generates multiple visualizations to understand your customer segments.

#### 7. **Analyze Segments**
```python
segmentation.analyze_clusters()
```
Provides detailed statistical analysis and business insights.

#### 8. **Save Results**
```python
segmentation.save_results('output_filename.csv')
```

##  Visualizations

The tool generates several types of visualizations:

### 1. **Elbow Method & Silhouette Analysis**
- Helps determine the optimal number of clusters
- Shows the trade-off between cluster count and performance

### 2. **Income vs Spending Score Scatter Plot**
- Main segmentation visualization
- Shows customer clusters in 2D space
- Includes cluster centroids marked with red X's

### 3. **Age Distribution by Cluster**
- Histogram showing age patterns in each segment
- Helps understand demographic characteristics

### 4. **PCA Visualization**
- Projects high-dimensional data to 2D
- Useful when you have many features
- Shows explained variance for each component

### 5. **Cluster Characteristics Heatmap**
- Color-coded matrix of cluster features
- Easy comparison across segments
- Darker colors indicate higher values

##  Understanding Results

### Sample Output Analysis:

```
 CLUSTER 0 (n=45 customers)
----------------------------------------
Age: 32.1 ± 8.5 years
Annual Income: $55.2k ± $15.3k
Spending Score: 75.8 ± 12.4
Gender: {'Female': 28, 'Male': 17}
```

### Typical Customer Segments:

1. **High-Value Customers**: High income, high spending
2. **Potential Customers**: High income, low spending
3. **Loyal Customers**: Low income, high spending
4. **Price-Sensitive**: Low income, low spending
5. **Average Customers**: Medium income, medium spending

## 🔧 Advanced Usage

### Custom Number of Clusters

```python
# Skip automatic detection and use specific number
segmentation.perform_clustering(n_clusters=5)
```

### Access Raw Results

```python
# Get cluster labels
labels = segmentation.labels

# Get scaled data
scaled_data = segmentation.scaled_data

# Get the trained model
kmeans_model = segmentation.kmeans

# Access original data with cluster labels
clustered_data = segmentation.data
```

### Modify Clustering Parameters

```python
from sklearn.cluster import KMeans

# Create custom KMeans model
custom_kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20,  # More initializations
    max_iter=500  # More iterations
)

# Use custom model
segmentation.kmeans = custom_kmeans
segmentation.labels = custom_kmeans.fit_predict(segmentation.scaled_data)
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### **1. "No data loaded" Error**
```
ValueError: No data loaded. Please load data first.
```
**Solution**: Make sure to call `load_data()` before other operations.

#### **2. Missing Column Error**
```
KeyError: 'Annual Income (k$)' not found
```
**Solution**: Check your CSV column names match exactly (including spaces and special characters).

#### **3. Empty Dataset After Preprocessing**
```
Dataset becomes empty after removing missing values
```
**Solution**: Check for missing values in your data and clean them before loading.

#### **4. Memory Error with Large Datasets**
```
MemoryError: Unable to allocate array
```
**Solution**: 
- Use data sampling for large datasets
- Consider using MiniBatchKMeans for very large datasets

#### **5. Poor Clustering Results**
**Symptoms**: All customers in one cluster or very uneven clusters
**Solutions**:
- Check data scaling and preprocessing
- Try different numbers of clusters manually
- Ensure your data has meaningful patterns to cluster

### Data Quality Checklist

 **Before running the analysis, ensure:**
- No duplicate CustomerIDs
- Reasonable value ranges (Age: 18-100, Income: >0, Spending: 1-100)
- Minimal missing values (<10% of data)
- At least 50+ customers for meaningful segmentation

## License

This project is licensed under the MIT License - see the License file for details.

##  Support

### Getting Help
- **Documentation**: Check this README first
- **Issues**: Create a GitHub issue for bugs or questions

### Best Practices
- Always start with sample data to understand the tool
- Visualize your raw data before clustering
- Validate results make business sense
- Use domain knowledge to interpret clusters

---
