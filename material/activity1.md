# Activity 1

> [!IMPORTANT]  
> During the activity, you are encouraged to question every line of code. Feel free to use your preferred LLM to assist with code-related inquiries.


----
## Part 1: Data Analysis of the Titanic Dataset with Python

### Introduction

The sinking of the RMS Titanic in 1912 was a monumental tragedy. While this event has been immortalized in stories and films, data analysis of the Titanic passenger dataset gives us a unique perspective on survival patterns and social dynamics during the disaster. 

In this tutorial, we will analyze the Titanic dataset using Python to uncover insights. Along the way, you'll learn the basics of data analysis, visualization, and deriving actionable conclusions.


### Prerequisites

Make sure you install the following libraries: 
- `pandas` (for data manipulation),
- `seaborn` and `matplotlib` (for data visualization).

Install them with:
```bash
!pip install pandas seaborn matplotlib
```


### Step 1: Import Libraries and Load the Dataset

Let's begin by importing the necessary libraries and loading the Titanic dataset, which is available as a built-in dataset in the Seaborn library:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
data = sns.load_dataset("titanic")
# Display the first few rows of the dataset
print(data.head())
```

### Step 2: Explore the Dataset

Before diving into visualizations, let's explore the dataset to understand its structure and contents.

#### Check Dataset Info
```python
print(data.info())
print(data.describe())
```

#### Check for Missing Values
```python
print(data.isnull().sum())
```
If there are missing values in critical columns like `age`, you might consider filling them (e.g., with the mean or median) or dropping them, depending on the context.

### Step 3: Data Visualizations

Visualization is crucial for uncovering patterns and relationships in the dataset.

#### Age Distribution by Gender
```python
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='age', kde=True, hue='sex')
plt.title('Age Distribution by Gender')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```
**Interpretation**: This plot shows the age distribution for male and female passengers. The KDE curve highlights patterns like age groups with higher passenger concentrations.


#### Survival Rate by Passenger Class and Gender
```python
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='class', y='survived', hue='sex')
plt.title('Survival Rate by Passenger Class and Gender')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()
```
**Interpretation**: This bar chart reveals survival probabilities by class and gender. For instance, women in first class may have had a significantly higher survival rate.

#### Survival Count by Embarkation Port
```python
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='embarked', hue='survived')
plt.title('Survival Count Based on Embarkation Port')
plt.xlabel('Embarkation Port')
plt.ylabel('Count')
plt.show()
```
**Interpretation**: Ports where passengers boarded (C: Cherbourg, Q: Queenstown, S: Southampton) may show different survival patterns.


#### Fare Distribution by Class and Survival
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='class', y='fare', hue='survived')
plt.ylim(0, 300)  # Limiting y-axis for better readability
plt.title('Fare Distribution by Class and Survival')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()
```
**Interpretation**: Higher fares generally correspond to first-class passengers, with varying survival rates.

#### Survival Count Based on Family Size
First, calculate the family size by combining the `sibsp` (siblings/spouses aboard) and `parch` (parents/children aboard) columns:
```python
data['family_size'] = data['sibsp'] + data['parch']
```
Now, visualize:
```python
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='family_size', hue='survived')
plt.title('Survival Count Based on Family Size')
plt.xlabel('Family Size')
plt.ylabel('Count')
plt.show()
```
**Interpretation**: Small families (1-3 members) tend to have better survival outcomes than larger ones.

### Step 4: Key Findings

Based on the above analysis, we can draw the following conclusions:
- **Gender and Survival**: Women had significantly higher survival rates than men.
- **Passenger Class**: First-class passengers enjoyed better survival outcomes, reflecting the influence of socio-economic factors.
- **Family Size**: Solo travelers and small families fared better compared to larger families.
- **Embarkation Port**: Survival rates varied depending on the boarding port.
- **Age and Survival**: Younger passengers, especially children, were more likely to survive.



----
## Part 2: ## Data Analysis of the Iris Dataset with Python

### Introduction

The Iris dataset is one of the most well-known datasets in machine learning. It contains information on iris flowers from three species (*setosa*, *versicolor*, and *virginica*), with measurements of their sepals and petals. In this lab, we will analyze the dataset using Python, create visualizations, and uncover interesting patterns.


![](./iris.png)

> [!NOTE]  
> In the context of the Iris flower dataset, **sepals** and **petals** are two important parts of a flower:
>  
> **Sepal**: This is the outermost part of the flower, usually green, and it protects the flower bud before it blooms. Sepals are often less colorful and sit beneath the petals. In the Iris dataset, the **sepal length** and **sepal width** are measured features that help distinguish between flower species.
>  
> **Petal**: Petals are the colorful parts of the flower that attract pollinators like bees or butterflies. They are located inside the sepals. In the dataset, the **petal length** and **petal width** are additional measured features used for classification.
>  
> The combination of these measurements (sepal length, sepal width, petal length, petal width) provides numerical attributes that are used to classify the Iris flower species: *Setosa*, *Versicolor*, and *Virginica*.


### Prerequisites

Ensure you install the following libraries:
- `pandas` for data manipulation,
- `seaborn` and `matplotlib` for data visualization.

Install them using:
```bash
pip install pandas seaborn matplotlib
```

### Step 1: Import Libraries and Load the Dataset

We'll start by importing the libraries and loading the Iris dataset, which is available as a built-in dataset in the Seaborn library:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
data = sns.load_dataset("iris")
# Display the first few rows
print(data.head())
```


### Step 2: Explore the Dataset

#### Check for Null Values
```python
print(data.isnull().sum())
```
Good news! The Iris dataset doesn't have missing values. This makes it easier to analyze directly.

#### Summary Statistics
```python
print(data.describe())
```
Review basic statistics, such as the mean, median, and standard deviation of each feature.


### Step 3: Data Visualization

#### Sepal Length Distribution by Species
```python
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='sepal_length', kde=True, hue='species')
plt.title('Distribution of Sepal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()
```
**What to Observe**: Compare how the distribution of sepal length varies across the three iris species.


#### Pairwise Relationships
```python
sns.pairplot(data=data, hue='species')
plt.suptitle('Pairwise Relationships Between Features', y=1.02)
plt.show()
```
**What to Observe**: This plot shows scatterplots for every pair of features. Look for clusters or separability between species.



#### Boxplot: Petal Length by Species
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='species', y='petal_length')
plt.title('Petal Length Distribution by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()
```
**What to Observe**: Note how petal length varies significantly between species, with minimal overlap. This makes it an important feature for classification.



### Step 4: Insights and Findings

Based on our analysis, here are some key observations:
1. **Feature Importance**: Features like petal length and petal width are highly discriminative for distinguishing species.
2. **Species Clustering**: *Setosa* forms a distinct cluster, while *versicolor* and *virginica* show some overlap.


<!-- 

### Step 4: Building a Basic Machine Learning Model

Let's build a simple model to classify iris species based on their features.

#### Split the Data
```python
from sklearn.model_selection import train_test_split

# Features and target
X = data.drop("species", axis=1)
y = data["species"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Train a Classifier
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make Predictions
y_pred = clf.predict(X_test)

# Evaluate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```
**What to Observe**: The model should achieve high accuracy (~95%+), given the simplicity and separability of the Iris dataset.
-->


