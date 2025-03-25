# **Part 1: Data and Datasets for Classical Machine Learning**  

### **Introduction**  

- **The Role of Data in Machine Learning**:  
  In classical ML, **data is the foundation** of model performance. Unlike traditional programming, where rules are explicitly coded, ML models learn from data to recognize patterns and make predictions. **If the data is messy, biased, or incomplete, the model’s performance suffers**—highlighting the importance of proper dataset preparation.  

- **Classical ML vs. Modern AI (LLMs)**:  
  - Classical ML models typically rely on **structured datasets** (tabular data with predefined features).  
  - Modern AI, such as **LLMs**, deals with **unstructured datasets** (e.g., raw text, images, or audio).  
  - Despite these differences, **the principles of data preprocessing remain the same**: cleaning, analyzing, engineering features, and preparing data for modeling.  

- **Objective of This Section**:  
  - Understand **why clean and structured data is crucial** for classical ML.  
  - Learn practical techniques for **cleaning, exploring, and transforming datasets** for predictive modeling.  

---

### **1. Importance of Data in Classical ML**  

- **Core Concept**:  
  - ML models **learn patterns from data**, not from explicit programming rules.  
  - The **quality, quantity, and representativeness of data** directly impact model accuracy and generalization.  

- **Real-World Data Challenges**:  
  - **Incomplete data**: Missing values in key features.  
  - **Noisy data**: Inconsistent or incorrect records.  
  - **Outliers**: Extreme values that distort the model.  
  - **Imbalanced data**: When one category dominates the dataset (e.g., fraud detection datasets with 99% non-fraud cases).  

 

---

### **2. Understanding the Dataset**  

- **Dataset Schema & Terminology**:  
  - A structured dataset is typically a **table with rows and columns**.  
  - **Features (independent variables)**: Input variables used to predict an outcome (e.g., age, salary, education level).  
  - **Target (dependent variable)**: The variable to be predicted (e.g., whether a customer will buy a product).  

- **Types of Features**:  
  - **Numerical Features**: Continuous values (e.g., income, age, temperature).  
  - **Categorical Features**: Discrete categories (e.g., gender, country, job title).  
  - **Ordinal Features**: Categorical variables with a meaningful order (e.g., education level: High School < Bachelor’s < Master’s).  

- **Example: Titanic Survival Dataset**  
  - **Features**: Passenger class, age, gender, fare price.  
  - **Target variable**: Whether the passenger survived (1) or not (0).  

---

### **3. Data Cleaning and Preprocessing**  

- **Common Data Issues & Solutions**:  

  1. **Handling Missing Data**  
  - Missing values can occur due to incomplete records or sensor failures.  
  - **Solutions**:  
    - Fill missing values using **mean/median/mode** for numerical data.  
    - Use **forward fill or backward fill** for time-series data.  
    - Drop rows/columns if missing data is excessive.  

  2. **Handling Incorrect Data Types**  
  - Example: A column storing numerical values as text (e.g., "20" instead of 20).  
  - Solution: Convert data types (`int`, `float`, `string`) accordingly.  

  3. **Handling Outliers**  
  - Example: A dataset of salaries where most are between $40k–$100k, but one entry is $10M.  
  - Solutions:  
    - **Cap extreme values** (e.g., replace 99th percentile values with a threshold).  
    - **Use transformations** (e.g., logarithmic scaling).  

  4. **Removing Duplicates**  
  - Duplicate rows can distort model training. Use `df.drop_duplicates()` in pandas to remove them.  

- **Why Data Cleaning Matters**:  
  - Poorly processed data **introduces bias**, leading to incorrect predictions.  
  - Clean data ensures **better generalization** across unseen examples.  

---

### **4. Exploratory Data Analysis (EDA)**  

- **Why Perform EDA?**  
  - Detects missing values, outliers, and incorrect data types.  
  - Helps understand distributions, correlations, and relationships between features.  

- **EDA Techniques**:  

  1. **Descriptive Statistics**  
  - `df.describe()` provides mean, median, min/max values, and standard deviations.  
  - Identifies potential skewness in numerical variables.  

  2. **Visualizing Data Distributions**  
  - **Histograms**: Show frequency distribution (e.g., age distribution of Titanic passengers).  
  - **Boxplots**: Identify outliers and variability.  
  - **Scatter plots**: Show relationships between two numerical variables (e.g., house price vs. square footage).  

  3. **Checking Correlations**  
  - Use a **correlation matrix** (`df.corr()`) to find relationships between variables.  
  - Example: Does higher fare correlate with higher survival rates?  

---

### **5. Feature Engineering**  

- **Goal**: Transform raw data into a **better representation** for ML models.  

- **Techniques**:  

  1. **Handling Categorical Variables**  
  - **Label encoding**: Convert categories to numbers (e.g., Male → 0, Female → 1).  
  - **One-hot encoding**: Convert categorical variables into multiple binary columns.  

  2. **Creating New Features**  
  - Example: In the Titanic dataset, **combine sibling/spouse count and parent/child count into "family size"**.  

  3. **Scaling Numerical Features**  
  - ML models like **logistic regression and k-means clustering** perform better when numerical values are scaled.  
  - Common techniques:  
    - **Min-max scaling**: Rescales features to [0,1].  
    - **Standardization**: Rescales to a mean of 0 and standard deviation of 1.  

---

### **6. Preparing the Dataset for Modeling**  

- **Final Steps Before Training**:  
  1. **Select relevant features and target variable**.  
  2. **Split dataset into training (80%) and test (20%) sets**.  
  3. **Balance dataset if needed** (e.g., oversampling minority classes in imbalanced datasets).  

- **Outcome**: The dataset is now **clean, structured, and ready for training**.  

---

### **7. Challenges in Real-World Data**  

- **Noisy Data**: Human errors, typos, or measurement inconsistencies.  
- **Bias in Data**: Overrepresentation of certain groups leading to biased predictions.  
- **Small Datasets**: Limited data can cause overfitting; solutions include **cross-validation** and **data augmentation**.  

---

### **Conclusion & Transition**  

- **Key Takeaways**:  
  - Data quality **directly impacts model performance**.  
  - Cleaning, analyzing, and transforming data **is essential before training ML models**.  
  - **Feature engineering and scaling improve accuracy** and efficiency.  


---
## Further Exploration: 

- Pandas
  - [Video ~50min: Pandas—Pandas for Data Science](https://www.youtube.com/watch?v=Yp3fccNNfjQ)
  - [Crash course](https://www.kaggle.com/learn/pandas)
- [Data Visualization](https://www.kaggle.com/learn/data-visualization)
- [How to Handle Missing Values](https://www.kaggle.com/code/alexisbcook/missing-values)
- [Video ~50min: NumPy for Data Science](https://www.youtube.com/watch?v=EmA_TuC2Vdk)
- Video Course: Intro to Machine Learning with Python
  - [Part 1: Welcome and Project Setup](https://youtu.be/rdaG53khzv0)
  - [Part 2: Exploratory Data Analysis](https://youtu.be/6BagRiSY1ds)
  - [Part 3: Train Test Split and Baseline Modeling](https://youtu.be/MufPx3L7nXM)




<!-- 
---
## Visualization: [Choosing Plot Types](https://www.kaggle.com/code/alexisbcook/choosing-plot-types-and-custom-styles) 

- Trends - A trend is defined as a pattern of change.
  - `sns.lineplot` - Line charts are best to show trends over a period of time, and multiple lines can be used to show trends in more than one group.
- Relationship - There are many different chart types that you can use to understand relationships between variables in your data.
  - sns.barplot - Bar charts are useful for comparing quantities corresponding to different groups.
  - sns.heatmap - Heatmaps can be used to find color-coded patterns in tables of numbers.
  - sns.scatterplot - Scatter plots show the relationship between two continuous variables; if color-coded, we can also show the relationship with a third categorical variable.
- Distribution - We visualize distributions to show the possible values that we can expect to see in a variable, along with how likely they are.
  - sns.histplot - Histograms show the distribution of a single numerical variable.
  - sns.kdeplot - KDE plots (or 2D KDE plots) show an estimated, smooth distribution of a single numerical variable (or two numerical variables).
  - sns.jointplot - This command is useful for simultaneously displaying a 2D KDE plot with the corresponding KDE plots for each individual variable.


<img src="./visualization.png" width="50%"> 
-->