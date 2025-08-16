# Personality Classification & Clustering with Azure ML Studio

## SUMMARY
This project demonstrates **Classification** and **Clustering** models built without coding using Azure Machine Learning Studio.

Such models can be valuable for:
- Career Counselling
- Student grouping and collaboration
- Tailoring teaching styles
- Personalized recommendation
- Cross-cultural communication

## OVERVIEW
This project applies Azure Machine Learning Studio to a publicly available personality dataset of 20,000 individuals with 30 behavioral attributes.

Two approaches were used:

- K-Means Clustering (Unsupervised) – to uncover natural groupings in the data without using labels.
- Multiclass Decision Forest Classification (Supervised) – to predict personality type with high accuracy using labeled data.

The project demonstrates the end-to-end machine learning pipeline—from data preparation and normalization to training, testing, and evaluation—using a no-code platform. This highlights how advanced AI techniques can be made accessible to non-programmers while still delivering meaningful business and research insights.

## DATASET INTRODUCTION
Dataset link: [Kaggle – Introvert, Extrovert & Ambivert Classification](https://www.kaggle.com/datasets/miadul/introvert-extrovert-and-ambivert-classification)

- It contains labeled personality data with three main types: Introvert, Extrovert, Ambivert.

- 1 categorical column: ```personality_type``` (target label)

- 30 numerical columns: behavioral attributes (e.g., ```social_energy```, ```alone_time_preference```, ```talkativeness```, ```deep_reflection```, ```group_comfort```, etc.)
  
On analysing the dataset, it was found out that the distribution was balanced across three different categories. 
- Ambiverts make up 33% (6,573 individuals),
- Introverts account for another 33% (6,570 individuals), and
- Extroverts represent 34% (6,857 individuals).
  
This balance ensures no single personality dominates, making it an ideal dataset for both clustering and classification.

<img width="500" height="500" alt="personality_type_distribution" src="https://github.com/user-attachments/assets/0cf69ac1-6ac8-489f-abe9-f704f9ffe551" />

## DATA PREPROCESSING
All data preparation was done using Azure ML Studio:
1. **Review Dataset**- Loaded the dataset, converted it to MLTable for AutoML job in ML Studio.
2. **For Clustering**- Excluded the target label (personality_type) for unsupervised grouping of individuals.
3. **For Classification**- Used **personality_type** as the target column.
4. **Missing data & Normalize Data**- No missing data were found, and all behavioral attributes were scaled to [0,1] for fair comparison.
5. **Split Data**- 70% training, and 30% testing for classification.

**Result**: dataset was clean, scaled, and structured for both pipelines.

## Methodology
### Clustering (Unsupervised)- K-Means
- **Algorithm**- K-Means
- **Number of Clusters**- 3 (aligning with three different personality categories)
- **Input**- 30 normalized behavioral attributes
- **Target**- Discover similarities among individuals.

 <img width="1677" height="935" alt="image" src="https://github.com/user-attachments/assets/bcb31a5c-1a3c-4d84-a609-c1cd771d249e" />
 
### Classification (Supervised)- Multiclass Decision Forest
 - **Algorithm**- Decision Forest (Ensemble of decision trees)
 - **Target Column**- ```personality_type```
 - **Features**- 30 behavioral attributes
 - **Split**- 70% training, 30% testing
 - **Output**- Predictive model with aggregate result from multiple trees.

<img width="1673" height="945" alt="image" src="https://github.com/user-attachments/assets/c59b09d9-8095-42c0-9ee9-b849c8159b05" />


## RESULTS
### Clustering
- **Cluster 1**: High sociability, high activity levels
- **Cluster 2**: Moderate across attributes
- **Cluster 3**: High introspection, lower social engagement

### Classification
- **Very high accuracy**: behavioral metrics are strong predictors of personality.
- **Balanced precision & recall**: model performs equally well across all classes.

<img width="749" height="259" alt="image" src="https://github.com/user-attachments/assets/e097bc0d-754b-490d-a1a9-e5380916f1b2" />

### Conclusion
Using Azure ML Studio, this project demonstrated both unsupervised clustering and supervised classification on personality data:

- K-Means Clustering → revealed natural segmentation based on behavior.

- Decision Forest Classification → achieved 98.28% accuracy in predicting personality types.

These results highlight:

- The dataset’s strong predictive power

- The suitability of Azure ML Studio for end-to-end ML pipelines

- How non-programmers can build impactful machine learning solutions
