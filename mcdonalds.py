import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from statsmodels.graphics.mosaicplot import mosaic
import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)


# Read the data
data = pd.read_csv(r"C:\\Python310\\Data\\mcdonalds.csv")  # Replace "your_data.csv" with the actual file path

# Debug: Check data loading
print("Shape of data after loading:", data.shape)

# Drop rows with non-numeric 'VisitFrequency' values
non_numeric_rows = data[~data['VisitFrequency'].apply(lambda x: x.isdigit())]
print("Rows with non-numeric 'VisitFrequency' values:")
print(non_numeric_rows)

# Convert 'VisitFrequency' to numeric based on some criteria
visit_frequency_mapping = {
    'Every day': 7,
    'Every two days': 3.5,
    'Every three days': 2.33,
    'Every four days': 1.75,
    'Every five days': 1.4,
    'Every six days': 1.17,
    'Every week': 1,
    'Every two weeks': 0.5,
    'Every three weeks': 1 / 3,
    'Every four weeks': 0.25,
    'Every month': 0.08,
    'Every two months': 0.04,
    'Every three months': 0.03,
    'Every four months': 0.02,
    'Every six months': 0.01,
    'Once a year': 1 / 12
}

data['VisitFrequency'] = data['VisitFrequency'].map(visit_frequency_mapping)

# Drop rows with NaN values in 'VisitFrequency'
data = data.dropna(subset=['VisitFrequency'])

# Encode 'Like' column using Label Encoding
label_encoder = LabelEncoder()
data['Like'] = label_encoder.fit_transform(data['Like'])

# Debug: Check 'Like' encoding
print("Encoded 'Like' values:")
print(data['Like'].unique())

# Drop non-numeric columns for PCA
data_numeric = data[['Age', 'VisitFrequency', 'Like']].copy()

# Debug: Check the inclusion of 'Like' in data_numeric
print("Columns in data_numeric:")
print(data_numeric.columns)

# Principal Component Analysis (PCA)
pca = PCA()

# Debug: Check if there's any data for PCA
if len(data_numeric) > 0:
    MD_pca = pca.fit_transform(data_numeric)
else:
    print("No data available for PCA.")

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=1234)
kmeans.fit(data_numeric)

# Visualization
plt.figure(figsize=(9, 4))
sns.scatterplot(x="VisitFrequency", y="Like", data=data_numeric, hue=kmeans.labels_, s=400, palette="Set1")
plt.title("Segment Evaluation Plot")
plt.xlabel("Visit Frequency")
plt.ylabel("Like")
plt.show()

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, random_state=1234)
gmm.fit(data_numeric)

# Mosaic Plot
k4_labels = kmeans.labels_
ct = pd.crosstab(k4_labels, data['Gender'])
plt.figure()
mosaic(ct.stack(), gap=0.01)
plt.show()

# Boxplot
df = pd.DataFrame({'Segment': k4_labels, 'Age': data['Age']})
plt.figure()
df.boxplot(by='Segment', column='Age')
plt.title('Parallel box-and-whisker plot of age by segment')
plt.suptitle('')
plt.show()

# Statistical Analysis
segment = pd.concat([pd.DataFrame({'Segment': k4_labels}),
                     data.groupby(k4_labels)['Age'].mean().reset_index(drop=True),
                     data.groupby(k4_labels)['VisitFrequency'].mean().reset_index(drop=True),
                     data.groupby(k4_labels)['Like'].mean().reset_index(drop=True),
                     data.groupby(k4_labels)['Gender'].apply(lambda x: x.mode()[0]).reset_index(drop=True)], axis=1)

# Debug: Check the segments
print(segment)

# Scatter plot for segment evaluation
plt.figure(figsize=(9, 4))
sns.scatterplot(x="VisitFrequency", y="Like", data=segment, hue="Segment", s=400, palette="Set1")
plt.title("Segment Evaluation Plot")
plt.xlabel("Visit Frequency")
plt.ylabel("Like")
plt.show()

