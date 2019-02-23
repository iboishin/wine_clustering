import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage  
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np

wines = pd.read_csv('C:/Users/moetoKompiutarche/Downloads/Clustering/Clustering_Wine.csv', header = 0)

wines.head()
list(wines)

# Remove consumer_segment for unsupervised clustering
wines_org = wines.drop(columns=['Customer_Segment'])

# Normalize variables to ensure similar scales
wines_scaled = preprocessing.scale(wines_org)
wines_scaled = pd.DataFrame(wines_scaled, columns = wines_org.columns)
wines_scaled.head()

# Create pairplot to see distribution
# Variables more or less normaly distributed
# Total_Phenols and Flavanoids seem to be correlated
sns.pairplot(wines_scaled)
plt.show()

# Factor analysis (using SVD): approximate the number of underlying factors in the data
# Judging by the graph, the 13 predictors approximately load on 3 or 4 factors
U, S, V = np.linalg.svd(wines_scaled)
eigvals = S**2 / np.sum(S**2)

x = np.arange(13) + 1

plt.plot(x, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

############ Determine the number of clusters in the data
# Determine the clustering distance method
linked = linkage(wines_scaled, 'ward')

# Based on the dendrogram, there are three groups
dendrogram(linked)
plt.show() 

############ Perform PCA to visualize the groupings later
wines_pca = PCA(n_components=2).fit(wines_scaled)
wines_pca_2d = pd.DataFrame(wines_pca.transform(wines_scaled))

############ AgglomerativeClustering clustering (hierarchical) with wines_scaled
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
wines_pca_2d['Target_H'] = cluster.fit_predict(wines_scaled)


############ K means (non-hierarchical) clustering with wines_scaled
# specifying random state for consistent cluster groupings
model = KMeans(n_clusters=3, random_state=10)
wines_pca_2d['Target_NH'] = model.fit_predict(wines_scaled)

############ Add back original/true targets
wines_pca_2d['Target_Original'] = wines['Customer_Segment']

############ Visualize the various groupings
# Groupings seem to be more or less accurate
plt.figure(figsize=(10,5))
plt.subplot(1, 3, 1)
plt.scatter(wines_pca_2d[0], wines_pca_2d[1], c=wines_pca_2d['Target_H'])
plt.title('Hierarchical Clustering')
plt.subplot(1, 3, 2)
plt.scatter(wines_pca_2d[0], wines_pca_2d[1], c=wines_pca_2d['Target_NH'])
plt.title('Non-Hierarchical Clustering')
plt.subplot(1, 3, 3)
plt.scatter(wines_pca_2d[0], wines_pca_2d[1], c=wines_pca_2d['Target_Original'])
plt.title('Original Clustering')
plt.show()

############ Calculate classification error for both methods
wines_pca_2d.dtypes

# rename the groups so that they correspond with consumer_segment names
wines_pca_2d.loc[wines_pca_2d.Target_H == 2, 'Target_H_Group'] = 1
wines_pca_2d.loc[wines_pca_2d.Target_H == 0, 'Target_H_Group'] = 2
wines_pca_2d.loc[wines_pca_2d.Target_H == 1, 'Target_H_Group'] = 3

wines_pca_2d.Target_NH = wines_pca_2d.Target_NH + 1

# how many wines are in each group?
counts = pd.DataFrame({'Target_H_Group': pd.value_counts(wines_pca_2d.Target_H_Group),
                       'Target_NH': pd.value_counts(wines_pca_2d.Target_NH),
                       'Target_Original': pd.value_counts(wines_pca_2d.Target_Original)})

# Calculate classification accuracy of the two models
accuracy_score(wines_pca_2d.Target_Original, wines_pca_2d.Target_H_Group)   # accuracy of 92.7%
accuracy_score(wines_pca_2d.Target_Original, wines_pca_2d.Target_NH)  # accuracy of 96.6%

############ Examine the characteristics of each group using NH
# Add predicted values to original wine values
wines_org['Target_NH'] = wines_pca_2d.Target_NH

# Plot the various distributions by group for the various wine characteristics
i = 0

plt.figure(figsize=(14,5))

for x in list(wines_org):
    i = i + 1
    plt.subplot(3, 5, i)
    plt.boxplot([wines_org[x][wines_org.Target_NH == 1].values,
             wines_org[x][wines_org.Target_NH == 2].values, 
             wines_org[x][wines_org.Target_NH == 3].values])
    plt.title(x)
    

plt.subplots_adjust(wspace = 0.25, hspace=0.65)
plt.show()

############ Test the wether the group mean differences are statistically significant
# Ran ANOVA to check mean difference between multiple groups
# Group means on all variables were statistically different with a p-value < 0.01
# This difference highlights the heterogenity between groups and therefore the successful grouping
# Note: an ANOVA tests whether /all/ of the groups are different or the same. As a result,
# there may be two groups with similar means but one with a different mean, which will result in a low p-value
from scipy.stats import f_oneway
for i in list(wines_org):
    f_oneway(wines_org[i][wines_org.Target_NH == 1], wines_org[i][wines_org.Target_NH == 2], wines_org[i][wines_org.Target_NH == 3]).pvalue

############ Export new wines_org dataframe to use in Dash for presentation
wines_org.to_csv('C:/Users/moetoKompiutarche/Downloads/Clustering/wines_org.csv', index=False)


############ Visualize the three groups on the three factors
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

py.sign_in(username='iboishin', api_key='Z8w63YzRc6TuGpZ1rGPJ')

wines_factors = PCA(n_components=3).fit(wines_scaled)
wines_factors = pd.DataFrame(wines_factors.transform(wines_scaled))
wines_factors['Target_NH'] = wines_pca_2d.Target_NH

trace0 = go.Scatter3d(
    x=wines_factors[0][wines_factors.Target_NH == 1].values,
    y=wines_factors[1][wines_factors.Target_NH == 1].values,
    z=wines_factors[2][wines_factors.Target_NH == 1].values,
    mode='markers',
    name = 'Group A',
    marker=dict(
        size=6,
        line=dict(
            color='rgba(214, 12, 140, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

trace1 = go.Scatter3d(
    x=wines_factors[0][wines_factors.Target_NH == 2].values,
    y=wines_factors[1][wines_factors.Target_NH == 2].values,
    z=wines_factors[2][wines_factors.Target_NH == 2].values,
    mode='markers',
    name = 'Group B',
    marker=dict(
        size=6,
        line=dict(
            color='rgba(0, 128, 128, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

trace2 = go.Scatter3d(
    x=wines_factors[0][wines_factors.Target_NH == 3].values,
    y=wines_factors[1][wines_factors.Target_NH == 3].values,
    z=wines_factors[2][wines_factors.Target_NH == 3].values,
    mode='markers',
    name = 'Group C',
    marker=dict(
        size=6,
        line=dict(
            color='rgba(0, 64, 255, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)


data = [trace0, trace1, trace2]
py.plot(data)