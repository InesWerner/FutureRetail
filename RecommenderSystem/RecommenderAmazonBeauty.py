# --------------------------------- Recommender System with TruncatedSVD ---------------------------------------

# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


# Loading the Amazon Beauty dataset
# Source: Kaggle (2018). https://www.kaggle.com/skillsmuggler/amazon-ratings
df_ratings = pd.read_csv('ratings_Beauty.csv')
#df_ratings = df_ratings.dropna()
print(df_ratings.head(10))


# Identify Duplicates
duplicates = df_ratings.duplicated(["UserId","ProductId", "Rating", "Timestamp"]).sum()
print('Number of duplicates: ',duplicates)


# Number of users, products and ratings in the ratings_Beauty.csv
print('Number of users: ', len(df_ratings.UserId.unique()))
print('Number of products: ', len(df_ratings.ProductId.unique()))
print('Number of ratings: ', df_ratings.shape[0])


# Popular products grouped by product and rating
popular_products = pd.DataFrame(df_ratings.groupby('ProductId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
print(most_popular.head(10))


# Plot for the most popular products arranged in descending order
params = {'legend.fontsize': 10,
          'legend.handlelength': 5}
plt.rcParams.update(params)

most_popular.head(30).plot(kind="bar")
plt.plot(most_popular.head(30))
plt.xlabel('ProductId')
plt.ylabel('Ratings')
plt.title('Most Popular Products')
plt.show()


# Utility Matrix
# Subset of Amazon Ratings
# Source: Rudrendu, Paul (2021). Recommendation system fpr e-commerce businesses. https://github.com/RudrenduPaul/Python-Ecommerce-recommendation-system-using-machine-learning/blob/master/Recommendation%20System%20-%20Paul.ipynb
amazon_ratings1 = df_ratings.head(10000)

ratings_utility_matrix = amazon_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
print('Utility Matrix:')
print(ratings_utility_matrix.head())


# Transpose of the utility matrix
# Source: (Kite, 2021). TruncatedSVD. https://www.kite.com/python/docs/sklearn.decomposition.TruncatedSVD
matrix = ratings_utility_matrix.T
print('Transposing of the utility matrix:')
print(matrix.head())
X1 = matrix

# Decomposing of the matrix
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(matrix)
print('Decomposed matrix:')
print(decomposed_matrix)


# Correlation matrix
correlation_matrix = np.corrcoef(decomposed_matrix)
print('Correlation matrix')
print(correlation_matrix)


# The customer buys a product with the following ProductId (Random productId)
r = matrix.index[1]
print('ProductId: ' + r)
i = r
product_names = list(matrix.index)
product_ID = product_names.index(i)


# Correlation for products with the product purchased by the customer
correlation_product_ID = correlation_matrix[product_ID]


# Creates a list with the product recommendations from the correlations matrix
Recommend = list(matrix.index[correlation_product_ID > 0.90])
# Removes the product that is already bought by the customer
Recommend.remove(i)


# Top products by the recommendation system to the customer based on the purchase history of all customers
print('You bought the product with the Id ' + r)
print('You may also be interested in the following products: ')
print(Recommend[0:4])


