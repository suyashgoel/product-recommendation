import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare, chi2_contingency
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def plot_num_var(df, num_var):
    plt.hist(df[num_var], bins=10, edgecolor='k', alpha=1.0)

    plt.xlabel(num_var)
    plt.ylabel('Count')

    print(num_var, "does not appear to contain any outliers. Therefore we can proceed.\n")

    plt.show()

# Calculates Cramér's V (assocation) between our categorical variables
def cramers(df):
    num_cols = df.shape[1]
    cramer_matrix = np.zeros((num_cols, num_cols))
    for cat1 in range(num_cols):
        for cat2 in range(num_cols):
            ct = pd.crosstab(df[df.columns[cat1]], df[df.columns[cat2]])
            c, p, dof, expected = chi2_contingency(ct)
            n = sum(np.sum(ct))
            k = min(ct.shape)
            cramer_matrix[cat1, cat2] = math.sqrt(c / (n * k))
    cramer_matrix = pd.DataFrame(
        cramer_matrix, columns=df.columns, index=df.columns)
    return cramer_matrix

def visualize_assocations(df):
    # Makes a heat map to check for correlation between numeric variables
    # Since we have low correlations, we can proceed
    plt.figure()
    sns.heatmap(df.select_dtypes(
    include=['int', 'float']).corr(), vmin=-1, vmax=1, annot=True)
    plt.title('Correlation Matrix')

    print('Since correlation between our numeric variables is relatively low, we can proceed.\n')

    plt.show()

    # Selects our categorical variables, or those of type 'object'
    cat_var = df.select_dtypes('object')

    # Creates a heatmap indicating the level of association between categorical variables
    # Since no variables are highly associated, we can proceed
    sns.heatmap(cramers(cat_var), vmin=0, vmax=1, annot=True)
    plt.title('Cramér\'s V Assocation Matrix')

    print('Cramér\'s V value measures association between categorical variables. Since values are relatively low between variables, we can proceed.\n')

    plt.show()
    
def scale(df):
    # StandardScaler scales numeric variables so KMeans can work more efficiently
    sc = StandardScaler()

    # scaled_df has numeric variables
    scaled_df = pd.concat([df["Age"],
                      df["Previous Purchases"]], axis=1)

    # Fit the standard scaler using our numeric variables
    sc.fit(scaled_df)

    # Apply a transformation to our numeric variables according to this model
    scaled_df = sc.transform(scaled_df)

    # Convert the transformed numeric data  into a DataFrame with the appropriate column names
    scaled_df = pd.DataFrame(scaled_df, columns=["Age", "Previous Purchases"])

    # Append categorical variable columns to scaled_df
    scaled_df = pd.concat([scaled_df, df['Gender'], df['Location'], df['Season'],
                      df['Subscription Status'], df['Frequency of Purchases']], axis=1)

    # print('We use StandardScaler to scale our numeric variables so they are equally weighted\n')

    return scaled_df, sc

def encode(df):

    # Create dummy variables for each categorical variable since KMeans only takes numeric data
    # This means we create a feature for each categorical variable's value (dropping the first since we won't lose information). A 1 indicates that a customer has the dummy variable attribute, and a 0 means it doesn't have it.
    # For example, we get a dummy variable Male, which will be 1 for customers with Gender == 'Male' and 0 for all other customers
    gender = pd.get_dummies(
    df['Gender'], drop_first=True, dtype=int)
    loc = pd.get_dummies(
    df['Location'], drop_first=True, dtype=int)
    season = pd.get_dummies(
    df['Season'], drop_first=True, dtype=int)
    status = pd.get_dummies(
    df['Subscription Status'], drop_first=True, dtype=int)
    freq = pd.get_dummies(
    df['Frequency of Purchases'], drop_first=True, dtype=int)

    # Construct DataFrame for our encoded and scaled data, adding our dummy variables to the scaled data
    encoded_and_scaled_df = df
    encoded_and_scaled_df = pd.concat(
    [df, gender, loc, season, status, freq], axis=1)

    # Remove the original categorical variables
    encoded_and_scaled_df.drop(columns=['Gender', 'Location', 'Season',
                           'Subscription Status', 'Frequency of Purchases'], inplace=True)

    # print('KMeans uses numeric variables, so we create dummy variables for categorical variables. This means we create binary columns for each category, with 1 indicating a customer being a part of a category. We replace these dummy variables without our original categorical variables.\n')

    return encoded_and_scaled_df

def reduce_dimensionality(df):
    # Define a PCA instance, intending to reduce to dimensions while our number of dimensions account for at least 80% of variance in the data. We fit this using our encoded and scaled data.
    pca = PCA(n_components=0.8)
    pca.fit(df)

    # Apply the PCA transformation, and make a DataFrame we will pass into KMeans.
    pca_df = pd.DataFrame(pca.transform(
    df))

    return pca_df, pca

def form_clusters(df, pca_df):

    # Define the number of clusters using user input
    num_clusters = int(input(
    'Enter the number of clusters you want to segment users into based on their demographics. This must be more than the number of recommendations you want: '))

    # Initialize a KMenas instance with num_clusters clusters, and we use init = 'k-means++', which means we begin with better centroid initialization
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++')

    # Fit KMeans using the 3-dimension DataFrame
    kmeans.fit(pca_df)

    # Predict the clusters customers are a part of
    labels = kmeans.predict(pca_df)

    # Add a column to our original DataFrame with each customer's cluster value
    df["Clusters"] = labels
    pd.options.mode.chained_assignment = None

    print("We have completed clustering.")

    return df, kmeans


def predict_cluster(encoded_and_scaled_df, sc, pca, kmeans):
    print("Now we can make predictions on a new user, allowing us to make product recommendations based on the preferences of similar users.\n")

    # Take user input to generate a customer we want to create recommendations for
    age = int(input('Enter the age of an existing customer you want recommendations for: '))
    gender = input('Enter the gender of the customer: ')
    location = input('Enter the US State the customer resides in: ')
    season = input('Enter the season the customer is making a purchase during: ')
    subscription_status = input('Provide the subscription status of the user: ')
    previous_purchases = int(
    input('Enter the number of purchases the customer the customer has made: '))
    frequency_of_purchases = input(
    'Enter the frequency at which the customer makes purchases: ')

    # Make an array with the user-inputted values
    row = [age, gender, location, season, subscription_status,
       previous_purchases, frequency_of_purchases]

    # Make a DataFrame using our inputted values corresponding to each column of updated_df, and we set our column values to be those of updated_df (same order)
    sample = pd.DataFrame([row], columns=['Age', 'Gender', 'Location', 'Season',
                      'Subscription Status', 'Previous Purchases', 'Frequency of Purchases'])

    # Scale numeric features for the new customer using the previous StandardScaler instance we fitted (using the same model)
    scaled_sample = pd.DataFrame(sc.transform(
    sample[['Age', 'Previous Purchases']]), columns=['Age', 'Previous Purchases'])
    # Add categorical variables to the scaled data
    scaled_sample = pd.concat([scaled_sample, sample['Gender'], sample['Location'], sample['Season'],
                          sample['Subscription Status'], sample['Frequency of Purchases']], axis=1)


    # Gets a list of our original encoded and scaled DataFrame columns, ignoring the first two since we already have Age and Previous Purchases
    bin_cols = list(encoded_and_scaled_df.columns)
    bin_cols = bin_cols[2:len(bin_cols)]

    # Copy our scaled sample onto a new DataFrame to include dummy variables
    encoded_and_scaled_sample = scaled_sample

    # Iterate through the columns, adding dummy variables to encoded_and_scaled_sample
    # If there is a match in a dummy variable name and one of the customer's categorical labels, label the dummy variable with 1. Otherwise, label it 0
    for col in bin_cols:
        if encoded_and_scaled_sample['Gender'][0] in col or encoded_and_scaled_sample['Location'][0] in col or encoded_and_scaled_sample['Season'][0] in col or encoded_and_scaled_sample['Subscription Status'][0] in col or encoded_and_scaled_sample['Frequency of Purchases'][0] in col:
            new_col = pd.DataFrame([1], columns=[col])
        else:
            new_col = pd.DataFrame([0], columns=[col])
        encoded_and_scaled_sample = pd.concat(
            [encoded_and_scaled_sample, new_col], axis=1)

    # Drop the original categorical variables since we have encoded categorical variables
    encoded_and_scaled_sample.drop(columns=[
                               'Gender', 'Location', 'Season', 'Subscription Status', 'Frequency of Purchases'], inplace=True)

    # Transforms the encoded and scaled sample using our previously fitted PCA instance (for dimensality reduction), and converts this into a DataFrame
    pca_sample = pd.DataFrame(pca.transform(
        encoded_and_scaled_sample))

    # Makes a prediction of the 3-dimension sample using the fitted KMeans, and retrieves the cluster value
    [cluster] = kmeans.predict(pca_sample)

    return cluster, sample

def make_recommendations(df, cluster):
    similar_customers = df.loc[df["Clusters"] == cluster]

    # Reduces potential recommendations to those with at least 3/5 star reviews for the products similar customers bought
    potential_recommendations = df[df['Review Rating'] >= 3]

    # Retrieves counts for the potential recommendations
    items = potential_recommendations['Item Purchased'].value_counts()

    # Takes user input to determine the number of product recommendations and prints them
    num_recs = int(
    input('Enter the number of recommendations you want for the customer: '))
    print("Here are the top", num_recs,
          "recommendations for this customer based on what similar customers have been satisifed with!\n")

    # Prints the [num_recs] products with the highest frequencies in items
    for i in range(num_recs):
        print(items.index[i])
    print()
    
def visualize(df, cluster, sample):

    # Creates a Colors column for visualization, labeling customers in the same cluster as the customer we made predictions on as orange, and all others as blue
    for index in df.index:
        if df.loc[index, 'Clusters'] == cluster:
            df.loc[index, 'Colors'] = 'orange'
        else:
            df.loc[index, 'Colors'] = 'blue'

    print('We can visualize relationships between variables on a scatterplot to help visualize our clusters with meaning. Let\'s plot our numeric variables, age and previous purchases.\n')

    # Gets user input to plot two variables against each other for the customers to visualize relationships within the cluster and out of it
    feature_one = 'Age'
    feature_two = 'Previous Purchases'

    non_similar_customers = df[df['Colors'] == 'blue']
    similar_customers = df[df['Colors'] == 'orange']

    # Plots the customer data for these features, using the colors we specify in df['Colors']
    plt.scatter(non_similar_customers[feature_one],
            non_similar_customers[feature_two], c=non_similar_customers['Colors'])

    plt.scatter(similar_customers[feature_one],
            similar_customers[feature_two], c=similar_customers['Colors'])

    # Retrieves the new customer's labels for these features
    sample_feature_one = sample['Age']
    sample_feature_two = sample['Previous Purchases']

    # Plots the new customer's data in a larger size with a red color
    plt.scatter(sample_feature_one, sample_feature_two, c='red', s=100)

    # Labels the plots with the feature names
    plt.xlabel(feature_one)
    plt.ylabel(feature_two)

    print('We plot', feature_one, 'and', feature_two,
          'with the red point being the customer we made predictions on, the orange points being customers in the same cluster, and blue points being all other customers.\n')

    # Shows the scatterplot
    plt.show()

def preprocess(selected_features_df, df):
    # Scales our numeric data
    scaled_df, sc = scale(selected_features_df)

    print("We have standardized our numeric data. This centers our data at mean 0 with standard deviation 1.\n")

    # Encodes our categorical variables
    encoded_and_scaled_df = encode(scaled_df)

    print("We have encoded our categorical data. KMeans only uses numeric variables, so we can turn each categorical variable into many numeric variables by creating a binary column for each category, with 1 meaning a user is a part of that category and 0 meaning a user is not a part of that category.\n")

    # Reduces dimensionality
    pca_df, pca = reduce_dimensionality(encoded_and_scaled_df)

    print("KMeans doesn't work well with high dimensionality, so we transform our data into principal components (lower dimensionality) with minimal data loss.\n")

    # Forms clusters
    selected_features_df, kmeans = form_clusters(selected_features_df, pca_df)
    df = pd.concat([df, selected_features_df['Clusters']], axis=1)

    # Generates customer data and predicts the cluster the customer is a part of
    cluster, sample = predict_cluster(encoded_and_scaled_df, sc, pca, kmeans)

    # Recommends products based on similar customers
    make_recommendations(df, cluster)

    # Visualizes clusters in 2-d
    visualize(df, cluster, sample)
    
def main():
    # This silences FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None  # default='warn

    print("Reading customer data.\n")

    # Reads the user data (.csv) as a DataFrame
    df = pd.read_csv(
        "/Users/suyashgoel/product-recommendation/shopping_behavior_updated.csv")
    
    print('Selecting our features as age, gender, location, season, subscription status, previous purchases, and frequency of pruchases.\n')

    # Selects the features we believe may help predict good recommendations
    selected_features_df = df[['Age', 'Gender', 'Location', 'Season',
                               'Subscription Status', 'Previous Purchases', 'Frequency of Purchases']]
    
    print("We will now visualize our numeric variables to identify potential outliers, since KMeans, the clustering algorithm we are using, is sensitive to outliers.\n")

    # Plots a histogram to detect any ages that appear to be outliers
    # There are no outliers
    plot_num_var(selected_features_df, 'Age')

    # Plots a histogram to detect any number of previous purchases that appear to be outliers
    # # There are no outliers
    plot_num_var(selected_features_df, 'Previous Purchases')

    print('We will now visualize association between variables since multicollinearity is problematic for KMeans, giving disproportionately high weight to highly corrleated variables.\n')
    visualize_assocations(selected_features_df)

    print("We will now begin to preprocess our data by standardizing numeric variables, encoding categorical variables, and reducing dimensionality\n")
    preprocess(selected_features_df, df)

if __name__ == "__main__":
    main()