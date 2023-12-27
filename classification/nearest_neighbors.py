import numpy as np
import pandas as pd
import os


from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def training_fitting(model, feature_list):
    combined_features, feature_vectorizers = training_transforming_features(feature_list)
    model.fit(combined_features)

    return feature_vectorizers



def training_transforming_features(feature_list):

    feature_vectorizers =[]
    features_vectorized = []
    for feature in feature_list:
        feature_vectorizer = TfidfVectorizer()
        feature = feature_vectorizer.fit_transform(feature)
        feature_vectorizers.append(feature_vectorizer)
        features_vectorized.append(feature)
    
    combined_features = hstack(features_vectorized)

    return combined_features, feature_vectorizers



def get_nearest_neighbors(model, new_feature_list, feature_vectorizers):
    new_combined_features = new_point_transforming_features(new_feature_list, feature_vectorizers)
    distances, indices = model.kneighbors(new_combined_features)

    return distances, indices



def new_point_transforming_features(new_feature_list, feature_vectorizers):
    
    features_vectorized = []
    n = 0
    for feature in new_feature_list:
        feature_vectorizer = feature_vectorizers[n]
        feature = feature_vectorizer.transform([feature])
        features_vectorized.append(feature)
        n += 1
    
    combined_features = hstack(features_vectorized)

    return combined_features



