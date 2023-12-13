# SpiceUpYourLife - A Machine Learning Exploration of Wine 

This wine recommendation model was created as a final project for the University of Pennsylvania Data Analysis & Visualization Bootcamp. We were challenged to solve, analyze, or visualize a problem using machine learning (ML) 

![glasses](./Resources/wine_photo.jpg) | width=300 height=200

## Description

The goal of this project is to use Machine Learning to find your preferred wine variety based on your palate. The original Kaggle dataset contained wines of various varieties, made in different countries, which were then described by several critics. Using this dataset, the sample size was narrowed down to wines produced in the top 12 countries, all of which had >1,000 distinctive wines reviewed. To predict which wines would best fit a user's taste profile, several machine learning models were applied to the dataset, such as Word2Vec, TSNE, PCA and KMeans. Using the resulting clusters, a taste description input can be manipulated, the model can determine which cluster the description would fall into, and return a list of wine varieties with similar profiles the user should try. The user interface portion of the project is a website, created using flask, html, and css, which allows a user to input a taste description and return a list of wine varieties.

ADD MORE INFO! ...Additional information - web scraping of Wine Folly

## Requirements


### Dependencies

This script was tested using a development environment containing Python 3.11. In addition, the following dependencies are required to run various aspects of the project: pandas, numpy, pathlib, gensim, nltk, sklearn, hvplot, seaborn, and matplotlib.

The project can also be run on Google Colab if dependencies are missing from the dev environment.

### Installation

Clone repo: git clone https://github.com/desireetrevino/SpiceUpYourLife.git

## Overview of Analysis

### Cleaning the Data

The original dataset had ~130K data points. The information was paired down to the 12 countries which each produced >1000 of the wines described in hopes of eliminating some of the noise in the model visualizations. 

### Preprocess Data

In order to convert the descriptions to vectors using Word2Vec, the description column had to first be manipluated. The Natural Language Toolkit (NLTK) was used to drop stop words, lemmatize, and tokenize the descriptions of each wine. 

### Compile, Train, and Evaluate the Models

* Word2Vec was used to train the model, it could return a list of words with similar vector points with the highest accuracy returning 70-80%. The model was also tested to show the similarty between different words. The descriptions were then converted into vector representations.

* TSNE 

* K Means

* PCA

### Results

ADD MORE! ...webpage

### Summary

ADD MORE! ...limited success. how to improve models, etc. 

## Resources

* Data for this project was collected from Kaggle:
    https://www.kaggle.com/datasets/mysarahmadbhat/wine-tasting

* The article "40 Wine Descriptions abd What They Really Mean" by Wine Folly was referenced:
    https://winefolly.com/tips/40-wine-descriptions/

* Royalty free photographs were found at pexels.com

### Collaborators
GitHub Usernames: crystalleelucas, desireetrevino, eldiscala, jackieoc, katyphillips, mtguadamuzruth, mwiley608