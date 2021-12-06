# Airbnb User Bookings Analysis

This repository documents the process of extracting insights from multiple Airbnb datasets, joining them on a common key, cleaning them, analyzing for insights, and presenting the results with graphs and business-impactful summaries.

The data is Airbnb user booking data from 2014 that was re-released in 2015. The data is in the data folder of this repository. 

## Requirements

This project uses the following Python libraries
* `pandas` : For analysing and getting insights from datasets.
* `NumPy` : For fast matrix operations.
* `matplotlib` : For creating graphs and plots.
* `seaborn` : For enhancing the style of matplotlib plots.
* `datetime` : For transforming date and time variables.

## Data Extraction

The Airbnb data that I processed can be accessed at the below link, and can be found in the data folder of this repository. Url:  
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

For this task, we used `pandas` which is a well-known data processing library.

First, we bring in the train and test data, preparing to combine them in the Python command in the next step.

```python
test_users = pd.read_csv('/Users/Desktop/airbnb_data/test_users.csv')
train_users = pd.read_csv('/Users /Desktop/airbnb_data/train_users_2.csv')
test_users.shape
train_users.shape
```

To get the full user dataset, we concatenate the train and test set that was presplit. We concatenate here instead of merging because we want the rows added on top of on another, and not joined by a common key, like during merging. 

```python
users = pd.concat((test_users, train_users), axis = 0, ignore_index= True)
users.shape
```

## Data Integrity

Before beginning any data process, it is imperative to check for data integrity. By checking the shape of the datasets, we see that the train and test data was successfully combined to join into the users dataset. 

Now, we will check if there are any duplicate rows in the users dataset.

```python
 users[users.duplicated(keep = False)]
```

There were no duplicate rows. Now, we check if there are multiple different entries under the same Airbnb user id:

```python
 users[users.duplicated(['id'],keep = False)]
```

The data shows that there are no users who booked twice in this dataset.



