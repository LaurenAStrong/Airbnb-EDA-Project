# Airbnb User Bookings Data Analysis

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

To get the full user dataset, we first concatenate the train and test set that was presplit. We concatenate here instead of merging because we want the rows added on top of on another, and not joined by a common key, like during merging. 

For this task, we used `pandas` which is a well-known data processing library.

```python
users = pd.concat((test_users, train_users), axis = 0, ignore_index= True)
```




