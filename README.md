# Airbnb User Bookings Analysis

This repository documents the process of extracting insights from multiple Airbnb datasets, joining them on a common key, cleaning them, analyzing for insights, and presenting the results with graphs and business-impactful summaries.

The data is Airbnb user booking data from 2014 that was re-released in 2015. The data is in the data folder of this repository. 

## Requirements

This project uses the following Python libraries
* `pandas` : For analyzing and getting insights from datasets.
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

The next step we perform is to calculate the number of null entries with a `Python` loop over the columns.

```python
 for i in users.columns:
    sum_null = users[i].isnull().sum()
    if sum_null != 0:
        print(i + " has {} null values.".format(sum_null))
        print()
```

The `print` output shows:

* date_first_booking has 186639 null values.
* age has 116866 null values.
* first_affiliate_tracked has 6085 null values.
* country_destination has 62096 null values.

Following checking for Null values, we also check for how many unique values are in each column. For example, checking how many genders are reported, as well as how many different browsers and device types are reported. It would be helpful for marketing and product to know which device types, etc bookings are being made through.

```python
for i in users.columns:
    n_unique  = users[i].nunique()
    if n_unique != 0:
        print(i + " has {} unique values.".format(n_unique))
        print()  
```

The `print` output shows:

* id has 275547 unique values (which is the total number of rows in our dataset)
* date_account_created has 1726 unique values.
* timestamp_first_active has 275547 unique values.
* date_first_booking has 1976 unique values.
* gender has 4 unique values.
* age has 145 unique values.
* signup_method has 4 unique values.
* signup_flow has 18 unique values.
* language has 26 unique values.
* affiliate_channel has 8 unique values.
* affiliate_provider has 18 unique values.
* first_affiliate_tracked has 7 unique values.
* signup_app has 4 unique values.
* first_device_type has 9 unique values.
* first_browser has 55 unique values.
* country_destination has 12 unique values.

## Data Integrity – Insights

* We see that the number of id’s is identical to the number of rows in our dataset.
* There are 4 genders listed as unique indicating that taking a look at those 4 genders would be useful since more than just male/female are listed.
* We will later take a look at the different browser and device types to see which had the best and lowest conversion rates for booking.
* We will also take a look at which languages and countries of destination are most and least common. 
* Date of first booking has a higher-than-average missing value count, so that will need to be further explored. My hypothesis is that most of those are because users entered into the signup flow, but did not book, thus they did not have a date of first booking to report.

# Data Cleaning Needed:


Clean gender:
```python
users.loc[users.gender == 'OTHER', 'gender'] = np.nan
users.gender.value_counts()
users.loc[users.gender == '-unknown-', 'gender'] = np.nan
```
Clean age:
Replace outlier ages with NaN. 
```python
airbnb_df.loc[airbnb_df.age > 100, 'age'] = np.nan
airbnb_df.loc[airbnb_df.age <15, 'age'] = np.nan
```


## Univariate Analysis – Insights

Mean age is 36:
```python
sns.boxplot(data=users.age)
```
![Users Age Boxplot](https://user-images.githubusercontent.com/91219409/145685371-0d2f20db-6aa2-4013-b77a-9de78b0d33eb.png)



Gender: more females than males. Over 30% unknown gender. 
```python
plt.figure(figsize=(14,8))
order1 = users['gender'].value_counts().index
sns.countplot(data = users, x = 'gender', order = order1, color = sns.color_palette()[0])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
order2 = users['gender'].value_counts()

for i in range(order2.shape[0]):
    count = order2[i]
    strt='{:0.1f}%'.format(100*count / users.shape[0])
    plt.text(i,count+1000,strt,ha='center')
```
![Gender Distribution Airbnb Project](https://user-images.githubusercontent.com/91219409/145685009-9238ad81-bae8-4c11-96c9-5396bc1ffeed.png)


Country of destination:
```python
Country: Mostly unknown, next highest is USA.
plt.figure(figsize=(14,8))
order1 = users['country_destination'].value_counts().index
sns.countplot(data = users, x = 'country_destination', order = order1, color = sns.color_palette()[0])
plt.xlabel('Country Destination')
plt.ylabel('Count')
plt.title('Country Destination')
order2 = users['country_destination'].value_counts()
```
![Age boxplot Airbnb Project User Booking Data](https://user-images.githubusercontent.com/91219409/145685059-d4ba2511-ffe3-469a-acf6-862376243539.png)


Time first active: Steady growth until 2014, then lots of growth
```python
sns.histplot(users.timestamp_first_active)
```
![Histogram of users first active](https://user-images.githubusercontent.com/91219409/145685242-bcc8db4c-ddf4-4ae9-8168-c8a27908e847.png)



## Bivariate Analysis – Insights



## Multivariate Analysis – Insights


