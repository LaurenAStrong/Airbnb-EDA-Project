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
users.duplicated().sum()
```

There were no duplicate rows. Now, we check if there are multiple different entries under the same Airbnb user id:

```python
users.id.duplicated().sum()
```

The data shows that there are no users who booked twice in this dataset.

Analyze age outliers:
```python
users[users.age > 100]['age'].describe()
```
* count    2690.000000
* mean      690.957249
* std       877.927570
* min       101.000000
* 25%       105.000000
* 50%       105.000000
* 75%      2014.000000
* max      2014.000000

```python
users[users.age <18]['age'].describe()
```
* count    188.000000
* mean      12.718085
* std        5.764569
* min        1.000000
* 25%        5.000000
* 50%       16.000000
* 75%       17.000000
* max       17.000000

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

Convert to categorical for plotting:
```python
categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method',
    'signup_flow'
]

for categorical_feature in categorical_features:
    users[categorical_feature] = users[categorical_feature].astype('category')
```

## Univariate Analysis – Insights
Age boxplot: Outliers have been removed
```python
sns.boxplot(data=airbnb_df.age, color='#FD5C64')
plt.title('Age Distribution')
```
![Age Distribution Updated with title](https://user-images.githubusercontent.com/91219409/145687398-036af343-4879-418c-a8e1-04f26eba457a.png)


Age dist plot:
```python
sns.distplot(users.age.dropna(), color='#FD5C64')
plt.xlabel('Age')
sns.despine()
```
![Age dist plot](https://user-images.githubusercontent.com/91219409/145686502-72239c18-3a9e-4b24-ae48-ebc9beaf9037.png)


Gender: more females than males. Over 30% unknown gender. 
```python
plt.figure(figsize=(14,8))
order1 = users['gender'].value_counts().index
sns.countplot(data = users, x = 'gender', order = order1, color = '#FD5C64')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
order2 = users['gender'].value_counts()

for i in range(order2.shape[0]):
    count = order2[i]
    strt='{:0.1f}%'.format(100*count / users.shape[0])
    plt.text(i,count+1000,strt,ha='center')
```
![Gender counts](https://user-images.githubusercontent.com/91219409/145686373-fbd5a757-fd92-4784-8b5c-27276e1cc60f.png)


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
![Better plot for desination country](https://user-images.githubusercontent.com/91219409/145686254-2e12d016-05fb-4566-96b6-7420769f41f8.png)


Convert time and date columns into appropriate format:
```python
users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format='%Y%m%d')
```

Time first active: Steady growth until 2014, then lots of growth
```python
sns.histplot(users.timestamp_first_active, color='#FD5C64')
plt.xlabel('Timestamp First Active')
plt.ylabel('Count')
plt.title('First Active Over Time')
```
![First Active Over Time](https://user-images.githubusercontent.com/91219409/145687351-adb6b745-c5bf-4ff2-b4e6-98516e2aed88.png)


```python
users.date_account_created.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')
```
![Date Account Created](https://user-images.githubusercontent.com/91219409/145687169-ef39b7b9-a5f6-42ef-a5ae-4e4237d7e9c6.png)


Devices used:
```python
order1 = users['first_device_type'].value_counts().index
sns.countplot(x='first_device_type', data=users,  order = order1, color = '#FD5C64')
plt.xticks(rotation=-65)
plt.xlabel('Device Type')
plt.ylabel('Count')
plt.title('Device Type Counts')
```
![Device Types Plot corrected](https://user-images.githubusercontent.com/91219409/145721641-6b317916-6430-42d1-9491-75a93fab9974.png)




## Bivariate Analysis – Insights

Device type - Destination plot:
```python
device1_destinations = users.loc[users['first_device_type'] == 'Mac Desktop', 'country_destination'].value_counts()
device2_destinations = users.loc[users['first_device_type'] == 'Windows Desktop', 'country_destination'].value_counts()
device1_destinations.plot(kind='bar', width=width, color='#0BF77D', position=0, label='Mac Desktop', rot=0)
device2_destinations.plot(kind='bar', width=width, color='#FD5C64', position=1, label='Windows Desktop', rot=0)
Bar width
width = 0.4
plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Device Type Counts')
plt.show()
```
![Device Type by Country Plot](https://user-images.githubusercontent.com/91219409/145722597-fecaf779-c9c6-40b9-8d45-ba48b86118a7.png)


Age - Destination plot:
```python
age = 50
less_than_50 = sum(users.loc[users['age'] < age, 'country_destination'].value_counts())
greater_than_50 = sum(users.loc[users['age'] > age, 'country_destination'].value_counts())
younger_destinations = users.loc[users['age'] < age, 'country_destination'].value_counts() / less_than_50 * 100
older_destinations = users.loc[users['age'] > age, 'country_destination'].value_counts() / greater_than_50 * 100
younger_destinations.plot(kind='bar', width=width, color='#0BF77D', position=0, label='Age Less Than 50', rot=0)
older_destinations.plot(kind='bar', width=width, color='#FD5C64', position=1, label='Age Greater Than 50', rot=0)
plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
sns.despine()
plt.show()
```
![Age versus destination plot](https://user-images.githubusercontent.com/91219409/145722776-f7fcf528-1976-423c-b593-bae1eaac1f7c.png)


Gender - Destination plot:
```python
women = sum(users['gender'] == 'FEMALE')
men = sum(users['gender'] == 'MALE')
female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100
male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100
width = 0.4
male_destinations.plot(kind='bar', width=width, color='#0BF77D', position=0, label='Male', rot=0)
female_destinations.plot(kind='bar', width=width, color='#FD5C64', position=1, label='Female', rot=0)
plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
plt.show()
```
![Destination by gender](https://user-images.githubusercontent.com/91219409/145722202-0751eaf7-b0c6-4131-86a6-5fd13d6c38b5.png)


# Part 2 - Metrics Analyses:

<img width="800" alt="Screen Shot 2021-12-12 at 9 46 37 AM" src="https://user-images.githubusercontent.com/91219409/145723448-154ab13b-9004-44d9-a374-28ef76f7a87e.png">
<img width="800" alt="Screen Shot 2021-12-12 at 9 44 39 AM" src="https://user-images.githubusercontent.com/91219409/145723400-4f1c9e57-2da6-4ae2-80b0-0d748ccc906e.png">




