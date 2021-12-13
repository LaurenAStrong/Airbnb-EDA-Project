<img width="1066" alt="Screen Shot 2021-12-12 at 10 23 18 AM" src="https://user-images.githubusercontent.com/91219409/145724671-33d5ce81-2afa-4ac0-ad90-d703454fce6f.png">



# Airbnb User Bookings Analysis

This repository documents the process of extracting insights from multiple Airbnb datasets, joining them, cleaning them, analyzing for insights, and presenting the results with graphs and business-impactful summaries.

The first data is Airbnb user booking data. We use this data to perform exploratory data analysis for Airbnb, providing stakeholders with key insights about Airbnb users.

The second part of this document outlines a second dataset, from Airbnb as well. This one provides product usage data and marketing data, and we will use it to extract insights for both marketing and product.

## Requirements

This project uses the following Python libraries
* `pandas` : For analyzing and getting insights from datasets.
* `NumPy` : For fast matrix operations.
* `matplotlib` : For creating graphs and plots.
* `seaborn` : For enhancing the style of matplotlib plots.
* `datetime` : For transforming date and time variables.

## Data Extraction

For this task, we used `pandas` which is a well-known data processing library.

First, we bring in the train and test data, preparing to combine them in the Python command in the next step.

```python
test_users = pd.read_csv('/Users/Desktop/airbnb_data/test_users.csv')
train_users = pd.read_csv('/Users /Desktop/airbnb_data/train_users_2.csv')
test_users.shape
train_users.shape
```

To get the full user dataset, we concatenate the train and test set that was presplit. We concatenate here instead of merging because we want the rows added on top of one another, and not joined by a common key, like during merging. 

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

Following checking for null values, we also check for how many unique values are in each column. For example, checking how many genders are reported, as well as how many different browsers and device types are reported. It would be helpful for marketing and product to know which device types, etc bookings are being made through.

```python
for i in users.columns:
    n_unique  = users[i].nunique()
    if n_unique != 0:
        print(i + " has {} unique values.".format(n_unique))
        print()  
```

The `print` output shows:

* id has 275547 unique values 
* timestamp_first_active has 275547 unique values.
* gender has 4 unique values.
* age has 145 unique values.
* signup_method has 4 unique values.
* language has 26 unique values.
* affiliate_channel has 8 unique values.
* affiliate_provider has 18 unique values.
* signup_app has 4 unique values.
* first_device_type has 9 unique values.
* first_browser has 55 unique values.
* country_destination has 12 unique values.

There are 4 genders listed, which needs exploring since we expected just female/male. 

Convert to categorical for plotting:
```python
categorical_variables = [
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

for i in categorical_variables:
    users[i] = users[i].astype('category')
```


## Data Cleaning
Here, we clean gender and transform the “-unknown-“ input as well as the “OTHER” input into nulls:
```python
users.loc[users.gender == 'OTHER', 'gender'] = np.nan
users.gender.value_counts()
users.loc[users.gender == '-unknown-', 'gender'] = np.nan
```

We now analyze the age variable, and in particular looks at the range of extreme values for users that listed their age as above 100, and below 18 (the legal limit for booking on Airbnb). The below analyses take a look at these 2 outlier groups:

```python
users[users.age > 100]['age'].describe()
```
* count    2690
* mean      690
* std       877
* min       101
* 25%       105
* 50%       105
* 75%      2014
* max      2014

```python
users[users.age <18]['age'].describe()
```
* count    188
* mean      12
* std        5
* min        1
* 25%        5
* 50%       16
* 75%       17
* max       17


Here, we clean the age variable to remove those outliers. Replace outlier ages with nulls. 
```python
users.loc[users.age > 100, 'age'] = np.nan
users.loc[users.age <18, 'age'] = np.nan
```

Before we begin our date and time analyses, we need to convert the date and time columns to the appropriate formats, so we do so below:
```python
users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format='%Y%m%d')
```


## Univariate Analysis – Insights
We check the spread of age, and see that outliers have indeed been eliminated. 

```python
sns.boxplot(data=users.age, color='#FD5C64')
plt.title('Age Distribution')
```
![Age Distribution Updated with title](https://user-images.githubusercontent.com/91219409/145687398-036af343-4879-418c-a8e1-04f26eba457a.png)

With an age distribution plot, we see a more clear pattern of the distribution of ages. The mean is around 34.  
```python
sns.distplot(users.age.dropna(), color='#FD5C64')
plt.xlabel('Age')
sns.despine()
```
![Age dist plot](https://user-images.githubusercontent.com/91219409/145686502-72239c18-3a9e-4b24-ae48-ebc9beaf9037.png)


During the gender analysis, we see that there are more females than males, and there are about 30% null genders listed. 

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

In our country analysis, we see that most folks listed “NDF”/unknown as their destination, followed by the USA.

```python
plt.figure(figsize=(14,8))
plot_order = users['country_destination'].value_counts().index
sns.countplot(data = users, x = 'country_destination', order = plot_order, color = '#FD5C64')
plt.xlabel('Country Destination')
plt.ylabel('Count')
plt.title('Country Destination')
```
![Better plot for desination country](https://user-images.githubusercontent.com/91219409/145686254-2e12d016-05fb-4566-96b6-7420769f41f8.png)


After looking at the time first active variable, we see that growth is consistently positive, and takes off in 2014. 

```python
users.date_account_created.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')
```
![Date Account Created](https://user-images.githubusercontent.com/91219409/145687169-ef39b7b9-a5f6-42ef-a5ae-4e4237d7e9c6.png)


Desktops beat out other devices by a landslide.

```python
order1 = users['first_device_type'].value_counts().index
sns.countplot(x='first_device_type', data=users,  order = order1, color = '#FD5C64')
plt.xticks(rotation=-65)
plt.xlabel('Device Type')
plt.ylabel('Count')
plt.title('Device Type Counts')
```
![Device Types Plot corrected](https://user-images.githubusercontent.com/91219409/145721641-6b317916-6430-42d1-9491-75a93fab9974.png)


Tuesdays are the most active days for users. 
```python
weekdays = []
for date in users.date_account_created:
    weekdays.append(date.weekday())
weekdays = pd.Series(weekdays)

sns.barplot(x = weekdays.value_counts().index, y=weekdays.value_counts().values, order=range(0,7))
plt.xlabel('Week Day')
sns.despine()
```
![Days of week active users](https://user-images.githubusercontent.com/91219409/145862569-871a8bae-40a1-40da-b2ef-4b0c342d98f1.png)



## Bivariate Analyses

Here we perform a device type by destination analysis. Macs are widely used for US.  

```python
device1_destinations = users.loc[users['first_device_type'] == 'Mac Desktop', 'country_destination'].value_counts()
device2_destinations = users.loc[users['first_device_type'] == 'Windows Desktop', 'country_destination'].value_counts()
device1_destinations.plot(kind='bar', width=width, color='#0BF77D', position=0, label='Mac Desktop', rot=0)
device2_destinations.plot(kind='bar', width=width, color='#FD5C64', position=1, label='Windows Desktop', rot=0)
width = 0.4
plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Device Type Counts')
plt.show()
```
![Device Type by Country Plot](https://user-images.githubusercontent.com/91219409/145722597-fecaf779-c9c6-40b9-8d45-ba48b86118a7.png)


In the age compared to destination plot, we see that Italy and “Other” are listed as destinations more in users above the age of 50, compared to users under the age of 50. The under 50 leads almost every other destination compared to the over 50 age group.

```python
age = 50
less_than_50 = sum(users.loc[users['age'] < age, 'country_destination'].value_counts())
greater_than_50 = sum(users.loc[users['age'] > age, 'country_destination'].value_counts())
less_than__destinations = users.loc[users['age'] < age, 'country_destination'].value_counts() / less_than_50 * 100
greater_than_destinations = users.loc[users['age'] > age, 'country_destination'].value_counts() / greater_than_50 * 100
less_than_destinations.plot(kind='bar', width=width, color='#0BF77D', position=0, label='Age Less Than 50', rot=0)
greater_than_destinations.plot(kind='bar', width=width, color='#FD5C64', position=1, label='Age Greater Than 50', rot=0)
width = 0.4
plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
sns.despine()
plt.show()
```
![Age versus destination plot](https://user-images.githubusercontent.com/91219409/145722776-f7fcf528-1976-423c-b593-bae1eaac1f7c.png)


In the gender compared to destination plot, we see that males list their destination to Canada more frequently than females do. 

```python
f = sum(users['gender'] == 'female')
m = sum(users['gender'] == 'male')
female_dest = users.loc[users['gender'] == 'female', 'country_destination'].value_counts() / f * 100
male_dest = users.loc[users['gender'] == 'male', 'country_destination'].value_counts() / m * 100
width = 0.4
male_dest.plot(kind='bar', width=width, color='#0BF77D', position=0, label='Male', rot=0)
female_dest.plot(kind='bar', width=width, color='#FD5C64', position=1, label='Female', rot=0)
plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
plt.show()
```
![Destination by gender](https://user-images.githubusercontent.com/91219409/145722202-0751eaf7-b0c6-4131-86a6-5fd13d6c38b5.png)






# Further Analyses:

<img width="800" alt="Screen Shot 2021-12-12 at 4 23 52 PM" src="https://user-images.githubusercontent.com/91219409/145735834-03ae22d6-0b06-4399-a30d-3205d1202688.png">


```python
airbnb_data.contact_channel_first.value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)
plt.xlabel('Contact Method')
sns.despine()
airbnb_book_it = airbnb_data[airbnb_data['contact_channel_first'] == 'contant_me']
airbnb_contact_me.ts_booking_at.value_counts().sum()
airbnb_book_it = airbnb_data[airbnb_data['contact_channel_first'] == 'book_it']
airbnb_book_it.ts_booking_at.value_counts().sum()
airbnb_instant_book = airbnb_data[airbnb_data['contact_channel_first'] == 'instant_book']
airbnb_instant_book.ts_booking_at.value_counts().sum()
```
![Contact method plot](https://user-images.githubusercontent.com/91219409/145860282-7cc1e7ea-7b4e-464a-8714-5607515ce658.png)

The "Contact Me" feature (where a user first messages a host and engages in messaging before deciding to book later) had 911 successful bookings out of 12828 uses for a conversion rate of 7.10%. The "Book It" feature, where a user makes a payment upfront, but the host has to accept the reservation request before the booking goes through, had 3983 successful bookings out of 8366 requests for a conversion rate of 47.61%. Lastly, the "Instant Book" feature, where a user can book instantly without having to wait for a host to respond, had a conversion rate of 100% with the highest sum of bookings as well at 6693. From this, we can see that "Instant Book" has a great conversion rate, and that the "Contact Me" feature has a substantially lower conversion rate. 

Recommendation to Product: Based on the conversion rates, I recommend we incentivize users to use the "Instant Book" feature over the "Contact Me" feature but would like to do more in-depth analyses with statistical analyses before making a final recommendation. Our next steps for this project are to analyze the Contact Me bookings further. In our data, we have the number of interactions, and I hypothesize that the number of interactions may have statistical significance within the Contact Me feature. I believe the Contact Me feature may be a worthwhile feature despite it's low conversion rate - for example, new users to the platform may favor the feature so they can get to know the hosts and grow trust on the platform. This analysis is something worth digging into by segmenting past versus new users of the Contact Me feature, as well as to perform statistical significance tests to see if Simpson's Paradox may be happening with the Contact Me feature conversion rate. 



