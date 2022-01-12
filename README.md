# Shopify Data Science Challenge

by Samuel Abhishek Jebakumar

* TOC
{:toc}


### 1) Importing Data and Dependancies

We start our analysis by importing the required dependacies after installing them into our virtual environment.


```python
import pandas as pd
import numpy as np
import plotly.express as px
import kaleido
```


```python
raw_data = pd.read_excel('./data/2019 Winter Data Science Intern Challenge Data Set.xlsx')
```

### 2) Feature Engineering

#### Overview

The columns (features) seem fairly interpretable. Each row represents a transaction (single instance of shopping) by a user at a shop


```python
prep_data = raw_data.copy()

display(raw_data.head())

prep_data.info()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>shop_id</th>
      <th>user_id</th>
      <th>order_amount</th>
      <th>total_items</th>
      <th>payment_method</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>53.0</td>
      <td>746.0</td>
      <td>224.0</td>
      <td>2.0</td>
      <td>cash</td>
      <td>2017-03-13 12:36:56.190</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>92.0</td>
      <td>925.0</td>
      <td>90.0</td>
      <td>1.0</td>
      <td>cash</td>
      <td>2017-03-03 17:38:51.999</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>44.0</td>
      <td>861.0</td>
      <td>144.0</td>
      <td>1.0</td>
      <td>cash</td>
      <td>2017-03-14 04:23:55.595</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>18.0</td>
      <td>935.0</td>
      <td>156.0</td>
      <td>1.0</td>
      <td>credit_card</td>
      <td>2017-03-26 12:43:36.649</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>18.0</td>
      <td>883.0</td>
      <td>156.0</td>
      <td>1.0</td>
      <td>credit_card</td>
      <td>2017-03-01 04:35:10.773</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5000 entries, 0 to 4999
    Data columns (total 7 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   order_id        5000 non-null   float64       
     1   shop_id         5000 non-null   float64       
     2   user_id         5000 non-null   float64       
     3   order_amount    5000 non-null   float64       
     4   total_items     5000 non-null   float64       
     5   payment_method  5000 non-null   object        
     6   created_at      5000 non-null   datetime64[ns]
    dtypes: datetime64[ns](1), float64(5), object(1)
    memory usage: 273.6+ KB
    

#### DataTypes

* Convert the ids to int datatype

* 'created_at' is imported correctly as datetime. Extract day and hour from 'created_at' datetime column.


```python
# using the .dt accessor to split the date column

prep_data['order_id'] = prep_data['order_id'].astype(int)

prep_data['shop_id'] = prep_data['shop_id'].astype(int)

prep_data['user_id'] = prep_data['user_id'].astype(int)

prep_data['day'] = prep_data['created_at'].dt.day

prep_data['weekday_n'] = prep_data['created_at'].dt.weekday

daydict = {
    0:"Monday",
    1:"Tuesday",
    2:"Wednesday",
    3:"Thursday",
    4:"Friday",
    5:"Saturday",
    6:"Sunday"
}

prep_data['weekday'] = prep_data['weekday_n'].map(daydict)

prep_data['hour'] = prep_data['created_at'].dt.hour
```

#### Calculate Price of Sneaker

Since each store sells only one type of sneaker, we can calculate the price of that sneaker by divinding the order amount by the number of items in the cart


```python
prep_data.insert(loc = 5, column = 'unit_amount', value = prep_data['order_amount'] / prep_data['total_items'])
```

#### Cleaned Data

This wraps up the basic feature engineering based on the information already provided to us. We will move on to exploring this data to understand trends and define performance metrics to quantify business performance.


```python
prep_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>shop_id</th>
      <th>user_id</th>
      <th>order_amount</th>
      <th>total_items</th>
      <th>unit_amount</th>
      <th>payment_method</th>
      <th>created_at</th>
      <th>day</th>
      <th>weekday_n</th>
      <th>weekday</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>53</td>
      <td>746</td>
      <td>224.0</td>
      <td>2.0</td>
      <td>112.0</td>
      <td>cash</td>
      <td>2017-03-13 12:36:56.190</td>
      <td>13</td>
      <td>0</td>
      <td>Monday</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>92</td>
      <td>925</td>
      <td>90.0</td>
      <td>1.0</td>
      <td>90.0</td>
      <td>cash</td>
      <td>2017-03-03 17:38:51.999</td>
      <td>3</td>
      <td>4</td>
      <td>Friday</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>44</td>
      <td>861</td>
      <td>144.0</td>
      <td>1.0</td>
      <td>144.0</td>
      <td>cash</td>
      <td>2017-03-14 04:23:55.595</td>
      <td>14</td>
      <td>1</td>
      <td>Tuesday</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>18</td>
      <td>935</td>
      <td>156.0</td>
      <td>1.0</td>
      <td>156.0</td>
      <td>credit_card</td>
      <td>2017-03-26 12:43:36.649</td>
      <td>26</td>
      <td>6</td>
      <td>Sunday</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>18</td>
      <td>883</td>
      <td>156.0</td>
      <td>1.0</td>
      <td>156.0</td>
      <td>credit_card</td>
      <td>2017-03-01 04:35:10.773</td>
      <td>1</td>
      <td>2</td>
      <td>Wednesday</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### 3) Data Exploration

#### Overview

We start our exploration by understanding how many unique stores, customers and payment methods we see in our dataset


```python
exp_data = prep_data.copy()

exp_data.nunique()
```




    order_id          5000
    shop_id            100
    user_id            301
    order_amount       258
    total_items          8
    unit_amount         58
    payment_method       3
    created_at        4995
    day                 30
    weekday_n            7
    weekday              7
    hour                24
    dtype: int64



There are 301 customers who contribute to 5000 transactions across 100 shops during the month of March 2017. 

Next we can try and visualize each order in the form of a scatter plot

#### Identifying Outliers


```python
fig1 = px.scatter(exp_data, x = 'created_at', y = 'order_amount', color = 'shop_id')
fig1.show('png')
```


    
![png](imgs/output_20_0.png)
    


There are several high-value outliers in the dataset. All observed high-value orders of $704,000 have taken place in shop 42.

Shop 78 too has recorded a number of high-value orders throughout the month of March

Let us further investigate these two stores as they will have a considerable impact on how performance metrics are to be defined

#### Exploring High-Value Shops 


```python
store42 = exp_data.query('shop_id == 42')

store78 = exp_data.query('shop_id == 78')

fig42 = px.histogram(store42, x = 'total_items', nbins=100, title='Shop 42 - # of items in each order')

fig42.show('png')

fig78 = px.histogram(store78, x = 'total_items', nbins=100,  title='Shop 78 - # of items in each order')

fig78.show('png')
```


    
![png](imgs/output_23_0.png)
    



    
![png](imgs/output_23_1.png)
    



```python
store42.append(store78).groupby('shop_id', as_index = False).mean()[['shop_id', 'unit_amount']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shop_id</th>
      <th>unit_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>352.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78</td>
      <td>25725.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Exploring Regular Shops

We can analyze all other stores to get a sense of the typical trends


```python
norm_data = exp_data.query("shop_id != 42 and shop_id != 78")

fig2 = px.scatter(norm_data, x = 'created_at', y ='order_amount')
fig2.show('png')

fig3 = px.box(norm_data, y = 'order_amount', points='all')
fig3.show('png')
```


    
![png](imgs/output_27_0.png)
    



    
![png](imgs/output_27_1.png)
    



```python
fignorm = px.histogram(norm_data, x = 'total_items', nbins=100,  title='Regular Stores - # of items in each order')
fignorm.show('png')
```


    
![png](imgs/output_28_0.png)
    



```python
norm_data.groupby('shop_id', as_index = False).mean()[['shop_id', 'unit_amount']].describe()[['unit_amount']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unit_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>150.22449</td>
    </tr>
    <tr>
      <th>std</th>
      <td>23.91675</td>
    </tr>
    <tr>
      <th>min</th>
      <td>90.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>132.25000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>153.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>165.75000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>201.00000</td>
    </tr>
  </tbody>
</table>
</div>



#### Summary of Exploratory Findings

* The data contains outlier transactions from stores 42 and 78
* Store 42: 
    * Sneakers priced at $352
    * Over 2000 units of sneakers sold in a single order in some cases
* Store 78:
    * Sneakers priced at $25,725
    * upto 6 sneakers sold in a single order
* Regular Stores:
    * Sneakers priced between $90 and $201
    * 91% of orders sold at most 3 sneakers

### 4) Defining Business Performance Metrics or KPIs

#### Pitfalls of using Average Order Value (AOV)

The key finding from the exporatory analysis was that the data contained outlier transactions from stores 42 and 78. These outliers could potentially over-estimate our Average Order Values (AOV) and skew our analysis.

Let's have a look at how the naive AOV analysis looks like when we don't remove the outliers


```python
full_aov = exp_data['order_amount'].mean()

print(f'The Naive Average Order Value (AOV) is ${round(full_aov,3)}')

fig4 = px.scatter(exp_data, x = 'created_at', y ='order_amount', color= 'shop_id')
fig4.add_hline(y=full_aov, line_width=2, line_dash="dash", line_color="black", annotation_text = f'Average Order Value = ${round(full_aov,3)}', annotation=dict(font_size=20))
fig4.show('png')
```

    The Naive Average Order Value (AOV) is $3145.128
    


    
![png](imgs/output_34_1.png)
    


#### Median Order Value (MOV)

The Average Order Value is $3145.128!! It's obvious that the AOV is over-exaggerated because of stores 42 and 78. 

An approach to get better estimates of central tendancy is by using the Median or Mode. These estimates do not get influenced by extreme outliers.

In the (extremely compressed) boxplot below we can see that the Median Order Value is $284.

<i>Hover over the visual to view quartiles</i>


```python
full_mov = exp_data['order_amount'].median()

print(f'The Median Order Value (MOV) is ${round(full_mov,3)}')

fig5 = px.box(exp_data, y ='order_amount', points= 'all')
fig5.add_hline(y=full_mov, line_dash="dash", annotation_text= f'Median Order Value (MOV) = ${full_mov}', annotation_position="top right", annotation=dict(font_size=20))
fig5.show('png')
```

    The Median Order Value (MOV) is $284.0
    


    
![png](imgs/output_37_1.png)
    


#### Piece-wise (AOV/MOV) Analysis

Alternatively, we can analyze the regular shops and the high order value shops separately to avoid distortion and make better business decisions.

##### Regular Stores


```python
norm_aov = norm_data['order_amount'].mean()
norm_mov = norm_data['order_amount'].median()

print(f'The Average Order Value (AOV) for regular stores is ${round(norm_aov,3)}')
print(f'The Median Order Value (MOV) for regular stores is ${round(norm_mov,3)}')

fig6 = px.scatter(norm_data, x = 'created_at', y ='order_amount')
fig6.add_hline(y=norm_aov, line_width=2, line_dash="dash", line_color="black", annotation_text = f'Average Order Value = ${round(norm_aov,3)}', annotation_position="top right", annotation=dict(font_size=20))
fig6.add_hline(y=norm_mov, line_width=2, line_dash="dash", line_color="black", annotation_text = f'Median Order Value = ${round(norm_mov,3)}', annotation_position="bottom right", annotation=dict(font_size=20))
fig6.show('png')
```

    The Average Order Value (AOV) for regular stores is $300.156
    The Median Order Value (MOV) for regular stores is $284.0
    


    
![png](imgs/output_40_1.png)
    


##### Store 42


```python
aov_42 = store42['order_amount'].mean()
mov_42 = store42['order_amount'].median()

print(f'The Average Order Value (AOV) for store 42 is ${round(aov_42,3)}')
print(f'The Median Order Value (MOV) for store 42 is ${round(mov_42,3)}')

fig7 = px.scatter(store42, x = 'created_at', y ='order_amount')
fig7.add_hline(y=aov_42, line_width=2, line_dash="dash", line_color="black", annotation_text = f'Average Order Value = ${round(aov_42,3)}', annotation_position='top right', annotation=dict(font_size=20))
fig7.add_hline(y=mov_42, line_width=2, line_dash="dash", line_color="black", annotation_text = f'Median Order Value = ${round(mov_42,3)}', annotation_position='bottom right', annotation=dict(font_size=20))
fig7.show('png')
```

    The Average Order Value (AOV) for store 42 is $235101.49
    The Median Order Value (MOV) for store 42 is $704.0
    


    
![png](imgs/output_42_1.png)
    


##### Store 78


```python
aov_78 = store78['order_amount'].mean()
mov_78 = store78['order_amount'].median()

print(f'The Average Order Value (AOV) for store 78 is ${round(aov_78,3)}')
print(f'The Median Order Value (MOV) for store 78 is ${round(mov_78,3)}')

fig8 = px.scatter(store78, x = 'created_at', y ='order_amount')
fig8.add_hline(y=aov_78, line_width=2, line_dash="dash", line_color="black", annotation_text = f'Average Order Value = ${round(aov_78,3)}', annotation_position='bottom right', annotation=dict(font_size=20))
fig8.add_hline(y=mov_78, line_width=2, line_dash="dash", line_color="black", annotation_text = f'Median Order Value = ${round(mov_78,3)}', annotation_position='top right', annotation=dict(font_size=20))
fig8.show('png') 
```

    The Average Order Value (AOV) for store 78 is $49213.043
    The Median Order Value (MOV) for store 78 is $51450.0
    


    
![png](imgs/output_44_1.png)
    


### Additional Insights

Analyzing Median Order Value (MOV) for each day of the week

Which days of the week record the most shopping activity?


```python
dayavg = exp_data.groupby('weekday', as_index= False).median()['order_amount'].mean()

fig9 = px.line(exp_data.groupby(['weekday_n', 'weekday'], as_index= False).median()[['weekday', 'order_amount']], x = 'weekday', y = 'order_amount', labels={'order_amount':'Median Order Value (MOV)'})
fig9.add_hline(y=dayavg, line_width=3, line_dash="dash", line_color="black", annotation_text = f'Average MOV = {round(dayavg,3)}')
fig9.show('png')
```


    
![png](imgs/output_47_0.png)
    


What are the peak hours of shopping during the day?


```python
hravg = exp_data.groupby('hour', as_index= False).median()['order_amount'].mean()

fig10 = px.line(exp_data.groupby('hour', as_index= False).median()[['hour', 'order_amount']], x = 'hour', y ='order_amount', labels={'order_amount':'Median Order Value (MOV)'})
fig10.add_hline(y=hravg, line_width=3, line_dash="dash", line_color="black", annotation_text = f'Average MOV = {round(hravg,3)}')
fig10.show('png')
```


    
![png](imgs/output_49_0.png)
    


#### Summary of Additional Insights

* Peak shopping days are Thursday, Saturday and Sunday
* Peak shopping hours are (4am - 9am), 2pm and 11pm

Promotional campaigns and deals could be designed based on peak shopping days and hours.
