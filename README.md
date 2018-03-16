## Analyze A/B Test Results

## Table of Contents
- [Introduction](#intro)
- [Part I - Probability](#probability)
- [Part II - A/B Test](#ab_test)
- [Part III - Regression](#regression)


<a id='intro'></a>
### Introduction

In this project, I work through understanding the results of an A/B test run by an e-commerce website.  My goal is to help the company understand if they should implement the new page, keep the old page, or run the experiment longer before making their decision.

<a id='probability'></a>
#### Part I - Probability


```python
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.axes as ax
%matplotlib inline
```


```python
df = pd.read_csv('ab_data.csv')
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>851104</td>
      <td>2017-01-21 22:11:48.556739</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>804228</td>
      <td>2017-01-12 08:01:45.159739</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>661590</td>
      <td>2017-01-11 16:55:06.154213</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>853541</td>
      <td>2017-01-08 18:28:03.143765</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>864975</td>
      <td>2017-01-21 01:52:26.210827</td>
      <td>control</td>
      <td>old_page</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    294478




```python
len(pd.unique(df['user_id']))
```




    290584




```python
conv = df.groupby(by='user_id')['converted'].max()
conv.sum()/len(conv)
```




    0.12104245244060237



The number of times the `new_page` and `treatment` don't line up--



```python
len(df.query('group == "treatment"').query('landing_page != "new_page"')) + len(df.query('group != "treatment"').query('landing_page == "new_page"'))
```




    3893




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 294478 entries, 0 to 294477
    Data columns (total 5 columns):
    user_id         294478 non-null int64
    timestamp       294478 non-null object
    group           294478 non-null object
    landing_page    294478 non-null object
    converted       294478 non-null int64
    dtypes: int64(2), object(3)
    memory usage: 11.2+ MB



```python
df2 = df.query('group == "treatment"').query('landing_page == "new_page"')
df2b = df.query('group == "control"').query('landing_page != "new_page"')
df2 = df2.append(df2b)
df2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>661590</td>
      <td>2017-01-11 16:55:06.154213</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>853541</td>
      <td>2017-01-08 18:28:03.143765</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>679687</td>
      <td>2017-01-19 03:26:46.940749</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>817355</td>
      <td>2017-01-04 17:58:08.979471</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>839785</td>
      <td>2017-01-15 18:11:06.610965</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
```




    0




```python
len(pd.unique(df2['user_id'])), len(df2)
```




    (290584, 290585)




```python
df2.groupby(by='user_id').size().reset_index(name='counts').sort_values(by=['counts']) #945971
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>630000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193713</th>
      <td>840694</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193714</th>
      <td>840695</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193715</th>
      <td>840696</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193716</th>
      <td>840697</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193717</th>
      <td>840698</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193718</th>
      <td>840699</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193719</th>
      <td>840700</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193720</th>
      <td>840701</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193721</th>
      <td>840702</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193722</th>
      <td>840703</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193723</th>
      <td>840704</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193724</th>
      <td>840705</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193725</th>
      <td>840706</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193726</th>
      <td>840707</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193727</th>
      <td>840708</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193728</th>
      <td>840709</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193729</th>
      <td>840710</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193743</th>
      <td>840724</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193742</th>
      <td>840723</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193741</th>
      <td>840722</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193740</th>
      <td>840721</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193739</th>
      <td>840720</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193738</th>
      <td>840719</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193712</th>
      <td>840693</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193737</th>
      <td>840718</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193735</th>
      <td>840716</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193734</th>
      <td>840715</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193733</th>
      <td>840714</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193732</th>
      <td>840713</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96848</th>
      <td>735285</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96847</th>
      <td>735284</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96853</th>
      <td>735292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96846</th>
      <td>735283</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96844</th>
      <td>735281</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96843</th>
      <td>735280</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96842</th>
      <td>735279</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96841</th>
      <td>735278</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96840</th>
      <td>735277</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96839</th>
      <td>735275</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96845</th>
      <td>735282</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96871</th>
      <td>735311</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96854</th>
      <td>735293</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96856</th>
      <td>735295</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96869</th>
      <td>735309</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96868</th>
      <td>735308</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96867</th>
      <td>735306</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96866</th>
      <td>735305</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96865</th>
      <td>735304</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96855</th>
      <td>735294</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96864</th>
      <td>735303</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96862</th>
      <td>735301</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96861</th>
      <td>735300</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96860</th>
      <td>735299</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96859</th>
      <td>735298</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96858</th>
      <td>735297</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96857</th>
      <td>735296</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96863</th>
      <td>735302</td>
      <td>1</td>
    </tr>
    <tr>
      <th>290583</th>
      <td>945999</td>
      <td>1</td>
    </tr>
    <tr>
      <th>131712</th>
      <td>773192</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>290584 rows × 2 columns</p>
</div>




```python
df2.query('user_id == "773192"')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1899</th>
      <td>773192</td>
      <td>2017-01-09 05:37:58.781806</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2893</th>
      <td>773192</td>
      <td>2017-01-14 02:55:59.590927</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = df2.drop(1899)

```


```python
df2['converted'].sum()/len(df2)
```




    0.11959708724499628



Given that an individual was in the `control` group, what is the probability they converted?


```python
cont = df2.query('group=="control"')['converted'].sum()/len(df2.query('group=="treatment"'))
cont
```




    0.12035647925125594



Given that an individual was in the `treatment` group, what is the probability they converted?


```python
treat = df2.query('group=="treatment"')['converted'].sum()/len(df2.query('group=="treatment"'))
treat 
```




    0.11880806551510564



What is the probability that an individual received the new page?


```python
len(df2.query('landing_page=="new_page"'))/len(df2)
```




    0.5000619442226688



The treatment group converted below average and below the rate the control group converted. The treatment has no effect with practical significance. I would not suggest switching to the new page.

<a id='ab_test'></a>
### Part II - A/B Test

Hypothesis:

**$H_{0}$**: **$p_{new}$** <= **$p_{old}$**

 **$H_{1}$**: **$p_{new}$** > **$p_{old}$**


```python
df2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>661590</td>
      <td>2017-01-11 16:55:06.154213</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>853541</td>
      <td>2017-01-08 18:28:03.143765</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>679687</td>
      <td>2017-01-19 03:26:46.940749</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>817355</td>
      <td>2017-01-04 17:58:08.979471</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>839785</td>
      <td>2017-01-15 18:11:06.610965</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



What is the **convert rate** for $p_{new}$ under the null? 


```python
p_new = df2['converted'].sum()/len(df2)
p_new
```




    0.11959708724499628



What is the **convert rate** for $p_{old}$ under the null? <br><br>


```python
p_old = df2['converted'].sum()/len(df2)
p_old
```




    0.11959708724499628



What is $n_{new}$?


```python
n_new = len(df2.query('landing_page=="new_page"'))
n_new
```




    145310



What is $n_{old}$?


```python
n_old = len(df2.query('landing_page=="old_page"'))
n_old
```




    145274



Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.


```python
new_page_converted = np.random.choice(2, size = 145311, p=[0.8805, 0.1195])
```

Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.


```python
old_page_converted = np.random.choice(2, size = 145274, p=[0.8805, 0.1195])
```

Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).


```python
(new_page_converted.sum()/len(new_page_converted)) - (old_page_converted.sum()/len(old_page_converted))
```




    -0.00056748933929881562



Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in a numpy array called **p_diffs**.


```python
p_diffs = np.random.binomial(n_new, p_new, 10000)/n_new - np.random.binomial(n_old, p_old, 10000)/n_old
p_diffs[:5]
```




    array([-0.0028721 , -0.00029128, -0.00303713,  0.00081711,  0.00239311])




```python
plt.hist(p_diffs)
plt.axvline(-0.00127,color='red')
```




    <matplotlib.lines.Line2D at 0x1184fa860>




![png](output_42_1.png)


What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?


```python
actual_diff = treat-cont
pd_df = pd.DataFrame(p_diffs)
pd_df.columns = ['a']
len(pd_df.query('a > @actual_diff'))/len(pd_df)
```




    0.8978



I calculated the critical value- the threshold for the practical significance in the differences between the new and old pages. Eighty-five percent of the differences were greater than the line.


```python
import statsmodels.api as sm

convert_old = df2.query("landing_page == 'old_page' and converted == 1").shape[0]
convert_new = df2.query("landing_page == 'new_page' and converted == 1").shape[0]
```

    /anaconda3/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools



```python
z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old], alternative='larger')
print(z_score, p_value)
```

    -1.31092419842 0.905058312759


Since the z-score of `-0.0247046451343` is less than the critical value of `1.959963984540054` and the p-value is so high at `0.51`, we can fail to reject the null hypotesis.

<a id='regression'></a>
### Part III - A regression approach

Logistic Regression

Logistic Regression

The goal is to use **statsmodels** to fit the regression model in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.


```python
df2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>661590</td>
      <td>2017-01-11 16:55:06.154213</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>853541</td>
      <td>2017-01-08 18:28:03.143765</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>679687</td>
      <td>2017-01-19 03:26:46.940749</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>817355</td>
      <td>2017-01-04 17:58:08.979471</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>839785</td>
      <td>2017-01-15 18:11:06.610965</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2['intercept']=1
df2[['ab_page','old_page']]= pd.get_dummies(df2['landing_page'])
df2 = df2.drop('old_page', axis = 1)
df2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
      <th>intercept</th>
      <th>ab_page</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>661590</td>
      <td>2017-01-11 16:55:06.154213</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>853541</td>
      <td>2017-01-08 18:28:03.143765</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>679687</td>
      <td>2017-01-19 03:26:46.940749</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>817355</td>
      <td>2017-01-04 17:58:08.979471</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>839785</td>
      <td>2017-01-15 18:11:06.610965</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
logit = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = logit.fit()
```

    Optimization terminated successfully.
             Current function value: 0.366118
             Iterations 6



```python
results.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>converted</td>    <th>  No. Observations:  </th>   <td>290584</td>   
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>290582</td>   
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>     1</td>   
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 16 Mar 2018</td> <th>  Pseudo R-squ.:     </th>  <td>8.077e-06</td> 
</tr>
<tr>
  <th>Time:</th>              <td>13:40:30</td>     <th>  Log-Likelihood:    </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th>   <td>0.1899</td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -1.9888</td> <td>    0.008</td> <td> -246.669</td> <td> 0.000</td> <td>   -2.005</td> <td>   -1.973</td>
</tr>
<tr>
  <th>ab_page</th>   <td>   -0.0150</td> <td>    0.011</td> <td>   -1.311</td> <td> 0.190</td> <td>   -0.037</td> <td>    0.007</td>
</tr>
</table>



The p-value associated with **ab_page** is 0.190. Because it is greater than 0.05 we fail to reject the null hypothesis, which in this case is that the new landing page is less effective or equal to the old one. <br><br>In Part II, the test identified whether the average conversion rates differ between page A and page B visitors in the population. A logistic regression estimates how the conversion rate varies by page visited. In other words, we're comparing the differences between two samples as opposed to the relationship between a dependent and independent variable. Moreover, the simulation and the z-test were one-sided tests, whereas the regression was not. <br><Br>

There are many factors that might influence whether or not someone converts besides which landing page they hit. For example, whether or not they are in the target market, which may be identified by age, gender, or other demographic information, might directly influence whether or not someone buys. Of course, when adding additional terms into the regressional model it's important to consider that they are not correlated; for exmaple, we wouldn't want to add both interest in softball and gender because those are correlated.

Does it appear that country had an impact on conversion? 


```python
countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
      <th>intercept</th>
      <th>ab_page</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>834778</th>
      <td>UK</td>
      <td>2017-01-14 23:08:43.304998</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>928468</th>
      <td>US</td>
      <td>2017-01-23 14:44:16.387854</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>822059</th>
      <td>UK</td>
      <td>2017-01-16 14:04:14.719771</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>711597</th>
      <td>UK</td>
      <td>2017-01-22 03:14:24.763511</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>710616</th>
      <td>UK</td>
      <td>2017-01-16 13:14:44.000513</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_new[['CA', 'UK', 'US']] = pd.get_dummies(df_new['country'])
df_new = df_new.drop('US', axis = 1)
df_new.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
      <th>intercept</th>
      <th>ab_page</th>
      <th>CA</th>
      <th>UK</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>834778</th>
      <td>UK</td>
      <td>2017-01-14 23:08:43.304998</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>928468</th>
      <td>US</td>
      <td>2017-01-23 14:44:16.387854</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>822059</th>
      <td>UK</td>
      <td>2017-01-16 14:04:14.719771</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>711597</th>
      <td>UK</td>
      <td>2017-01-22 03:14:24.763511</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>710616</th>
      <td>UK</td>
      <td>2017-01-16 13:14:44.000513</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
logit = sm.Logit(df_new['converted'], df_new[['intercept', 'CA', 'UK']])
results = logit.fit()
results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.366116
             Iterations 6





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>converted</td>    <th>  No. Observations:  </th>   <td>290584</td>   
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>290581</td>   
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>     2</td>   
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 16 Mar 2018</td> <th>  Pseudo R-squ.:     </th>  <td>1.521e-05</td> 
</tr>
<tr>
  <th>Time:</th>              <td>13:40:32</td>     <th>  Log-Likelihood:    </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th>   <td>0.1984</td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -1.9967</td> <td>    0.007</td> <td> -292.314</td> <td> 0.000</td> <td>   -2.010</td> <td>   -1.983</td>
</tr>
<tr>
  <th>CA</th>        <td>   -0.0408</td> <td>    0.027</td> <td>   -1.518</td> <td> 0.129</td> <td>   -0.093</td> <td>    0.012</td>
</tr>
<tr>
  <th>UK</th>        <td>    0.0099</td> <td>    0.013</td> <td>    0.746</td> <td> 0.456</td> <td>   -0.016</td> <td>    0.036</td>
</tr>
</table>



Country did not have significant effect on conversion rate.

But could page *and* country?


```python
logit = sm.Logit(df_new['converted'], df_new[['intercept', 'CA', 'UK', 'ab_page']])
results = logit.fit()
results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.366113
             Iterations 6





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>converted</td>    <th>  No. Observations:  </th>   <td>290584</td>   
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>290580</td>   
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>     3</td>   
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 16 Mar 2018</td> <th>  Pseudo R-squ.:     </th>  <td>2.323e-05</td> 
</tr>
<tr>
  <th>Time:</th>              <td>13:40:33</td>     <th>  Log-Likelihood:    </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th>   <td>0.1760</td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -1.9893</td> <td>    0.009</td> <td> -223.763</td> <td> 0.000</td> <td>   -2.007</td> <td>   -1.972</td>
</tr>
<tr>
  <th>CA</th>        <td>   -0.0408</td> <td>    0.027</td> <td>   -1.516</td> <td> 0.130</td> <td>   -0.093</td> <td>    0.012</td>
</tr>
<tr>
  <th>UK</th>        <td>    0.0099</td> <td>    0.013</td> <td>    0.743</td> <td> 0.457</td> <td>   -0.016</td> <td>    0.036</td>
</tr>
<tr>
  <th>ab_page</th>   <td>   -0.0149</td> <td>    0.011</td> <td>   -1.307</td> <td> 0.191</td> <td>   -0.037</td> <td>    0.007</td>
</tr>
</table>



All of the p-values related to country or page are wll past the .05 threshold, or even the .1 threshold if we were being generous. I would say none of these factors are particularly good predictors of conversion. 

