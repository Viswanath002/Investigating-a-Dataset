# Investigating-a-Dataset
Introduction This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not the patients show up for the appointment based on characteristic of patient included in each row.  'Scheduled Day' tells us on which day patient set their appointment  'Neighborhood' tells us about the location of the hospital 'Scholarship' tells us whether the patient is enrolled in the welfare program
Questions
What features are important to know inorder to predict of a patient will show up for appointment or not.
Does the dataset contain any outliers or missing values to address.
Can we perform Dimensionality reduction based on correlation between features
Does any feature requires tranformation considering any outliers in the dataset
Data Wrangling

# importing the necessary libraries for data analysis
import pandas as pd
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats
​
​

# Load your dataset to Jupite Notebooks and performing initial level analysis
​
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.shape               
(110527, 14)
After loading the dataset , and analyzing the shape it was found that data has 14 columns and 110527 entries.

# Code to look for any missing values present in the dataset
​
df.isnull().sum()


PatientId         0
AppointmentID     0
Gender            0
ScheduledDay      0
AppointmentDay    0
Age               0
Neighbourhood     0
Scholarship       0
Hipertension      0
Diabetes          0
Alcoholism        0
Handcap           0
SMS_received      0
No-show           0
dtype: int64
The dataset was found to have no missing values

df.duplicated().sum() 
0
The dataset was found to have no duplicates entries present

df.dtypes # The features ScheduledDay and AppointmentDay needs to be changes to datetimeformat
PatientId         float64
AppointmentID       int64
Gender             object
ScheduledDay       object
AppointmentDay     object
Age                 int64
Neighbourhood      object
Scholarship         int64
Hipertension        int64
Diabetes            int64
Alcoholism          int64
Handcap             int64
SMS_received        int64
No-show            object
dtype: object
Following observation of data types from the dataset
Patient ID is of type - float
Appointment ID ,Age ,Scholarship,Hipertension, Diabetes,Alcoholism,Handcap,Sms_received is of type 'int'
Gender, Scheduledday,Appointment Day,Neighbourhood ,No -show is of type - 'string'

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
Converting Features to datetime format
ScheduledDay and Appointment Day features were found to be in wrong format but are now converted to datetime format

df.head(1) # The revised features with datetime format has been reflected in the below dataframe
PatientId	AppointmentID	Gender	ScheduledDay	AppointmentDay	Age	Neighbourhood	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received	No-show
0	2.987250e+13	5642903	F	2016-04-29 18:38:08	2016-04-29	62	JARDIM DA PENHA	0	1	0	0	0	0	No
Dealing with Outliers

df.hist(figsize =(15,15));
​

Histogram Observations
From the above graph we can interpret that data is not normally distibuted for various features
Appointment ID seems to reflect Left skewered distribution
Age feature seems to be show right skewered distribution
Observations from Outliers
'Age' feature seems to be skewered to the right because of outliers and needs to be normalised

np.sqrt(df['Age']).plot(kind = 'hist' , title = 'Histogram Distribution for Age' ,figsize = (8,5));
plt.xlabel('Distribution')
/opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:1: RuntimeWarning: invalid value encountered in sqrt
  """Entry point for launching an IPython kernel.
Text(0.5,0,'Distribution')

Observation from above plot
The graph above reflects the Age feature now being tranformed to a normal distribution and all the skewness has now been removed

df['Age'].plot(kind = 'box' , title = 'Outlier Assesment')
plt.ylabel('years')
Text(0,0.5,'years')

Age feature outlier solution
The Age feature's outlier has been tranformed using sqrt function and the data now seems to be in normal distribution
The figure depicts 3 different types of quartile 75,50,25 the value which is present above 75 and below 25 seems to represent the outliers
The value present at the 50th quartile represents the mean

Exploratory Data Analysis

df.corr()
PatientId	AppointmentID	Age	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received
PatientId	1.000000	0.004039	-0.004139	-0.002880	-0.006441	0.001605	0.011011	-0.007916	-0.009749
AppointmentID	0.004039	1.000000	-0.019126	0.022615	0.012752	0.022628	0.032944	0.014106	-0.256618
Age	-0.004139	-0.019126	1.000000	-0.092457	0.504586	0.292391	0.095811	0.078033	0.012643
Scholarship	-0.002880	0.022615	-0.092457	1.000000	-0.019729	-0.024894	0.035022	-0.008586	0.001194
Hipertension	-0.006441	0.012752	0.504586	-0.019729	1.000000	0.433086	0.087971	0.080083	-0.006267
Diabetes	0.001605	0.022628	0.292391	-0.024894	0.433086	1.000000	0.018474	0.057530	-0.014550
Alcoholism	0.011011	0.032944	0.095811	0.035022	0.087971	0.018474	1.000000	0.004648	-0.026147
Handcap	-0.007916	0.014106	0.078033	-0.008586	0.080083	0.057530	0.004648	1.000000	-0.024161
SMS_received	-0.009749	-0.256618	0.012643	0.001194	-0.006267	-0.014550	-0.026147	-0.024161	1.000000
Correlation - Matrix Analysis
Positive correlation observed between Age and Hipertension 0.50
Inverse correlation observed between SMS_received and Appointment_ID

df.plot.scatter('Age','Hipertension')  # Scatterplot  seems to show no correlation between age and Hipertension
<matplotlib.axes._subplots.AxesSubplot at 0x7f0828905fd0>

The scatterplot above reveals that there seems to be not much of a correlation between Age and Hipertension

Determining relationship between Age and No-show

df.groupby('No-show')['Age'].mean().plot(kind = 'bar' , title = 'No-show Analysis')
plt.ylabel('Age')
Text(0,0.5,'Age')

Members with age > 35 yrs where found to be present on the fixed appointmentdate

Chi square test to determine correlation between Gender and No-show
Two categorical variable are said to be in correlation if the p-value was found to be (lesser than 0.05)

table = pd.crosstab(df['Gender'],df['No-show']) # Chi-square test to analyze correlation between catrgorical variables

from scipy.stats import chi2_contingency

chi2,p,dof,expected = chi2_contingency(table.values)

print (chi2,p) # No Correlation between Gender and Target variable -- since p> 0.05
1.85343697924 0.173384181898
Chi sqaure test to determine correlation between Neighbourhood and No-show

table = pd.crosstab(df['Neighbourhood'],df['No-show']) # Chi-square test to analyze correlation between Neighbourhood and No-show

from scipy.stats import chi2_contingency

chi2,p,dof,expected = chi2_contingency(table.values)

print(chi2,p) # No correlation between Neighbourhood and Taget Variable (No-show)
491.927869497 1.54243592622e-60
Chi square test to determine relation between Hipertension and No-show
Type Markdown and LaTeX: 𝛼2

table = pd.crosstab(df['Hipertension'],df['No-show'])
​

from scipy.stats import chi2_contingency

chi2,p,dof,expected = chi2_contingency(table.values)

print(chi2,p) # P- values seems to be greater thatn 0.05 , no correlation observed between variables
140.651443005 1.91761092383e-32
Chi square test to determine the relation between Diabetes and No-show

table = pd.crosstab(df['Diabetes'],df['No-show'])

from scipy.stats import chi2_contingency

chi2,p,dof,expected = chi2_contingency(table.values)

print(chi2,p) # NO correlation between Diabetes and No-show 
25.3226094911 4.84990457523e-07
Chi square test to determine the relation between Scholarship and No-show

table = pd.crosstab(df['Scholarship'],df['No-show'])

from scipy.stats import chi2_contingency

chi2,p,dof,expected = chi2_contingency(table.values)

print(chi2,p) # NO Correlation between Alochol and No-show
93.5771972953 3.90662491385e-22
Conclusion
From the analysis it was found that Patient Age , Hipertension , Gender,Neighborhood would help us better to determine No-show appointment
The Dataset was found to have no missing values
'Age' was found to have outlier and right skewered but the same was normalized using transformation technique
Correlation for numerical and categorical features were analyzed and was found that we cannot remove any feature and require all to do further analysis

Limitations observed from Analysis
Correlation between numerical and categorical features were difficult to determine
Not able to check relationship between categorical variables diagramatically
Scatterplot cannot be used here to analyze since most of the features are categorical and only patient and appointment ID can be used for scatterplot since it is numerical

from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])
