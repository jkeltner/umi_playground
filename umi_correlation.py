# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Upstart Macro Index (UMI) Playground
# An open source tool to take a look at the latest UMI values, correaltions to a few broadly available macroeconomic variables, and a quick correlation model built from those variables.

# %% [markdown]
# ## UMI Import and Review
# Grab and format the data in the CSV file and give a few quick view of the data.

# %%
from datetime import datetime
from dotenv import load_dotenv
from fredapi import Fred
from IPython.display import display, clear_output
import ipywidgets as widgets
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

UPSTART_TEAL = '#00b1ac'

plt.rcParams['figure.figsize'] = [8.0, 6.0]

# function to add gray recession bars to a seaborn plot
def addRecessions(graph):
    recession_periods = [('2020-02', '2020-04'),]
    for period in recession_periods:
        start, end = period
        graph.axvspan(start, end,facecolor='gray', alpha=0.2)


# %%
# Grab our UMI data, format, and describe
umi_data = pd.read_csv('data/umi.csv')
def convert_date(date_str):
    date = datetime.strptime(date_str, '%b \'%y')
    return date.strftime('%Y-%m')
umi_data['date'] = umi_data['date'].apply(convert_date) # convert dates into a more format that will sort properly
umi_data.describe()


# %%
# for those who prefer a visual look...
sns.boxplot(y='umi', data=umi_data, color=UPSTART_TEAL)

# %%
# Quick graph of UMI over time
graph = sns.lineplot(data=umi_data, x='date', y='umi', color=UPSTART_TEAL) # draw the UMI line
x_ticks = umi_data['date'][::3] # include every 3rd month == quarters
graph.set(xticks=x_ticks)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90) # set the labels and rotate 90 degrees
graph.axhline(1.0, color="gray") # add a grey line at UMI of 1.0
addRecessions(graph)


# %% [markdown]
# ## Public Data Downloads
# Now we're going to download some data on the macro environment during these same time periods and see how they may or may not relate to UMI.

# %%
# make a copy of the UMI data and start pulling in all of our fun data sources!
macro_data = umi_data.copy()
macro_data.set_index('date', inplace=True)

# Pull in a bunch of data from the FRED API to build out a UMI + macro data set
# To see what each series is check out https://fred.stlouisfed.org/ or 
#   use fred.get_series_info() -- e.g. fred.get_series_info('PSAVERT') or fred.get_series_info('UNRATE')
# You can also use fred.search() to discover new series -- e.g. fred.search('unemployment') or fred.search('savings')

fred = Fred(api_key=FRED_API_KEY)
fred_series = ['PSAVERT', 'CORESTICKM159SFRBATL', 'MEDCPIM158SFRBCLE', 'UNRATE', 'FEDFUNDS', 'T10Y2YM']
for series in fred_series:
    data = fred.get_series(series)
    data.index = data.index.strftime('%Y-%m')
    data.name = series
    data = pd.DataFrame(data)
    # data[series+"_pct_change"] = data.pct_change() # add M|M change of each variable to our data set as well
    macro_data = macro_data.join(data)

# let's check out the basic correlation of UMI with other macro data 
correlation = macro_data.corr()
# goign to sort these by largest absolute value of correlation
correlation['umi_abs'] = correlation['umi'].abs()
correlation.sort_values(by='umi_abs', ascending=False, inplace=True)
correlation.drop('umi_abs', axis=1, inplace=True)
# it's also to notice how correlated some of these macro data points are with each other
display(correlation)

# %% [markdown]
# ## Correlation Model
# Now that we have the basic macro variables and have looked at the correlation factors, let's see if we can build a correlation model that predicts UMI based on these variables.

# %%
# let's try to predict UMI using a linear regression model and our macro data

# fill in NaN values with the previous value
train_data = macro_data.fillna(method='ffill')

# let's initialize our model
corr_model = LinearRegression()

X = train_data.drop('umi', axis=1)
y = train_data['umi']

# Fit the model to the entire data set
corr_model.fit(X, y)

# Use the trained model to make predictions on the same data set
model_predictions = corr_model.predict(X)

# %%
prediction_data = umi_data.copy()
prediction_data['umi_preds'] = model_predictions
graph = sns.lineplot(data=prediction_data, x='date', y='umi', color=UPSTART_TEAL, label="UMI") # draw the UMI line
sns.lineplot(data=prediction_data, x='date', y='umi_preds', color="orange", linestyle='--', label="Correlation Model") # draw the UMI predictions line
graph.axhline(1.0, color="gray") # add a grey line at UMI of 1.0
x_ticks = umi_data['date'][::3] # include every 3rd month == quarters
graph.set(xticks=x_ticks)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90) # set the labels and rotate 90 degrees
graph.legend()
addRecessions(graph)


# %% [markdown]
# ## UMI Predictions
# We can also use the UMI correlation model we built above to make predictions about what the UMI value might be in the future under circumstancs defined by values of those macro variables. *It is important to remember that this model is a trained with data from a limited number of macroeconomic environments - and it may not hold up as well in future macroeconomic environments. It should also be noted that many of these variabels are highly correlated, so while the model may make predictions on any value for each variable - some of those combinations may not make sense in the real world.*

# %%
macro_predictions = macro_data.drop('umi', axis=1).fillna(method='ffill').tail(1)
macro_variables = macro_predictions.columns

output = widgets.Output()

def update_variable(change, variable):
    with output:
        macro_predictions[variable][0] = change.new
        umi_predictions = corr_model.predict(macro_predictions)
        result = umi_predictions[0]
        clear_output(wait=True)
        display(f'UMI Prediction: {result:.2f}')

for variable in macro_variables:
    w = widgets.FloatText(value=macro_predictions[variable][0], description=variable)
    w.observe(lambda change, variable=variable: update_variable(change, variable=variable), names='value')
    display(w)

display(output)


# %%

# %%
