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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.linear_model import LinearRegression

# use the official Upstart teal!
UPSTART_TEAL = '#00b1ac'

# setting up some plot size and dpi for nice graphics
plt.rcParams['figure.figsize'] = [8.0, 6.0]
plt.rcParams['figure.dpi'] = 300

# function to add gray recession bars to a seaborn plot
def addRecessions(graph):
    recession_periods = [('2020-02', '2020-04'),]
    for period in recession_periods:
        start, end = period
        graph.axvspan(start, end,facecolor='gray', alpha=0.2)


# %%
# Grab our UMI data, format, and describe
umi_data = pd.read_csv('data/umi_data.csv')
umi_data.describe()


# %%
# for those who prefer a visiual look...
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
# ## Simple Time Series Predictions
# Going to build a simple ARIMA model to do time series predictions and forecast the next 1 month UMI.

# %%
from statsmodels.tsa.arima.model import ARIMA

arima_data = umi_data['umi']
# define model
arima_model = ARIMA(arima_data, order=(1, 1, 1))
#train model
arima_model_fit = arima_model.fit()
# make predictions
model_predictions = arima_model_fit.predict(start=0, end=len(arima_data))

# %%
#let's graph our ARIMA model data
prediction_data = umi_data.copy()
prediction_data['arima'] = model_predictions
prediction_data['arima'].iloc[0] = np.nan # set the first value to NaN
graph = sns.lineplot(data=prediction_data, x='date', y='umi', color=UPSTART_TEAL, label="UMI") # draw the UMI line
sns.lineplot(data=prediction_data, x='date', y='arima', color="orange", linestyle='--', label="ARIMA Model") # draw the UMI predictions line
graph.axhline(1.0, color="gray") # add a grey line at UMI of 1.0
x_ticks = umi_data['date'][::3] # include every 3rd month == quarters
graph.set(xticks=x_ticks)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90) # set the labels and rotate 90 degrees
graph.legend()
addRecessions(graph)

# %%
# print out the final value as next month's prediction
print('Next month UMI ARIMA prediction: ', model_predictions[len(model_predictions)-1])

# %% [markdown]
# ## Macro Data Correlations
# Now we want to look at the correlation between UMI and some data about the state of the macroeconomic environment.

# %%
# make a copy of the UMI data and start pulling in all of our fun data sources!
# if you want to see the source of this data, check out the data_processing notebook
macro_data = pd.read_csv('data/macro_data.csv')

# let's check out the basic correlation of UMI with other macro data 
correlation = macro_data.corr(numeric_only=True)
# goign to sort these by largest absolute value of correlation
correlation['umi_abs'] = correlation['umi'].abs()
correlation.sort_values(by='umi_abs', ascending=False, inplace=True)
correlation.drop('umi_abs', axis=1, inplace=True)
sorted_index = correlation.index
correlation = correlation.reindex(columns=sorted_index)
# easier to view this as a heatmap
graph = sns.heatmap(
    correlation, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
graph.set_xticklabels(
    graph.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# %% [markdown]
# ## Correlation Model
# Now that we have the basic macro variables and have looked at the correlation factors, let's see if we can build a correlation model that predicts UMI based on these variables.

# %%
# fill in NaN values with the previous value
train_data = macro_data.fillna(method='ffill')

# let's initialize our model
corr_model = LinearRegression()

X = train_data.drop('umi', axis=1)
y = train_data['umi']

X.tail()
# Fit the model to the entire data set
corr_model.fit(X, y)

# Use the trained model to make predictions on the same data set
model_predictions = corr_model.predict(X)

# %%
# now we just need to ad the model's predicted values to our UMI chart to see how it does!
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
macro_predictions = macro_data.drop('umi', axis=1).fillna(method='ffill').tail(1).reset_index(drop=True)
macro_variables = macro_predictions.columns

output = widgets.Output()

def update_variable(change, variable):
    with output:
        macro_predictions[variable][0] = change.new
        umi_predictions = corr_model.predict(macro_predictions)
        result = umi_predictions[0]
        clear_output(wait=True)
        display(f'Correlation Model Output: {result:.2f}')

for variable in macro_variables:
    w = widgets.FloatText(value=macro_predictions[variable][0], description=variable)
    w.observe(lambda change, variable=variable: update_variable(change, variable=variable), names='value')
    display(w)
    
display(output)



# %%
