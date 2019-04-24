import dash
import dash_core_components as dcc
import dash_html_components as html
import dash
import dash_core_components as dcc
import dash_html_components as html
import os
import dash_table
import dash_table.FormatTemplate as FormatTemplate
from dash.dependencies import Input, Output, State
from dash_table.Format import Format, Scheme, Sign, Symbol
import dash_table.FormatTemplate as FormatTemplate
import pandas as pd
import plotly.graph_objs as go
import re
import json
import urllib
import urllib.parse
import datetime as dt

from functools import reduce

TABLE_COLS = ["percent_abs_words",	"post_length",	"sentiment", "subjectivity",
			  "time_of_day_daytime", "time_of_day_evening"	"time_of_day_morning",
			  "time_of_day_overnight", "labels", "created_at", "author", "selftext",
			  "cleaned_text"]
FORMAT_COLS = [{"name": i, "id": i} for i in TABLE_COLS]



# Load data.
df = pd.read_csv('redditors.csv')
df.index = pd.DatetimeIndex(df["created_at"])
df = df.sort_index()


available_patients = df['author'].unique()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'}


app.layout = html.Div(children=[html.H1(children='Mental Health Time Series Tracking',),
                                html.H3(children='Developed by Leah Pilossoph, RN',),
             html.Div(children=[html.H3('Filter by Patient:'),
             	


             	dcc.Dropdown(id='patient-select',
             		options=[{'label': i, 'value': i} for i in available_patients],
             		value=None,
             		searchable=True,
             		style={'width': '40%'},
             		multi=False),
              
              	dcc.DatePickerRange(id='my-date-picker-range',
              		start_date_placeholder_text='Select a date',
              		end_date_placeholder_text='Select a date',
                    min_date_allowed=dt.datetime(2016, 8, 1),
                    max_date_allowed=dt.datetime(2019, 4, 19),
                    updatemode='bothdates',
                    initial_visible_month=dt.datetime(2019, 4, 1),
                    start_date=dt.datetime(2016, 8, 1),
                    end_date=dt.datetime(2019, 4, 30)),
               html.Div(id='output-container-date-picker-range'),

               
               html.H3(children='Percent of Absolutist Words In Post (Scaled 0-5)'),               
              	dcc.Graph(id='percent_abs_words-graphic'),
              	html.H3(children='Sentiment of Post (Scaled 0-5)'),
              	dcc.Graph(id='sentiment-graphic'),])])




@app.callback(
    Output('percent_abs_words-graphic', 'figure'),

    [Input('patient-select', 'value'),
    dash.dependencies.Input('my-date-picker-range', 'start_date'),
    dash.dependencies.Input('my-date-picker-range', 'end_date')])

def update_percabs(patient, start_date, end_date):
	df = pd.read_csv('redditors.csv')
	traces = []
	df.index = pd.DatetimeIndex(df["created_at"])
	df = df.sort_index()

	dff = df[df['author'] == patient]
	dfff = dff.loc[start_date:end_date,:] 
	
	return {'data': [go.Scatter(x=dfff['created_at'],
            y=dfff['percent_abs_words'],
            text = dfff['percent_abs_words'],
            mode='lines+markers',
            opacity=1.0,
            marker={'size': 15,'line': {'width': 1, 'color': 'red'}},
            name='Percentage of Absolutist Words in Text Scaled (0-5)')],
			'layout': [go.Layout(
            xaxis={'type': 'date', 'title': 'Date', 'range': [start_date, end_date]},
            legend={'x': 0, 'y': 1},
            yaxis={'title': 'Percent Abs Words',
            	   'range': [0, 5]},)]}
@app.callback(
    Output('sentiment-graphic', 'figure'),

    [Input('patient-select', 'value'),
    dash.dependencies.Input('my-date-picker-range', 'start_date'),
    dash.dependencies.Input('my-date-picker-range', 'end_date')])

def update_sentiment(patient, start_date, end_date):
	df = pd.read_csv('redditors.csv')
	traces = []
	df.index = pd.DatetimeIndex(df["created_at"])
	df = df.sort_index()

	dff = df[df['author'] == patient]
	dfff = dff.loc[start_date:end_date,:] 
	return {'data': [go.Scatter(x=dfff['created_at'],
            y=dfff['_sentiment'],
            text = dfff['_sentiment'],
            mode='lines+markers',
            opacity=0.5,
            marker={'size': 15,'line': {'width': 0.5, 'color': 'blue'}},
            name='Sentiment of text Scaled (0-5)')],
			'layout': [go.Layout(
            xaxis={'type': 'date', 'title': 'Date', 'range': [start_date, end_date]},
            legend={'x': 0, 'y': 1},

            yaxis={'title': 'Sentiment',
            	   'range': [0, 5]},)]}
 
	

server = app.server



if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
print('done')