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
             	
             	dash_table.DataTable(id='table',
             		columns=[{"name": i, "id": i} for i in df.columns],
             		data=df.to_dict("rows"),),

             	dcc.Dropdown(id='patient-select',
             		options=[{'label': i, 'value': i} for i in available_patients],
             		value=None,
             		searchable=True,
             		style={'width': '40%'},
             		multi=False),]),
              
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


              	dcc.Graph(id='indicator-graphic')])




@app.callback(
    Output('indicator-graphic', 'figure'),

    [Input('patient-select', 'value'),
    dash.dependencies.Input('my-date-picker-range', 'start_date'),
    dash.dependencies.Input('my-date-picker-range', 'end_date')])

def update_output(patient, start_date, end_date):




	traces = []
	dff = df[df['author'] == patient]
	dfff = dff.loc[start_date:end_date,:] 
	
	
	traces.append(go.Scatter(
            x=dfff['created_at'],
            y=dfff['_percent_abs_words'],
            text = dfff['_time_of_day_overnight'],
            mode='lines+markers',
            opacity=0.7,
            marker={'size': 15,'line': {'width': 0.5, 'color': 'blue'}},
            name='Percentage of Absolutist Words in Text')),
	traces.append(go.Scatter(
            x=dfff['created_at'],
            y=dfff['_sentiment'],
            text=dfff['_sentiment'],
            mode='lines+markers',
            opacity=0.7,
            marker={'size': 15,'line': {'width': 0.5, 'color': 'orange'}},
            name='Sentiment Score'))
	traces.append(go.Scatter(
            x=dfff['created_at'],
            y=dfff['_time_of_day_overnight'],
            text =dfff['_time_of_day_overnight'],
            mode='lines+markers',
            opacity=0.7,
            marker={'size': 15,'line': {'width': 0.5, 'color': 'green'}},
            name='Written Overnight?'))

	figure= {'data': traces,'layout': go.Layout(
            xaxis={'type': 'date', 'title': 'Date', 'range': [start_date, end_date]},
            
            yaxis={'title': 'Percent Abs Words, Time of Day, Sentiment',
            	   'range': [0, 5]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest')}
	return figure

server = app.server



if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
print('done')