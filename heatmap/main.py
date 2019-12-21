import os
from random import randint

import dash
from flask import Flask

from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from pandas import read_csv

# Mapbox key to display the map
MAPBOX = 'pk.eyJ1Ijoicm11aXIiLCJhIjoiY2o1MjBxcnkwMDdnZTJ3bHl5bXdxNW9uaCJ9.QR6f0fRLkHzmCgL70u5Hzw'

# Make the colours consistent for each type of accident
SEVERITY_LOOKUP = {'Fatal': 'red',
                   'Serious': 'orange',
                   'Slight': 'yellow'}

SLIGHT_FRAC = 0.2
SERIOUS_FRAC = 0.5

DAYSORT = dict(zip(['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday'],
                   [4, 0, 5, 6, 3, 1, 2]))

# Set the global font family
FONT_FAMILY = "Arial"
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data/2017.csv')

acc = read_csv(filename).dropna(how='any', axis=0)

acc = acc[~acc['Speed_limit'].isin([0, 10])]
# Create an hour column
acc['Hour'] = acc['Time'].apply(lambda x: int(x[:2]))

print([column for column in acc])

external_scripts = [{
    'src': 'assets/layui.js'
}]
external_stylesheets = [{
    'href': 'assets/layui.css',
    'rel': 'stylesheet'
},
    {
    'href': 'assets/web.css',
    'rel': 'stylesheet'
    }
]

server = Flask(__name__, static_url_path="")
server.secret_key = os.environ.get('secret_key', 'secret')
app = dash.Dash(__name__, server=server, external_scripts=external_scripts, external_stylesheets=external_stylesheets)

# Main layout container
app.layout = html.Div([
    html.Ul([
        html.Li([
            html.A(
                'Go Back',
                href='https://sf2977.wixsite.com/bigdata',
                className='layui-nav-item',
            )
        ],
            className='layui-nav-item',
        ),
        html.Li([
            html.A(
                'Prediction',
                href='http://127.0.0.1:5000',
                className='layui-nav-item',
            )
        ],
            className='layui-nav-item',
        ),
        html.Li([
            dcc.Upload(
                id='accyear',
                children=html.A('Upload File')
            ),
        ],
            className='layui-nav-item',
        ),
        html.Li([
            html.H3(id="res"),
        ],
            className='layui-nav-item',
        )
    ],
        className='layui-nav',
    ),
    html.Div([
        html.Div([  # Holds the widgets & Descriptions
            html.Div([
                html.Label('Severity'),
                dcc.Checklist(  # Checklist for the three different severity values
                    options=[
                        {'label': sev, 'value': sev} for sev in acc['Accident_Severity'].unique()
                    ],
                    value=[sev for sev in acc['Accident_Severity'].unique()],
                    labelStyle={
                        'display': 'inline-block',
                        'paddingRight': 10,
                        'paddingLeft': 10,
                        'paddingBottom': 5,
                    },
                    id="severity",
                ),

                html.Label('Day of Week'),
                dcc.Checklist(  # Checklist for the dats of week, sorted using the sorting dict created earlier
                    options=[
                        {'label': day[:3], 'value': day} for day in
                        sorted(acc['Day_of_Week'].unique(), key=lambda k: DAYSORT[k])
                    ],
                    value=[day for day in acc['Day_of_Week'].unique()],
                    labelStyle={  # Different padding for the checklist elements
                        'display': 'inline-block',
                        'paddingRight': 10,
                        'paddingLeft': 10,
                        'paddingBottom': 5,
                    },
                    id="day",
                ),

                html.Label('Hour of Day'),
                html.Div([
                    dcc.RangeSlider(  # Slider to select the number of hours
                        id="hour",
                        count=1,
                        min=-acc['Hour'].min(),
                        max=acc['Hour'].max(),
                        step=1,
                        value=[acc['Hour'].min(), acc['Hour'].max()],
                        marks={str(h): str(h) for h in range(acc['Hour'].min(), acc['Hour'].max() + 1)},
                    ),
                ],
                    className="demo-slider",
                    style={
                        "paddingBottom": 30
                    }
                ),

                html.Div(
                    html.Label('''Urban or Rural Area'''),
                ),

                html.Div([
                    dcc.RadioItems(
                        id='area',
                        options=[
                                    {'label': area, 'value': area} for area in acc['Urban_or_Rural_Area'].unique()
                                ] + [{'label': 'All', 'value': 'All'}],
                        value="All",
                        labelStyle={
                            'paddingTop': 10,
                            'paddingRight': 15,
                            'paddingLeft': 5,
                            'paddingBottom': 5,
                        },
                        style={
                            'display': 'inline-block',
                            'paddingTop': 15,
                            'paddingRight': 20,
                            'paddingLeft': 5,
                            'paddingBottom': 5,
                        },
                    ),
                ],
                    style={
                        "paddingBottom": 10
                    }
                ),

                html.Label('''Weather Condition  '''),
                html.Div(
                    dcc.Dropdown(
                        id='weather',
                        options=[
                                    {'label': weather, 'value': weather} for weather in
                                    acc['Weather_Conditions'].unique()
                                ] + [{'label': 'All', 'value': 'All'}],
                        value='All'
                    ),

                    style={
                        "paddingBottom": 20
                    }
                ),

                html.Label('''Light Condition  '''),
                html.Div(
                    dcc.Dropdown(
                        id='light',
                        options=[
                                    {'label': light, 'value': light} for light in acc['Light_Conditions'].unique()
                                ] + [{'label': 'All', 'value': 'All'}],
                        value='All'
                    ),
                    style={
                        "paddingBottom": 20
                    },
                ),

                html.Label('''Road Types'''),
                html.Div(
                    dcc.Dropdown(
                        id='road',
                        options=[
                                    {'label': road, 'value': road} for road in acc['Road_Type'].unique()
                                ] + [{'label': 'All', 'value': 'All'}],
                        value='All'
                    ),
                    style={
                        "paddingBottom": 20
                    },
                ),

                html.Label('''Road Surface Conditions'''),
                html.Div(
                    dcc.Dropdown(
                        id='surface',
                        options=[
                                    {'label': sur, 'value': sur} for sur in acc['Road_Surface_Conditions'].unique()
                                ] + [{'label': 'All', 'value': 'All'}],
                        value='All'
                    ),
                    style={
                        "paddingBottom": 20
                    },
                ),

                html.Label('''Speed Limit '''),
                dcc.Dropdown(
                    id='speed',
                    options=[
                                {'label': speed, 'value': speed} for speed in acc['Speed_limit'].unique()
                            ] + [{'label': 'All', 'value': 'All'}],
                    value='All',
                ),

            ],
            ),
        ],
            style={
                'paddingTop': 30,
                'paddingBottom':15
            }
        )],
        className='box',
        style={
            "width": '30%',
            'display': 'inline-block',
            'paddingLeft': 30,
            'paddingRight': 30,
            'paddingTop': 0,
            'boxSizing': 'border-box',
        }
    ),

    html.Div([  # Holds the map & the widgets
        dcc.Graph(id="map")  # Holds the map in a div to apply styling to it
    ],
        style={
            "width": '65%',
            "height": 600,
            'float': 'right',
            'paddingRight': 50,
            'paddingLeft': 10,
            'paddingTop': 15,
            'boxSizing': 'border-box',
            'fontFamily': FONT_FAMILY
        }),
],
    style={
        'background-image':'url("assets/bkg-1.jpg")',
        'background-repeat': 'no-repeat',
        'background-size': 'cover',
        'height': 900
    }
)

@app.callback(
    Output('res','children'),
    [Input('accyear', 'filename')])
#update year according to file name
def get_year(accyear):
    global dirname
    global filename
    global acc
    if accyear != None:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'data/'+accyear.split('.')[0]+'.csv')
        acc = read_csv(filename).dropna(how='any', axis=0)
        acc = acc[~acc['Speed_limit'].isin([0, 10])]
        acc['Hour'] = acc['Time'].apply(lambda x: int(x[:2]))
        return accyear.split('.')[0]

@app.callback(
    Output('map', 'figure'),
     [Input('severity', 'value'),
     Input('day', 'value'),
     Input('hour', 'value'),
     Input('area', 'value'),
     Input('weather', 'value'),
     Input('light', 'value'),
     Input('road', 'value'),
     Input('speed', 'value'),
     Input('surface', 'value')
     ]
)
def updateMapBox(severity, days, time, area, weather, light, road, speed, surface):
    hours = [i for i in range(time[0], time[1] + 1)]
    df = acc[
        (acc['Accident_Severity'].isin(severity)) &
        (acc['Day_of_Week'].isin(days)) &
        (acc['Hour'].isin(hours))
        ]

    if area != 'All':
        df = df[df['Urban_or_Rural_Area'] == area]

    if weather != 'All':
        df = df[df['Weather_Conditions'] == weather]

    if light != 'All':
          df = df[df['Light_Conditions'] == light]

    if road != 'All':
        df = df[df['Road_Type'] == road]

    if speed != 'All':
        df = df[df['Speed_limit'] == speed]

    if surface != 'All':
        df = df[df['Road_Surface_Conditions'] == surface]

    traces = []
    for sev in sorted(severity, reverse=True):
        # Set the downsample fraction depending on the severity
        sample = 1
        if sev == 'Slight':
            sample = SLIGHT_FRAC
        elif sev == 'Serious':
            sample = SERIOUS_FRAC
        # Downsample the dataframe and filter to the current value of severity
        acc3 = df[df['Accident_Severity'] == sev].sample(frac=sample)

        # Scattermapbox trace for each severity
        traces.append({
            'type': 'scattermapbox',
            'mode': 'markers',
            'lat': acc3['Latitude'],
            'lon': acc3['Longitude'],
            'marker': {
                'color': SEVERITY_LOOKUP[sev],  # Keep the colour consistent
                'size': 2,
            },
            'hoverinfo': 'text',
            'name': sev,
            'legendgroup': sev,
            'showlegend': False,
            'text': acc3['Local_Authority_(District)']  # Text will show location
        })

    layout = {
        'height': 600,
        'paper_bgcolor': 'rgb(26,25,25)',
        'font': {
            'color': 'rgb(250,250,250'
        },  # Set this to match the colour of the sea in the mapbox colourscheme
        'autosize': True,
        'hovermode': 'closest',
        'mapbox': {
            'accesstoken': MAPBOX,
            'center': {  # Set the geographic centre - trial and error
                'lat': 54.5,
                'lon': -2
            },
            'zoom': 5,
            'style': 'mapbox://styles/mapbox/streets-v11',
        },
        'margin': {'t': 0,
                   'b': 0,
                   'l': 0,
                   'r': 0},
        'legend': {
            'font': {'color': 'white'},
            'orientation': 'h',
            'x': 0,
            'y': 1.01
        }
    }
    fig = dict(data=traces, layout=layout)
    return fig


# Run the Dash app
if __name__ == '__main__':
    app.server.run(host='0.0.0.0', port=8000, debug=False, threaded=True)



