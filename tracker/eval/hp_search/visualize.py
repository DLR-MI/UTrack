import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_ace
from dash.exceptions import PreventUpdate
import plotly.express as px


DATA = ['id','dataset','tracker', 'ablation', 'seed', 'HOTA','MOTA','IDF1','DetA','AssA','FPS','HOTA_mean','HOTA_std']
RESULTS_HP = None
MIN_ALPHA = 0.4

df = pd.DataFrame(columns=DATA)
seen = set()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

header = html.Div()
title = html.Div(
    [
        html.H3("Tracking performance"),
    ],
    style={
        "paddingLeft":"1%", 
        "display": "flex", 
        "justifyContent": "center", 
        "alignItems": "center"
    }
)

app.layout = html.Div([
    header,
    title,
    html.Div([
        dcc.Graph(id='graph')
    ], id='fig', style={'display': 'none'}),
    html.Div([
        dcc.Dropdown(
            ['by tracker', 'by dataset'], 
            ['by tracker'], 
            id='facet',
            style={'width': '20%'}
        ),
    ]),
    html.Div([
        html.H4('The config of this experiment is:'),
        dash_ace.DashAceEditor(
            id='exp-id',
            theme='github',
            mode='python',
            tabSize=4,
            height='400px',
        ),
    ], id='inspect', style={'display': 'none'}),
    dcc.Interval(
        id='interval',
        interval=60*1000, # in milliseconds
        n_intervals=0
    )
])


@app.callback(
    [Output('graph', 'figure'), Output("fig", "style")],
    Input('interval', 'n_intervals'),
    State('facet', 'value')
)
def update_figure(n, facet):
    global df
    print(f'{datetime.now()} Checking result files on RESULTS_HP')
    if not os.listdir(RESULTS_HP):
        return {}, {'display': 'none'}
    for filename in Path(RESULTS_HP).glob('*.csv'):
        df_new = pd.read_csv(filename)
        if not filename.stem in seen:
            df_new[['ablation', 'tracker', 'seed']] = df_new.tracker.str.split('_', expand=True)
            df = pd.concat((
                df if not df.empty else None, df_new
            ))
            seen.add(filename.stem)
    facets = facet[0].split(' ')[-1] if isinstance(facet, list) else facet.split(' ')[-1]
    if facets == 'tracker':
        category_orders={'tracker': ['byte', 'botsort', 'deep']}
    else:
        category_orders={'dataset': ['MOT17', 'MOT20', 'dancetrack', 'kitti']}
    fig = px.scatter(
        df, 
        x="IDF1", 
        y="HOTA", 
        color="ablation", 
        facet_col=facets, 
        facet_col_wrap=2, 
        facet_row_spacing=0.12,
        facet_col_spacing=0.06,
        hover_data=['tracker'],
        custom_data=["file", "id"],
        category_orders=category_orders
    )
    fig.update_layout(transition_duration=500, width=800)
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)
    fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig, {'display': 'block'}


@app.callback(
    [Output("exp-id", "value"), Output("inspect", "style")],
    Input("graph", "hoverData"),
)
def hover(hoverData):
    if not hoverData:
        raise PreventUpdate
    filename, id, _ = hoverData['points'][0]['customdata']
    with open(f'{RESULTS_HP}/{filename}.json', 'r') as file:
        exps = json.load(file)
    for key, value in exps.items():
        if key.startswith(id):
            config = value
            break
    print(f'Selected config with id {id}')
    exp_config = json.dumps(config, sort_keys=False, indent=2)
    return exp_config, {'display': 'block'}


def make_parser():
    parser = argparse.ArgumentParser("Dash")
    parser.add_argument("--results_hp", type=str, default='/data/track_results_hp', help="path to results of hp search")
    parser.add_argument("--host", type=str, default="localhost", help="host for visualization server")
    parser.add_argument("--port", type=int, default=8085, help="port for visualization server")
    return parser.parse_args()


def run_dashboard(results_hp, host, port):
    global RESULTS_HP
    RESULTS_HP = results_hp
    app.run(host=host, port=port)
    
    
if __name__ == '__main__':
    args = make_parser()
    RESULTS_HP = args.results_hp
    app.run(host=args.host, port=args.port, debug=True)
    