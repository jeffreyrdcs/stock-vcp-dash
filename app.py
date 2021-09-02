import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime, timedelta, date
import time



def make_performance_table(df):
    '''
    Return a dash definition of an HTML table for a dataframe. For daily performance.
    # [html.Tr([html.Td(4),html.Td(2)]), html.Tr([html.Td(5),html.Td(1),html.Td(1)]) ]
    '''
    table = []
    for ind, col in enumerate(df.columns):
        if 'Tickers' in df.columns[ind]:
            pass
        elif df.columns[ind][0:5] == 'Gauge':
            table.append(html.Tr([html.Td(df.columns[ind]), html.Td(f'{df[col].values[0]:.2f}', className='tdcol2')]))
        elif df.columns[ind][0:5] == 'Stock':
            table.append(html.Tr([html.Td(df.columns[ind]), html.Td(f'{df[col].values[0]:.2f}')]))
        elif df.columns[ind] == 'AD Percent' or df.columns[ind][0:5] == 'Perce':
            table.append(html.Tr([html.Td(df.columns[ind]), html.Td(f'{df[col].values[0]:.2f}%')]))
        else:
            table.append(html.Tr([html.Td(df.columns[ind]), html.Td(f'{df[col].values[0]}')]))
    return html.Table(table)


def make_stock_info_table(df):
    '''
    Return a dash definition of an HTML table for a dataframe. For selected stock info.
    Only return top 50 of them
    '''
    num_out = 50
    sel_df = df.loc[0:num_out-1]
    default_url = 'https://uk.finance.yahoo.com/quote/'

    table = []

    # Header of the table
    header = [html.Td('Rank')]
    for col in sel_df.columns:
        header.append(html.Td(col))
    table.append(html.Tr(header))

    for ind, col in sel_df.iterrows():
        tmprow = []

        for key, item in zip(col.keys(), col):
            if key == 'Ticker':
                tmprow.append(html.Td(html.A(f'{item}', target='_blank', href=default_url+item)))
            elif key == 'Volume':
                tmprow.append(html.Td(f'{item:.0f}'))
            elif key == 'RS Rank':
                tmprow.append(html.Td(f'{item:.4f}'))
            elif key[0:2] == '52':
                tmprow.append(html.Td(f'{item:.2f}', className='tdfixsize'))
            elif key[0:2] == 'Ch' and item >= 0:
                tmprow.append(html.Td(f'+{item:.2f}', className='tdchangepos'))
            elif key[0:2] == 'Ch' and item < 0:
                tmprow.append(html.Td(f'{item:.2f}', className='tdchangeneg'))
            else:
                tmprow.append(html.Td(f'{item:.2f}'))
        tmprow.insert(0, html.Td(str(ind+1)))
        table.append(html.Tr(tmprow))

    return html.Table(table, className='trfixsize')


def convert_str_column(in_df):
    ''' Convert the string coloumn back into a array of float '''
    in_df = re.sub("[\[\]']",'',in_df)
    return np.array(in_df.split(',')).astype(np.float)


def convert_str_column_str(in_df):
    ''' Convert the string coloumn back into a array of str '''
    in_df = re.sub("[\[\]']",'',in_df)
    return np.array(in_df.split(','))


def date_to_dropdown_list(in_df):
    '''
    Get the index of the input dataframe and turn the index
    into a dictionary for the dropdown list
    '''
    out_list = []
    tmp_arr = in_df.index.to_numpy()
    for i in tmp_arr:
        out_list.append({'label':str(i), 'value':str(i)})

    return out_list


def ticker_to_dropdown_list(in_df):
    '''
    Get the ticker column of the input dataframe and turn it
    into a dictionary for the dropdown list
    '''
    out_list = []
    tmp_arr = in_df['Ticker'].to_numpy()
    for ind, i in enumerate(tmp_arr):
        out_list.append({'label':str(ind+1)+'. '+str(i), 'value':str(i)})

    return out_list


def get_ohlc_data(in_ticker):
    '''
    Fetch OHLC data from local csv file, format the dataFrame and compute SMA
    '''
    db_add = '../stock_vcpscreener/db_yfinance/'
    tdf = pd.read_csv(db_add+in_ticker.strip().ljust(5,'_')+'.csv')
    tdf['Date'] = pd.to_datetime(tdf['Date'])
    tdf.set_index('Date', inplace=True)

    tdf['SMA_20'] = tdf['Adj Close'].rolling(window=20).mean()
    tdf['SMA_50'] = tdf['Adj Close'].rolling(window=50).mean()
    tdf['SMA_200'] = tdf['Adj Close'].rolling(window=200).mean()

    return tdf


def get_ohlc_data_web(in_ticker):
    '''
    Fetch OHLC data from yahoo finance, format the dataFrame and compute SMA
    '''
    yf.pdr_override()
    curr_day = datetime.utcnow() - timedelta(hours=5)       # UTC -5, i.e. US NY timezone

    tdf = pdr.get_data_yahoo(in_ticker.strip(),
                            start=curr_day.date()-timedelta(days=365),
                            end=curr_day.date(), threads=False)    # a year of data
    # tdf['Date'] = pd.to_datetime(tdf['Date'])
    # tdf.set_index('Date', inplace=True)
    tdf['SMA_20'] = tdf['Adj Close'].rolling(window=20).mean()
    tdf['SMA_50'] = tdf['Adj Close'].rolling(window=50).mean()
    tdf['SMA_200'] = tdf['Adj Close'].rolling(window=200).mean()

    return tdf


app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
app.title = "Stock Analysis Report"

server = app.server


# Read in the daily stock data
stock_info_url = 'https://raw.githubusercontent.com/jeffreyrdcs/stock-vcpscreener/main/daily_selected_stock_info.csv'
df = pd.read_csv(stock_info_url)
df = df.set_index('Date')

# Read in the corresponding info dataset of the most recent day
selected_info_url = f'https://raw.githubusercontent.com/jeffreyrdcs/stock-vcpscreener/main/output/selected_stock_{df.index[-1]}.csv'
df_info = pd.read_csv(selected_info_url)
df_info = df_info.drop(df_info.columns[0], axis=1)

# Move the location of the change columns
df_info.rename(columns={'Change': 'Change_tmp'}, inplace=True)
df_info.rename(columns={'Change (%)': 'Change_%_tmp'}, inplace=True)
df_info.insert(6, 'Change', df_info['Change_tmp'])
df_info.insert(7, 'Change (%)', df_info['Change_%_tmp'])
df_info = df_info.drop('Change_tmp', axis=1)
df_info = df_info.drop('Change_%_tmp', axis=1)

# Get the list for droplist
out_list = date_to_dropdown_list(df)

# Get the stock list of the first day for droplist
out_stock_list = ticker_to_dropdown_list(df_info)

# Make a display copy
df_dis = pd.DataFrame([],index=df.index)
df_dis['Number of stocks monitored'] = df['Number of stock']
df_dis['Advanced / Declined stock'] = df['Advanced (Day)'].astype(str) +' / '+ df['Declined (Day)'].astype(str)
df_dis['AD Percent'] = (df['Advanced (Day)'] - df['Declined (Day)'])/(df['Advanced (Day)'] + df['Declined (Day)']) * 100
df_dis['New 52W high / New 52W low'] = df['New High'].astype(str) +' / '+ df['New Low'].astype(str)
df_dis['Gauge (Billion $)'] = df['Gauge'] / 1e9
df_dis['Percentage of stocks above its 20 Day SMA (SMA 20)'] = df['Stock above 20-DMA']
df_dis['Percentage of stocks above its 50 Day SMA (SMA 50)'] = df['Stock above 50-DMA']
df_dis['Percentage of stocks with SMA 20 > SMA 50'] = df['Stock with 20-DMA > 50-DMA']
# df_dis['Percentage of stocks with 50 Day SMA > 200 Day SMA'] = df['Stock with 50-DMA > 200-DMA']
df_dis['Percentage of stocks with SMA 50 > SMA 150 > SMA 200'] = df['Stock with 50 > 150 > 200-DMA']
df_dis['Percentage of stocks trending 200 Day SMA'] = df['Stock with 200-DMA is rising']
df_dis['Number of stocks that fit the criteria'] = df['Number of Stock that fit condition']
df_dis['Percentage of stocks that fit the criteria'] = df['Number of Stock that fit condition(%)']

# Convert the string column into an object column
df['Breadth Percentage'] = df['Breadth Percentage'].apply(convert_str_column)
df['Tickers that fit the conditions'] = df['Tickers that fit the conditions'].apply(convert_str_column_str)
df['RS rating of Tickers'] = df['RS rating of Tickers'].apply(convert_str_column)
df['RS rank of Tickers'] = df['RS rank of Tickers'].apply(convert_str_column)

# Compute the AD Percentage
# df['Tmp'] = (df['Advanced (Day)'] - df['Declined (Day)'])/(df['Advanced (Day)'] + df['Declined (Day)']) * 100
# df.insert(3, 'AD Percent', df['Tmp'])
# df = df.drop('Tmp', axis=1)


# ------------------------------------------------------------------------------
# Page layout
app.layout = html.Div([

    # html.H4("Dashboard with Dash", style={'text-align': 'center'}, className="padded"),

    dcc.Dropdown(id="check_date",
                 options=out_list,
                 multi=False,
                 value=out_list[-1]['value'],
                 style={'width': "50%", 'float':'right'}
                 ),

    html.Div(id='output_container', children=[],
            style={"margin-bottom": "20px"}, className='headtitle padded'
            ),

    html.Div(
        [
            html.H6("What are we showing here?"),
            html.Br([]),
            dcc.Markdown('''
            The top stocks in the US market are selected based on multiple criteria applied to \
            the simple moving averages and price performance over the last year. \
            This report is generated based on the output of a custom US stock screener package 'stock_vcpscreener'. \
            The source code of the stock screener package and this dashboard can be found [here](https://github.com/jeffreyrdcs/stock-vcpscreener) at my github. \
            The screener calculates various market breadth indicators and selects stocks on a daily basis based on the criteria. \
            To rank the selected stocks, a rating score is computed using past performances, similar to the IBD RS rating. \
            The rating and rank of the stock can be found in the summary table.''',
            style={"color": "#ffffff"},
            className="row",
            ),
        ],
        className='s-summary',
        style={"margin-bottom": "15px"},
    ),


    # Row 1
    html.Div(
        [
            html.Div(
                [
                    html.H6(
                        "Stock Ratings (Top 50 that fit the criteria)", className="subtitle padded"
                    ),
                    dcc.Graph(id='stock_bar',
                              figure={},
                              config={"displayModeBar": False})
                ],
                className="twelve columns",
            ),
        ],
        className="row",
        style={"margin-bottom": "5px"},
    ),


    # Row 2
    html.Div(
        [
            html.Div(
                [
                    html.H6(
                        "Daily Market Breadth", className="subtitle padded"
                    ),
                    dcc.Graph(id='breadth_hist',
                              figure={},
                              config={"displayModeBar": False})
                ],
                className="six columns",
            ),
            html.Div(
                [
                    html.H6(
                        "Daily Market Performance",
                        className="subtitle padded",
                    ),
                    html.Div(make_performance_table(df_dis[df.index == df.index[-1]]),
                        id='daily_report'
                        ),
                ],
                className="six columns",
            ),
        ],
        className="row",
        style={"margin-bottom": "5px"},
    ),

    # Row 3
    html.Div(
        [
            html.Div(
                [
                    html.H6(
                        "Stock Summary (Top 50 that fit the criteria)", className="subtitle padded"
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                html.Div(make_stock_info_table(df_info),
                                    id='stock_report'
                                    ),
                                ], className="table-scroll"
                            ),
                        ], className="table-wrapper"),
                ],
                className="twelve columns",
            ),
        ],
        className="row",
        style={"margin-bottom": "15px"},
    ),

    # Row 4
    html.Div(
        [
            html.Div(
                [
                    html.H6(
                        "Charts", className="subtitle padded",
                    ),
                ], className="twelve columns"),

            html.Div(
                [
                    dcc.Dropdown(id="check_stock1",
                                 options=out_stock_list,
                                 multi=False,
                                 value=out_stock_list[0]['value'],
                                 style={'width': '50%'}
                                 ),
                    dcc.Graph(id='stock_chart1',
                              figure={},
                              config={"displayModeBar": False})
                ],
                className="six columns",
            ),

            html.Div(
                [
                    dcc.Dropdown(id="check_stock2",
                                 options=out_stock_list,
                                 multi=False,
                                 value=out_stock_list[1]['value'],
                                 style={'width': '50%'}
                                 ),
                    dcc.Graph(id='stock_chart2',
                              figure={},
                              config={"displayModeBar": False})
                ],
                className="six columns",
            ),

            html.Div(
                [
                    dcc.Dropdown(id="check_stock3",
                                 options=out_stock_list,
                                 multi=False,
                                 value=out_stock_list[2]['value'],
                                 style={'width': '50%'}
                                 ),
                    dcc.Graph(id='stock_chart3',
                              figure={},
                              config={"displayModeBar": False})
                ],
                className="six columns",
            ),

            html.Div(
                [
                    dcc.Dropdown(id="check_stock4",
                                 options=out_stock_list,
                                 multi=False,
                                 value=out_stock_list[3]['value'],
                                 style={'width': '50%'}
                                 ),
                    dcc.Graph(id='stock_chart4',
                              figure={},
                              config={"displayModeBar": False})
                ],
                className="six columns",
            ),
        ],
        className="row",
        style={"margin-bottom": "15px"},
    ),
],className="page")


# ------------------------------------------------------------------------------
# Call back functions
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='breadth_hist', component_property='figure'),
     Output(component_id='daily_report', component_property='children'),
     Output(component_id='stock_bar', component_property='figure'),
     Output(component_id='stock_report', component_property='children'),
     Output(component_id='check_stock1', component_property='options'),
     Output(component_id='check_stock2', component_property='options'),
     Output(component_id='check_stock3', component_property='options'),
     Output(component_id='check_stock4', component_property='options'),
     Output(component_id='check_stock1', component_property='value'),
     Output(component_id='check_stock2', component_property='value'),
     Output(component_id='check_stock3', component_property='value'),
     Output(component_id='check_stock4', component_property='value')
     ],
    [Input(component_id='check_date', component_property='value')]
)
def display_page(in_check_date):
    print(f'Viewing report for {in_check_date}')

    dff = df.copy()
    dff_dis = df_dis.copy()
    dff = dff[dff.index == in_check_date]
    dff_dis = dff_dis[dff_dis.index == in_check_date]

    # For the daily breadth histogram
    dff_histodata = dff['Breadth Percentage'].to_numpy()[0]
    to_plot = (dff_histodata > -20) & (dff_histodata < 20)
    # print(len(dff_histodata[to_plot]))

    # Read in the corresponding info dataset
    selected_info_url = f'https://raw.githubusercontent.com/jeffreyrdcs/stock-vcpscreener/main/output/selected_stock_{in_check_date}.csv'
    dff_info = pd.read_csv(selected_info_url)
    dff_info = dff_info.drop(dff_info.columns[0], axis=1)

    # Move the location of the change columns
    dff_info.rename(columns={'Change': 'Change_tmp'}, inplace=True)
    dff_info.rename(columns={'Change (%)': 'Change_%_tmp'}, inplace=True)
    dff_info.insert(6, 'Change', dff_info['Change_tmp'])
    dff_info.insert(7, 'Change (%)', dff_info['Change_%_tmp'])
    dff_info = dff_info.drop('Change_tmp', axis=1)
    dff_info = dff_info.drop('Change_%_tmp', axis=1)


    # Update text in the status container
    container = "US Stock Market Analysis Report for {}".format(in_check_date)  #, len(dff['Breadth Percentage'].values[0]

    # Update stock rating plot
    fig = px.bar(
                y=dff['RS rating of Tickers'].iloc[0][0:50], color_continuous_scale=px.colors.sequential.Greens_r[1:7],
                color=np.linspace(0,255,50),
                x=dff['Tickers that fit the conditions'].iloc[0][0:50],
                orientation='v')
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(showlegend=False, autosize=False, hovermode="x",
                      height=200, width=945,
                      font_family="Arial",
                      margin={
                            "r": 0,
                            "t": 2,
                            "b": 20,
                            "l": 10,
                                },
                      plot_bgcolor='rgba(250,250,250,1)',
                      yaxis_title='RS Rating',
                      xaxis_title="", title_x=0.5, title_y=1.0)

    # Update daily breadth plot
    fig2 = px.histogram(x=dff_histodata[to_plot], range_x=[-20,20], nbins=100,
                    labels={"value": "Percentage Change (%)"},
                    color_discrete_sequence=['#009900'], title='')
    fig2.add_annotation(xref="x domain",
                    yref="y domain",
                    x=0.025,
                    y=0.975,
                    showarrow=False,
                    text=f"Net Breadth (AD Percent) = {dff_dis['AD Percent'].values[0]:.2f}%"
                )
    fig2.add_vline(x=0, line_width=2, line_dash="dash",
                   line_color='rgba(227,227,227,0.75)')
    # fig2.add_shape(type='line', xref='x', yref='y',
    #                         x0=0, y0=0, x1=0, y1=1000, line=dict(dash='dash', color='rgba(227,227,227,0.75)', width=1.5)
    #             )
    fig2.update_layout(showlegend=False, autosize=False, hovermode="x",
                      height=275, width=450,
                      font_family="Arial",
                      margin={
                            "r": 0,
                            "t": 2,
                            "b": 20,
                            "l": 10,
                                },
                      plot_bgcolor='rgba(250,250,250,1)',
                      xaxis_title='Percentage Change (%)',
                      yaxis_title="Count", title_x=0.5, title_y=1.0)


    # Update daily table
    table_daily = make_performance_table(dff_dis)

    # Update stock info table
    table_info = make_stock_info_table(dff_info)

    # Update the check_stock1 and 2 dropdown list
    out_stock_list1 = ticker_to_dropdown_list(dff_info)

    return container, fig2, table_daily, fig, table_info, out_stock_list1, out_stock_list1, out_stock_list1, out_stock_list1, \
        out_stock_list1[0]['value'], out_stock_list1[1]['value'], out_stock_list1[2]['value'], out_stock_list1[3]['value'],


# Callback to update the four stock OHLC chart. Made them individual function since we can change the stock for each chart
# Callback to update stock_graph1
@app.callback(
     [Output(component_id='stock_chart1', component_property='figure')],
    [Input(component_id='check_stock1', component_property='value'),
    Input(component_id='check_date', component_property='value')]
)
def display_stock_graph1(in_ticker, in_date):
    '''
    Currently OHLC data is fetched online
    '''
    df = get_ohlc_data_web(in_ticker)
    # df = get_ohlc_data(in_ticker)

    fig = go.Figure(data=[go.Ohlc(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'], name=in_ticker),
                        go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'),
                        go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='green', width=1), name='SMA 50'),
                        go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='darkblue', width=1), name='SMA 200'),
                    ])

    fig.update_layout(showlegend=False, autosize=False, hovermode="x",
                      height=370, width=465,
                      font_family="Arial",
                      margin={
                            "r": 0,
                            "t": 10,
                            "b": 20,
                            "l": 0,
                                },
                      plot_bgcolor='rgba(250,250,250,1)',
                      xaxis_title='',
                      yaxis_title='', title_x=0.5, title_y=1.0)

    fig.add_vline(x=in_date, line_width=2, line_dash="dash",
                   line_color='rgba(227,227,227,0.75)')

    return fig,


# Callback to update stock_graph2
@app.callback(
     [Output(component_id='stock_chart2', component_property='figure')],
    [Input(component_id='check_stock2', component_property='value'),
    Input(component_id='check_date', component_property='value')]
)
def display_stock_graph2(in_ticker, in_date):
    '''
    Currently OHLC data is fetched online
    '''
    df = get_ohlc_data_web(in_ticker)
    # df = get_ohlc_data(in_ticker)

    fig = go.Figure(data=[go.Ohlc(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'], name=in_ticker),
                        go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'),
                        go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='green', width=1), name='SMA 50'),
                        go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='darkblue', width=1), name='SMA 200'),
                    ])

    fig.update_layout(showlegend=False, autosize=False, hovermode="x",
                      height=370, width=465,
                      font_family="Arial",
                      margin={
                            "r": 0,
                            "t": 10,
                            "b": 20,
                            "l": 0,
                                },
                      plot_bgcolor='rgba(250,250,250,1)',
                      xaxis_title='',
                      yaxis_title='', title_x=0.5, title_y=1.0)

    fig.add_vline(x=in_date, line_width=2, line_dash="dash",
                   line_color='rgba(227,227,227,0.75)')

    return fig,


# Callback to update stock_graph3
@app.callback(
     [Output(component_id='stock_chart3', component_property='figure')],
    [Input(component_id='check_stock3', component_property='value'),
    Input(component_id='check_date', component_property='value')]
)
def display_stock_graph3(in_ticker, in_date):
    '''
    Currently OHLC data is fetched online
    '''
    df = get_ohlc_data_web(in_ticker)
    # df = get_ohlc_data(in_ticker)

    fig = go.Figure(data=[go.Ohlc(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'], name=in_ticker),
                        go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'),
                        go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='green', width=1), name='SMA 50'),
                        go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='darkblue', width=1), name='SMA 200'),
                    ])

    fig.update_layout(showlegend=False, autosize=False, hovermode="x",
                      height=370, width=465,
                      font_family="Arial",
                      margin={
                            "r": 0,
                            "t": 10,
                            "b": 20,
                            "l": 0,
                                },
                      plot_bgcolor='rgba(250,250,250,1)',
                      xaxis_title='',
                      yaxis_title='', title_x=0.5, title_y=1.0)

    fig.add_vline(x=in_date, line_width=2, line_dash="dash",
                   line_color='rgba(227,227,227,0.75)')

    return fig,


# Callback to update stock_graph4
@app.callback(
     [Output(component_id='stock_chart4', component_property='figure')],
    [Input(component_id='check_stock4', component_property='value'),
    Input(component_id='check_date', component_property='value')]
)
def display_stock_graph4(in_ticker, in_date):
    '''
    Currently OHLC data is fetched online
    '''
    df = get_ohlc_data_web(in_ticker)
    # df = get_ohlc_data(in_ticker)

    fig = go.Figure(data=[go.Ohlc(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'], name=in_ticker),
                        go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'),
                        go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='green', width=1), name='SMA 50'),
                        go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='darkblue', width=1), name='SMA 200'),
                    ])

    fig.update_layout(showlegend=False, autosize=False, hovermode="x",
                      height=370, width=465,
                      font_family="Arial",
                      margin={
                            "r": 0,
                            "t": 10,
                            "b": 20,
                            "l": 0,
                                },
                      plot_bgcolor='rgba(250,250,250,1)',
                      xaxis_title='',
                      yaxis_title='', title_x=0.5, title_y=1.0)

    fig.add_vline(x=in_date, line_width=2, line_dash="dash",
                   line_color='rgba(227,227,227,0.75)')

    return fig,


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)  #debug=True

