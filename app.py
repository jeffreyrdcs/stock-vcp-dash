import re
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pandas_datareader import data as pdr

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import yfinance as yf

_DEFAULT_URL_PATH_NAME = "https://uk.finance.yahoo.com/quote/"
_STOCK_INFO_URL_PATH_NAME = 'https://raw.githubusercontent.com/jeffreyrdcs/stock-vcpscreener/main/'
_MAX_NUM_OF_STOCK_TO_DISPLAY = 50
_NUM_OF_STOCK_CHART_TO_DISPLAY = 4


app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
app.title = "Stock Analysis Report"

server = app.server


def make_performance_table(df):
    """
    Return a dash definition of an HTML table for a dataframe. For daily performance.
    # [html.Tr([html.Td(4),html.Td(2)]), html.Tr([html.Td(5),html.Td(1),html.Td(1)]) ]
    """
    table = []

    for col in df.columns:
        if "Tickers" in col:
            continue
        elif col[0:5] == "Gauge":
            table.append(html.Tr([html.Td(col), html.Td(f"{df[col].values[0]:.2f}", className="tdcol2")]))
        elif col[0:5] == "Stock":
            table.append(html.Tr([html.Td(col), html.Td(f"{df[col].values[0]:.2f}")]))
        elif col == "AD Percent" or col[0:5] == "Perce":
            table.append(html.Tr([html.Td(col), html.Td(f"{df[col].values[0]:.2f}%")]))
        else:
            table.append(html.Tr([html.Td(col), html.Td(f"{df[col].values[0]}")]))

    return html.Table(table)


def make_stock_info_table(df):
    """Return a dash definition of an HTML table for a dataframe. For selected stock info. Only return the top 50."""
    selected_stock_df = df.loc[0:_MAX_NUM_OF_STOCK_TO_DISPLAY - 1]

    table = []

    header = [html.Td("Rank")]
    for col in selected_stock_df.columns:
        header.append(html.Td(col))
    table.append(html.Tr(header))

    for ind, col in selected_stock_df.iterrows():
        tmprow = []

        for key, item in zip(col.keys(), col):
            if key == "Ticker":
                tmprow.append(html.Td(html.A(f'{item}', target="_blank", href=_DEFAULT_URL_PATH_NAME + item)))
            elif key == "Volume":
                tmprow.append(html.Td(f'{item:.0f}'))
            elif key == "RS Rank":
                tmprow.append(html.Td(f'{item:.4f}'))
            elif key[0:2] == "52":
                tmprow.append(html.Td(f'{item:.2f}', className="tdfixsize"))
            elif key[0:6] == "Change" and item >= 0:
                tmprow.append(html.Td(f'+{item:.2f}', className="tdchangepos"))
            elif key[0:6] == "Change" and item < 0:
                tmprow.append(html.Td(f'{item:.2f}', className="tdchangeneg"))
            else:
                tmprow.append(html.Td(f'{item:.2f}'))

        rank = str(ind + 1)
        tmprow.insert(0, html.Td(rank))
        table.append(html.Tr(tmprow))

    return html.Table(table, className="trfixsize")


def _convert_str_list_column_to_float(in_str):
    """Convert the string column back into an array of float. Could use np.fromstring(mmmm.strip("[]"), sep=',')."""
    in_str = re.sub("[\[\]']", '', in_str)
    return np.array(in_str.split(',')).astype(float)


def _convert_str_list_column_to_str(in_str):
    """Convert the string column back into an array of str. Could use literal_eval"""
    in_str = re.sub("[\[\]']", '', in_str)
    return np.array(in_str.split(','))


def _get_dropdown_list_from_date_index(in_df):
    """Convert the date index of the input dataframe into a dict for the dropdown list."""
    df_index = in_df.index.to_numpy()
    return [{'label': str(date_index), 'value': str(date_index)} for date_index in df_index]


def _get_dropdown_list_from_ticker(in_df):
    """Convert the ticker column of the input dataframe into a dict for the dropdown list."""
    df_ticker = in_df['Ticker'].to_numpy()
    return [{'label': str(ind + 1) + '. ' + str(ticker), 'value': str(ticker)} for ind, ticker in enumerate(df_ticker)]


def get_ohlc_data(in_ticker):
    """Fetch OHLC data from csv, format the DataFrame and compute SMA."""
    db_dir_add = '../stock_vcpscreener/db_yfinance/'
    db_filename = in_ticker.strip().ljust(5, '_') + '.csv'
    ticker_data_df = pd.read_csv(db_dir_add + db_filename)
    ticker_data_df["Date"] = pd.to_datetime(ticker_data_df["Date"])
    ticker_data_df = ticker_data_df.set_index("Date")

    ticker_data_df["SMA_20"] = ticker_data_df["Adj Close"].rolling(window=20).mean()
    ticker_data_df["SMA_50"] = ticker_data_df["Adj Close"].rolling(window=50).mean()
    ticker_data_df["SMA_200"] = ticker_data_df["Adj Close"].rolling(window=200).mean()

    return ticker_data_df


def get_ohlc_data_web(in_ticker):
    """Fetch OHLC data from yahoo finance, format the DataFrame and compute SMA."""
    yf.pdr_override()
    curr_day = datetime.utcnow() - timedelta(hours=5)  # UTC -5, i.e. set to US NY timezone

    # Fetch a year's worth of data
    ticker_data_df = pdr.get_data_yahoo(in_ticker.strip(),
                                        start=curr_day.date() - timedelta(days=365),
                                        end=curr_day.date(),
                                        threads=False)

    ticker_data_df["SMA_20"] = ticker_data_df["Adj Close"].rolling(window=20).mean()
    ticker_data_df["SMA_50"] = ticker_data_df["Adj Close"].rolling(window=50).mean()
    ticker_data_df["SMA_200"] = ticker_data_df["Adj Close"].rolling(window=200).mean()

    return ticker_data_df


def serve_layout():
    # Read the daily stock data
    daily_stock_file = f"{_STOCK_INFO_URL_PATH_NAME}daily_selected_stock_info.csv"
    global df
    df = pd.read_csv(daily_stock_file)
    df = df.set_index('Date')

    # Convert the string column into an object column
    df['Breadth Percentage'] = df['Breadth Percentage'].apply(_convert_str_list_column_to_float)
    df['Tickers that fit the conditions'] = df['Tickers that fit the conditions'].apply(_convert_str_list_column_to_str)
    df['RS rating of Tickers'] = df['RS rating of Tickers'].apply(_convert_str_list_column_to_float)
    df['RS rank of Tickers'] = df['RS rank of Tickers'].apply(_convert_str_list_column_to_float)

    # Read the corresponding info dataset of the most recent day
    selected_info_file = f'{_STOCK_INFO_URL_PATH_NAME}output/selected_stock_{df.index[-1]}.csv'
    df_info = pd.read_csv(selected_info_file)
    df_info = df_info.drop(df_info.columns[0], axis=1)

    # Reorder the columns, move the location of the change columns
    df_info = df_info[
        ['Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Change', 'Change (%)', 'Volume', '52 Week Min',
         '52 Week Max', 'RS Rating', 'RS Rank']
    ]

    # Get the list for drop list
    out_date_list = _get_dropdown_list_from_date_index(df)

    # Get the stock list of the most recent day for drop list
    out_stock_list = _get_dropdown_list_from_ticker(df_info)

    # Make a display copy
    global df_dis
    df_dis = pd.DataFrame([], index=df.index)
    df_dis['Number of stocks monitored'] = df['Number of stock']
    df_dis['Advanced / Declined stock'] = df['Advanced (Day)'].astype(str) + " / " + df['Declined (Day)'].astype(str)
    df_dis['AD Percent'] = (df['Advanced (Day)'] - df['Declined (Day)']) / (
                df['Advanced (Day)'] + df['Declined (Day)']) * 100
    df_dis['New 52W high / New 52W low'] = df['New High'].astype(str) + " / " + df['New Low'].astype(str)
    df_dis['Gauge (Billion $)'] = df['Gauge'] / 1e9
    df_dis['Percentage of stocks above its 20 Day SMA (SMA 20)'] = df['Stock above 20-DMA']
    df_dis['Percentage of stocks above its 50 Day SMA (SMA 50)'] = df['Stock above 50-DMA']
    df_dis['Percentage of stocks with SMA 20 > SMA 50'] = df['Stock with 20-DMA > 50-DMA']
    # df_dis['Percentage of stocks with 50 Day SMA > 200 Day SMA'] = df['Stock with 50-DMA > 200-DMA']
    df_dis['Percentage of stocks with SMA 50 > SMA 150 > SMA 200'] = df['Stock with 50 > 150 > 200-DMA']
    df_dis['Percentage of stocks trending 200 Day SMA'] = df['Stock with 200-DMA is rising']
    df_dis['Number of stocks that fit the criteria'] = df['Number of Stock that fit condition']
    df_dis['Percentage of stocks that fit the criteria'] = df['Number of Stock that fit condition(%)']

    up_layout = html.Div([
        # html.H4("Dashboard with Dash", style={'text-align': 'center'}, className="padded"),
        dcc.Dropdown(id="check_date",
                     options=out_date_list,
                     multi=False,
                     value=out_date_list[-1]["value"],
                     style={"width": "50%", "float": "right"}
                     ),
        html.Div(id="output_container", children=[], style={"margin-bottom": "20px"}, className="headtitle padded"),
        html.Div(
            [
                html.H6("What are we showing here?"),
                html.Br([]),
                dcc.Markdown('''
                The top stocks in the US market are selected based on multiple criteria applied to \
                the simple moving averages and price performance over the last year. \
                This report is generated based on the output of a custom US stock screener package 'stock_vcpscreener'. \
                The source code of the stock screener package and this dashboard can be found \
                [here](https://github.com/jeffreyrdcs/stock-vcpscreener) at my github. \
                The screener calculates various market breadth indicators and selects stocks on a daily basis based on \
                the criteria. To rank the selected stocks, a rating score is computed using past performances, similar \
                to the IBD RS rating. The rating and rank of the stock can be found in the summary table.''',
                             style={"color": "#ffffff"},
                             className="row",
                             ),
            ],
            className="s-summary",
            style={"margin-bottom": "15px"},
        ),

        # Row 1
        html.Div(
            [
                html.Div(
                    [
                        html.H6("Stock Ratings (Top 50 that fit the criteria)", className="subtitle padded"),
                        dcc.Graph(id="stock_bar",
                                  figure={},
                                  config={"displayModeBar": False, "responsive": True})
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
                            "Daily Market Breadth",
                            className="subtitle padded"
                        ),
                        dcc.Graph(id='breadth_hist', figure={}, config={"displayModeBar": False})
                    ],
                    className="six columns",
                ),
                html.Div(
                    [
                        html.H6(
                            "Daily Market Performance",
                            className="subtitle padded",
                        ),
                        html.Div(
                            make_performance_table(df_dis[df.index == df.index[-1]]),
                            id="daily_report",
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
                            "Stock Summary (Top 50 that fit the criteria)",
                            className="subtitle padded"
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            make_stock_info_table(df_info),
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
            _get_chart_divs(out_stock_list),
            className="row",
            style={"margin-bottom": "15px"},
        ),
    ], className="page")

    return up_layout


def _get_chart_divs(stock_list):
    """Return a list of Divs depend on the number of items in the stock list."""
    num_of_selected_stocks = len(stock_list)

    chart_divs = [
        html.Div(
            [
                html.H6(
                    "Charts",
                    className="subtitle padded"
                ),
            ],
            className="twelve columns",
        )
    ]

    for i in range(1, _NUM_OF_STOCK_CHART_TO_DISPLAY + 1):
        if i > num_of_selected_stocks - 1:
            chart_divs.append(
                html.Div(
                    _get_dropdown_list_and_chart(f"check_stock{i}", stock_list, stock_list[0]["value"],
                                                 f"stock_chart{i}"),
                    className="six columns",
                ),
            )
        else:
            chart_divs.append(
                html.Div(
                    _get_dropdown_list_and_chart(f"check_stock{i}", stock_list, stock_list[i]["value"],
                                                 f"stock_chart{i}"),
                    className="six columns",
                ),
            )

    return chart_divs


def _get_dropdown_list_and_chart(dropdown_id, dropdown_options, dropdown_value, graph_id):
    return [
               dcc.Dropdown(
                   id=dropdown_id,
                   options=dropdown_options,
                   multi=False,
                   value=dropdown_value,
                   style={"width": "50%"},
               ),
               dcc.Graph(
                   id=graph_id,
                   figure={},
                   config={"displayModeBar": False}),
           ]


# ------------------------------------------------------------------------------
# Page layout
app.layout = serve_layout

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

    df_match_date = df[df.index == in_check_date].copy()
    df_dis_match_date = df_dis[df_dis.index == in_check_date].copy()

    # For the daily breadth histogram
    df_match_date_histodata = df_match_date['Breadth Percentage'].iloc[0]
    histogram_range_to_plot = (df_match_date_histodata > -20) & (df_match_date_histodata < 20)

    # Read the corresponding info dataset of the selected date
    selected_info_file = f'{_STOCK_INFO_URL_PATH_NAME}output/selected_stock_{in_check_date}.csv'
    df_match_date_info = pd.read_csv(selected_info_file)
    df_match_date_info = df_match_date_info.drop(df_match_date_info.columns[0], axis=1)

    # Reorder the columns, move the location of the change columns
    df_match_date_info = df_match_date_info[
        ['Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Change', 'Change (%)', 'Volume', '52 Week Min',
         '52 Week Max', 'RS Rating', 'RS Rank']
    ]

    if df_match_date_info.shape[0] >= _MAX_NUM_OF_STOCK_TO_DISPLAY:
        num_of_stocks_to_display = _MAX_NUM_OF_STOCK_TO_DISPLAY
    else:
        num_of_stocks_to_display = df_match_date_info.shape[0]

    # Update text in the status container
    container = f"US Stock Market Analysis Report for {in_check_date}"

    # Update stock rating plot
    fig = px.bar(
        x=df_match_date['Tickers that fit the conditions'].iloc[0][0:num_of_stocks_to_display],
        y=df_match_date['RS rating of Tickers'].iloc[0][0:num_of_stocks_to_display],
        color_continuous_scale=px.colors.sequential.Greens_r[1:7],
        color=np.linspace(0, 255, num_of_stocks_to_display),
        orientation='v'
    )
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
    fig2 = px.histogram(x=df_match_date_histodata[histogram_range_to_plot], range_x=[-20, 20], nbins=100,
                        labels={"value": "Percentage Change (%)"},
                        color_discrete_sequence=['#009900'], title='')
    fig2.add_annotation(xref="x domain",
                        yref="y domain",
                        x=0.025,
                        y=0.975,
                        showarrow=False,
                        text=f"Net Breadth (AD Percent) = {df_dis_match_date['AD Percent'].values[0]:.2f}%"
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

    # Update daily performance table and stock info table
    table_daily = make_performance_table(df_dis_match_date)
    table_info = make_stock_info_table(df_match_date_info)

    # Update the check_stock1 and 2 dropdown list
    out_stock_list = _get_dropdown_list_from_ticker(df_match_date_info)

    # Get the default values for the stock charts
    out_stock_value1, out_stock_value2, out_stock_value3, out_stock_value4 = _get_default_stock_chart_values(
        out_stock_list, df_match_date_info
    )

    return container, fig2, table_daily, fig, table_info, \
           out_stock_list, out_stock_list, out_stock_list, out_stock_list, \
           out_stock_value1, out_stock_value2, out_stock_value3, out_stock_value4


def _get_default_stock_chart_values(stock_list, df):
    num_of_stocks = df.shape[0]

    output_list = []
    if num_of_stocks < _NUM_OF_STOCK_CHART_TO_DISPLAY:
        for stock in stock_list:
            output_list.append(stock["value"])

        while len(output_list) < _NUM_OF_STOCK_CHART_TO_DISPLAY:
            output_list.append(stock_list[0]["value"])

        return output_list
    else:
        return [stock_list[i]["value"] for i in range(_NUM_OF_STOCK_CHART_TO_DISPLAY)]


# Callbacks to update the stock OHLC charts. Made them individual function so that we can change the input for them.
@app.callback(
    Output(component_id='stock_chart1', component_property='figure'),
    [Input(component_id='check_stock1', component_property='value'),
     Input(component_id='check_date', component_property='value')]
)
def display_stock_graph1(in_ticker, in_date):
    """Currently OHLC data is fetched online."""
    stock_df = get_ohlc_data_web(in_ticker)
    # stock_df = get_ohlc_data(in_ticker)

    return _get_stock_graph(stock_df, in_ticker, in_date)


@app.callback(
    Output(component_id='stock_chart2', component_property='figure'),
    [Input(component_id='check_stock2', component_property='value'),
     Input(component_id='check_date', component_property='value')]
)
def display_stock_graph2(in_ticker, in_date):
    stock_df = get_ohlc_data_web(in_ticker)

    return _get_stock_graph(stock_df, in_ticker, in_date)


@app.callback(
    Output(component_id='stock_chart3', component_property='figure'),
    [Input(component_id='check_stock3', component_property='value'),
     Input(component_id='check_date', component_property='value')]
)
def display_stock_graph3(in_ticker, in_date):
    stock_df = get_ohlc_data_web(in_ticker)
    
    return _get_stock_graph(stock_df, in_ticker, in_date)


@app.callback(
    Output(component_id='stock_chart4', component_property='figure'),
    [Input(component_id='check_stock4', component_property='value'),
     Input(component_id='check_date', component_property='value')]
)
def display_stock_graph4(in_ticker, in_date):
    stock_df = get_ohlc_data_web(in_ticker)

    return _get_stock_graph(stock_df, in_ticker, in_date)


def _get_stock_graph(stock_df, in_ticker, in_date):
    fig = go.Figure(
        data=[
            go.Ohlc(
                x=stock_df.index,
                open=stock_df["Open"],
                high=stock_df["High"],
                low=stock_df["Low"],
                close=stock_df["Close"],
                name=in_ticker,
            ),
            go.Scatter(x=stock_df.index, y=stock_df["SMA_20"], line=dict(color="orange", width=1), name="SMA 20"),
            go.Scatter(x=stock_df.index, y=stock_df["SMA_50"], line=dict(color="green", width=1), name="SMA 50"),
            go.Scatter(x=stock_df.index, y=stock_df["SMA_200"], line=dict(color="darkblue", width=1), name="SMA 200"),
        ]
    )

    fig.update_layout(
        showlegend=False, autosize=False, hovermode="x",
        height=370, width=465,
        font_family="Arial",
        margin={
            "r": 0,
            "t": 10,
            "b": 20,
            "l": 0,
        },
        plot_bgcolor="rgba(250,250,250,1)",
        xaxis_title="", yaxis_title="",
        title_x=0.5, title_y=1.0,
    )
    fig.add_vline(x=in_date, line_width=2, line_dash="dash", line_color="rgba(227,227,227,0.75)")

    return fig

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)  # debug=True
