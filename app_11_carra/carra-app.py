import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import yfinance as yf
import sqlite3
from datetime import datetime
from datetime import timedelta
from collections import OrderedDict

st.title('CARRA throughput')

st.markdown("""
This app plots the CARRA data
* Data source: SQLite file from CARRA
""")

st.sidebar.header('Select stream')

# Read the sqlite file
def load_data(ifile="/home/cap/data/from_ecmwf/dbases_harmon/carra_daily_logs.db",tname="daily_logs"):
    conn=sqlite3.connect(ifile)
    sql_comm = "SELECT * FROM "+tname
    df = pd.read_sql(sql_comm, conn)
    return df

df = load_data()
sector = df.groupby('stream')

# Sidebar - Sector selection
sorted_sector_unique = sorted( df['stream'].unique() )
selected_stream = st.sidebar.multiselect('Stream', sorted_sector_unique, sorted_sector_unique)


# Filtering data
df_selected_stream = df[ (df['stream'].isin(selected_stream)) ]

st.header('Display streams')
st.write('Data Dimension: ' + str(df_selected_stream.shape[0]) + ' rows and ' + str(df_selected_stream.shape[1]) + ' columns.')
st.dataframe(df_selected_stream[["stream","yyyymmdd","simdate"]])


def read_stream(stream,year,data):
    '''
    Count days present for all days,
    starting at the beginning of each year
    Return a data frame with dates and number of days produced on each day
    '''
    if stream == "carra_pan":
        startDate= datetime(2021,5,10) #Start of the counting
        endDate=datetime.today()
    elif stream != "carra_pan" and year == 2021:
        startDate= datetime(year,1,1) #Start of the counting
        endDate=datetime.today()
    elif stream != "carra_pan" and year != 2021:
        startDate= datetime(year,1,1) #Start of the counting
        endDate= datetime(year,12,31)
    #Set array of all dates I want to searh for
    dates = []
    dates = [startDate + timedelta(days=x) for x in range(0, (endDate-startDate).days+1)]
    #Temporary dict to collect all dates
    collect_data=OrderedDict()
    collect_data["Ndays"]=[]
    collect_data["Dates"] = []

    if stream == "All": #Go through all streams, ignore stream name in selection
        for date in dates:
            this_date = datetime.strftime(date,"%Y/%m/%d")
            count = data[data.yyyymmdd == this_date].drop_duplicates(subset="simdate",keep="last").shape[0]
            collect_data["Dates"].append(date)
            collect_data["Ndays"].append(count)
    elif "," in stream: #This is a list of streams. Go through selected streams
        for date in dates:
            this_date = datetime.strftime(date,"%Y/%m/%d")
            for this_stream in stream.split(","):
                count = data[(data.yyyymmdd == this_date) & (data.stream == this_stream)].drop_duplicates(subset="simdate",keep="last").shape[0]
                collect_data["Ndays"].append(count)
                collect_data["Dates"].append(date)

    else: #Consider the stream name in counting
        for date in dates:
            this_date = datetime.strftime(date,"%Y/%m/%d")
            count = data[(data.yyyymmdd == this_date) & (data.stream == stream)].drop_duplicates(subset="simdate",keep="last").shape[0]
            collect_data["Ndays"].append(count)
            collect_data["Dates"].append(date)
    data_count = pd.DataFrame({"Dates": collect_data["Dates"],
                                "Ndays": collect_data["Ndays"]})
    data_count = data_count.astype({"Ndays": int})
    return data_count
def plot_domain(data,stream,date_range):
    fig, ax = plt.subplots()
    #print(f"Checking stream {stream}")
    #figName = args.output_file
    #fig = plt.figure()
    dates=data['Dates'].tolist()
    #ax = plt.subplot(111)
    ax.bar(data['Dates'].values,data['Ndays'].values,edgecolor=['k']*len(dates))
    plot_ini = date_range[0]#,'%Y/%m/%d')
    plot_end = date_range[1]#,'%Y/%m/%d')
    plt.xticks(rotation=40)
    xfmt = md.DateFormatter('%Y%m%d')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlim([plot_ini, plot_end])
    #Make damn axis integer
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.xaxis_date()
    return st.pyplot(fig)



def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot()

#num_company = st.sidebar.slider('Number of Companies', 1, 5)
#num_streams = st.sidebar.slider('Number of streams', 1,10)
start_date = st.sidebar.date_input('start date', datetime(2019,3,26))
end_date = st.sidebar.date_input('end date', datetime.today())


if st.button('Show Plots'):
    st.header('Throughput')
    for stream in selected_stream: #df_selected_stream.stream:
        year = int(datetime.strftime(start_date,"%Y"))
        df_plot = read_stream(stream,year,df)
        date_range = [start_date,end_date]
        plot_domain(df_plot,stream,date_range)
