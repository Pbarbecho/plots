import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
import matplotlib.ticker as mtick
plt.rcParams.update({'font.size': 18})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)


def filter_by_distance(df):
    space = 100 # distance space in x axis
    max_distance = 500 # max distance of rrx message
    iterations = max_distance / space
    distances_df = pd.DataFrame()
    for iter in range(int(iterations)):
        df_temp = df[df.DistanceToBS.between(left=iter * space, right=iter * space + space)]
        df_temp['Distance'] = iter * space + space
        df_temp['PDR'] = ( df_temp.Nodeid_x.count()/ df_temp.Nodeid_y.count() ) * 100
        distances_df = pd.concat([df_temp, distances_df], ignore_index=True)
    return distances_df

def read_data_dsrc():
    tx = "/Users/Pablo/Documentos/INFORME/dsrc/car_sender.csv"
    tx_columns = [ 'Nodeid','msgId','Creation_Time','Bit_Length','DistanceToBS','Tx_Time']
    df_tx = pd.read_csv(tx, names=tx_columns)


    rx = "/Users/Pablo/Documentos/INFORME/dsrc/server_receiver.csv"
    rx_columns = ['Nodeid','msgId','Creation_Time','Bit_Length','Rx_Time']
    df_rx = pd.read_csv(rx, names=rx_columns)
    df_rx['Delay'] = (df_rx['Rx_Time'] - df_rx['Creation_Time']) * 1000  # compute msg delay (ms)

    # Filter data
    tx_filter_metrics = ['Nodeid', 'msgId', 'DistanceToBS']
    tx_fitered_df = df_tx.filter(items=tx_filter_metrics)

    rx_filter_metrics = ['Nodeid', 'msgId', 'Delay']
    rx_fitered_df = df_rx.filter(items=rx_filter_metrics)

    df_merged = pd.merge(rx_fitered_df, tx_fitered_df, on='msgId', how='outer')

    # pdr
    df_pdr_filtered_dist = filter_by_distance(df_merged)
    df_pdr = filter_group(df_pdr_filtered_dist, 'PDR')

    # delay
    df_merged.dropna(inplace=True)
    df_delay_filtered_dist = filter_by_distance(df_merged)
    df_delay = filter_group(df_delay_filtered_dist, 'Delay')

    print('DSRC Mean Delay:', df_delay['mean'].mean())
    print('DSRC Mean PDR:', df_pdr['mean'].mean())
    print('DSRC STD PDR:', df_pdr['std'].mean())
    return df_pdr,df_delay


def read_data_cv2x():
    "Read data coming from OMNET simulations and build dataframes"
    tx = "/Users/Pablo/Documentos/INFORME/cv2x/car_sender.csv"
    tx_columns = [ 'Dir','Nodeid','SrcId','DstId','msgId','Creation_Time','N_Frame','FrameId','Dst_Port','Size','DistanceToBS']
    df_tx = pd.read_csv(tx, names=tx_columns)

    rx = "/Users/Pablo/Documentos/INFORME/cv2x/server_receiver.csv"
    rx_columns = ['Dir', 'Nodeid', 'TxId', 'RxId', 'msgId', 'o_Treeid', 'Creation_Time', 'Curr_Time', 'FrameId']
    df_rx = pd.read_csv(rx, names=rx_columns)
    df_rx['Delay'] = (df_rx['Curr_Time'] - df_rx['Creation_Time']) * 1000  # compute msg delay (ms)

    # Filter data
    tx_filter_metrics = ['Nodeid','msgId','DistanceToBS']
    tx_fitered_df = df_tx.filter(items=tx_filter_metrics)

    rx_filter_metrics = ['Nodeid','o_Treeid','Delay']
    rx_fitered_df = df_rx.filter(items=rx_filter_metrics)

    #tx_fitered_df.to_csv('/Users/Pablo/Desktop/INFORME/Filtered/tx_filtered.csv')
    df_merged = pd.merge(rx_fitered_df, tx_fitered_df, left_on='o_Treeid',right_on='msgId', how='outer')

    # pdr
    df_pdr_filtered_dist = filter_by_distance(df_merged)
    df_pdr = filter_group(df_pdr_filtered_dist, 'PDR')

    # delay
    df_merged.dropna(inplace=True)
    df_delay_filtered_dist = filter_by_distance(df_merged)
    df_delay = filter_group(df_delay_filtered_dist, 'Delay')
    print('CV2X Mean Delay:', df_delay['mean'].mean())
    print('CV2X Mean PDR:', df_pdr['mean'].mean())
    print('CV2X STD PDR:', df_pdr['std'].mean())
    return df_pdr,df_delay


def filter_group(df, name_filter):
    df = df.groupby("Distance").agg([np.mean, np.std])
    df = df[name_filter]
    return df


def plots():
    df_dsrc_pdr, df_dsrc_delay = read_data_cv2x()
    df_cv2x_pdr, df_cv2x_delay  = read_data_dsrc()
    df_pdr_merged = pd.merge(df_dsrc_pdr, df_cv2x_pdr, on='Distance', how='outer').replace(np.nan, 0).reset_index()
    df_delay_merged = pd.merge(df_dsrc_delay, df_cv2x_delay, on='Distance', how='outer').replace(np.nan, 0).reset_index()


    # Subplots PDR with several columns
    ax = df_pdr_merged.plot(position=1,x="Distance", color='b', y="mean_x", yerr="std_x",  hatch='xx',
                      label='3GPP C-V2X', kind="bar", edgecolor='black',width=0.4,)
    df_pdr_merged.plot( position=0,x="Distance", color='r', y="mean_y",yerr="std_y",  hatch='oo',
                           label='IEEE802.11p', kind="bar", ax=ax, edgecolor='black',width=0.4,)
    plt.xticks(rotation=360)
    ax.set_ylabel('PDR [%]')
    ax.set_xlabel('Distance [m]')
    plt.xlim(-1, 5)
    plt.show()


    # Subplots DELAY with several columns
    ax = df_delay_merged.plot(position=1,x="Distance", color='b', y="mean_x", yerr="std_x",  hatch='xx',
                      label='3GPP C-V2X', kind="bar", edgecolor='black',width=0.4,)
    df_delay_merged.plot( position=0,x="Distance", color='r', y="mean_y",yerr="std_y",  hatch='oo',
                           label='IEEE802.11p', kind="bar", ax=ax, edgecolor='black',width=0.4,)
    plt.xticks(rotation=360)
    ax.set_ylabel('Latency [ms]')
    ax.set_xlabel('Distance [m]')
    plt.xlim(-1, 5)
    plt.show()


def delay_plot(df):
    df = df.groupby("Distance").agg([np.mean, np.std])
    df_delay = df['Delay']


    df_delay.plot.bar(y="mean", yerr="std", color='w',hatch='x',
                   label='C-V2X', ax=ax, edgecolor='black')

    plt.xticks(rotation=360)
    ax.set_ylabel('Delay [ms]')
    ax.set_xlabel('Distance [m]')
    fig.tight_layout()
    plt.show()

"""
#para plotear varios plots 
# Subplots as having two types of quality
fig, ax = plt.subplots()

for key, group in df.groupby('quality'):
    group.plot('insert', 'mean', yerr='std',
               label=key, ax=ax)

plt.show()
"""

def prepare_data_to_plot(df):
    # prepare data
    df = df.rename(columns={'tripinfo_arrival': 'Arrived', 'tripinfo_duration':'Trip Time', 'tripinfo_routeLength':'Trip Length', 'emissions_CO2_abs':'Emissions'})
    metrics = ['Arrived', 'Trip Time', 'Trip Length', 'Emissions']
    df = df.filter(items=metrics)
    df['Trip Length'] = df['Trip Length']/1000
    df['Emissions'] = df['Emissions'] / 1000
    temp_dic = {}
    for m in metrics:
        if m == 'Arrived':
            temp_dic[m] = [df[m].count(), 0]
        elif m == 'Emissions':
            temp_dic[m] = [df[m].mean(), df[m].std()]
        else:
            temp_dic[m] = [df[m].mean(), df[m].std()]
    df = pd.DataFrame.from_dict(temp_dic).T.reset_index()
    df = df.rename(columns={'index':'Metric',0:'Mean', 1:'Std'})
    print(df)
    return df


def bar_plot_pattern(df):
    rt_data = build_plot_array(df, 'RandomTrips')
    dua_data = build_plot_array(df, 'DUARouter')
    ma_data = build_plot_array(df, 'MARouter')

    labels = ['1', '5','10', '15']

    # Setting the positions and width for the bars
    pos = list(range(len(rt_data)))
    width = 0.15  # the width of a bar

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(6, 4))

    plt.bar(pos, dua_data, width,
                   alpha=1,
                   color='w',
                   hatch='x',  # this one defines the fill pattern
                   label=labels[0], edgecolor='black')

    plt.bar([p + width for p in pos], rt_data, width,
            alpha=1,
            color='w',
            hatch='o',
            label=labels[1], edgecolor='black')

    plt.bar([p + width * 2 for p in pos], ma_data, width,
            alpha=0.5,
            color='k',
            hatch='',
            label=labels[2], edgecolor='black')


    # Setting axis labels and ticks
    #ax.set_ylabel(r'Number of vehicles')
    #ax.set_ylabel(r'Trip Length [km]')

    ax.set_ylabel(r'Trip Time [s]')

    #ax.set_xlabel(r"Scaling Factor")
    #ax.set_title('Grouped bar plot')
    ax.set_xticks([p + 1.5 * width for p in pos])
    ax.set_xticklabels(labels)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 5)
    plt.ylim([0, max(rt_data + dua_data + ma_data) * 1.5])

    # Adding the legend and showing the plot
    plt.legend(['DUARouter', 'RandomTrips', 'MARouter'], loc='upper right')
    plt.grid()
    plt.show()
    #fig.savefig('/Users/Pablo/Desktop/LUIS_BECA/PAPER STG/figures/scaling_factor_time.pdf')


def single_plot(folders, df):
    df = prepare_data_to_plot(df).T
    df = df.rename(columns=df.iloc[0])
    df = df.iloc[1:]
    plot_df = df.iloc[0:1]

    fig, axes = plt.subplots(4, figsize=(4,4))
    plot_df.plot(kind='barh',subplots=True, grid=False, sharex=False, ax=axes, legend=False)

    axes[0].errorbar(df.loc['Mean', 'Arrived'],0,xerr=df.loc['Std', 'Arrived'], fmt='o', color='Black', elinewidth=1, capthick=1,
                 errorevery=1, alpha=1, ms=4, capsize=5)
    axes[1].errorbar(df.loc['Mean', 'Trip Time'], 0, xerr=df.loc['Std', 'Trip Time'], fmt='o', color='Black', elinewidth=1, capthick=1,
                 errorevery=1, alpha=1, ms=4, capsize=5)
    axes[2].errorbar(df.loc['Mean', 'Trip Length'], 0, xerr=df.loc['Std', 'Trip Length'], fmt='o', color='Black', elinewidth=1, capthick=1,
                 errorevery=1, alpha=1, ms=4, capsize=5)
    axes[3].errorbar(df.loc['Mean', 'Emissions'], 0, xerr=df.loc['Std', 'Emissions'], fmt='o', color='Black', elinewidth=1, capthick=1,
                 errorevery=1, alpha=1, ms=4, capsize=5)

    axes[0].set_title(r'Number of arrived cars ')
    axes[1].set_title(r'Trip Time [s]')
    axes[2].set_title(r'Trip Length [Km]')
    axes[3].set_title(r'CO2 [g]')
    fig.tight_layout()
    # convert to html and save in html folder
    mpld3.save_html(fig, os.path.join(folders.html, 'plot.html'), template_type='simple')
    #fig_folder = "/Users/Pablo/Desktop/plot.pdf"
    #fig.savefig(fig_folder)


plots()
