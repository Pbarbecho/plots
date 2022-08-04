import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d
import seaborn as sns



plt.rcParams.update({'font.size': 18})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)


def filter_by_distance(df):
    space = 100 # distance space in x axis
    max_distance = 600 # max distance of rrx message
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
    plt.xlim(-1, 6)
    plt.show()


    # Subplots DELAY with several columns
    ax = df_delay_merged.plot(position=1,x="Distance", color='b', y="mean_x", yerr="std_x",  hatch='xx',
                      label='3GPP C-V2X', kind="bar", edgecolor='black',width=0.4,)
    df_delay_merged.plot( position=0,x="Distance", color='r', y="mean_y",yerr="std_y",  hatch='oo',
                           label='IEEE802.11p', kind="bar", ax=ax, edgecolor='black',width=0.4,)
    plt.xticks(rotation=360)
    ax.set_ylabel('Latency [ms]')
    ax.set_xlabel('Distance [m]')
    plt.xlim(-1, 6)
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


def read_tripinfo_file_TL():
    "Read data coming from OMNET simulations and build dataframes"
    p_emissions = "/Users/Pablo/Desktop/RL/plots/energy/rl/tripinfo.xml.csv"
    o_emissions = "/Users/Pablo/Desktop/RL/plots/energy/norl/tripinfo.xml.csv"

    p_emissions_df = pd.read_csv(p_emissions)
    o_emissions_df = pd.read_csv(o_emissions)

    p_emissions_df = p_emissions_df.loc[p_emissions_df['tripinfo_id'] == 'leader']
    print(p_emissions_df.keys())
    p_emissions_df = p_emissions_df.filter(
        items=["emissions_NOx_abs","tripinfo_timeLoss", "tripinfo_vType", "tripinfo_routeLength", "tripinfo_duration",
               "tripinfo_waitingTime", "emissions_CO2_abs", "emissions_fuel_abs"])
    print('here')
    o_emissions_df = o_emissions_df.loc[o_emissions_df['tripinfo_id'] == 'leader']
    o_emissions_df = o_emissions_df.filter(
        items=["emissions_NOx_abs","tripinfo_timeLoss", "tripinfo_vType", "tripinfo_routeLength", "tripinfo_duration",
               "tripinfo_waitingTime", "emissions_CO2_abs", "emissions_fuel_abs"])

    return pd.merge(p_emissions_df, o_emissions_df, suffixes=('_p', '_o'), on='tripinfo_vType',
                         how='outer').replace(np.nan, 0).reset_index()



def emissions():
    "Read data coming from OMNET simulations and build dataframes"
    p_emissions = "/Users/Pablo/Desktop/RESULT/predict/emission.xml.csv"
    o_emissions = "/Users/Pablo/Desktop/RESULT/original/emission.xml.csv"

    p_emissions_df = pd.read_csv(p_emissions)
    o_emissions_df = pd.read_csv(o_emissions)

    p_emissions_df = p_emissions_df.loc[p_emissions_df['vehicle_id']=='leader']
    p_emissions_df = p_emissions_df.filter(items=["vehicle_electricity","timestep_time", "vehicle_CO2", "vehicle_NOx", "vehicle_fuel", "vehicle_speed"])

    o_emissions_df = o_emissions_df.loc[o_emissions_df['vehicle_id'] == 'leader']
    o_emissions_df = o_emissions_df.filter(items=["vehicle_electricity","timestep_time", "vehicle_CO2", "vehicle_NOx", "vehicle_fuel", "vehicle_speed"])

    df_merged = pd.merge(p_emissions_df, o_emissions_df, suffixes=('_p', '_o'), on='timestep_time', how='outer').replace(np.nan, 0).reset_index()

    fig, axes = plt.subplots(figsize=(8, 5))

    # CO2
    df_merged['co2_o'] = df_merged['vehicle_CO2_o']/1000
    df_merged['co2_p'] = df_merged['vehicle_CO2_p'] / 1000
    df_merged.plot.line(subplots=True, grid=False,ax=axes,color='red',x='timestep_time',y='co2_o', label='No prediction')
    df_merged.plot.line(subplots=True, grid=False,ax=axes,color='green', x='timestep_time',y='co2_p',label='Prediction')

    # NOX
    #print(df_merged.keys())
    #df_merged.plot.line(subplots=True, grid=False,ax=axes,color='red',x='timestep_time',y='vehicle_NOx_o', label='No prediction')
    #df_merged.plot.line(subplots=True, grid=False,ax=axes,color='green', x='timestep_time',y='vehicle_NOx_p',label='Prediction')

    #axes.set_ylabel(r'NOx [mg/s]')
    axes.set_xlabel(r'Time [s]')
    axes.set_ylabel('CO2 [g/s]')
    plt.show()

    fig, axes = plt.subplots(figsize=(8, 5))
    df_merged.plot.line(subplots=True, grid=False, ax=axes, color='red', x='timestep_time', y='vehicle_speed_o', label='No prediction')
    df_merged.plot.line(subplots=True, grid=False, ax=axes, color='green', x='timestep_time', y='vehicle_speed_p',label='Prediction')
    axes.set_ylabel(r'Speed [m/s]')
    axes.set_xlabel(r'Time [s]')
    # axes.set_title('CO2 emissions')
    plt.show()

def triptime(df_merged):
    # Setting the positions and width for the bars
    pos = list(range(2))
    width = 0.2  # the width of a bar

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(8, 5))

    plt.bar(0, df_merged['tripinfo_duration_p'], width,
            alpha=1,
            color='green',
            hatch='x',  # this one defines the fill pattern
            label="Prediction",
            edgecolor='black')

    plt.bar(0.2, df_merged['tripinfo_duration_o'], width,
            alpha=1,
            color='red',
            hatch='o',
            label="No prediction",
            edgecolor='black')

    #ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    plt.bar(0.5, df_merged['tripinfo_timeLoss_p'], width,
            alpha=1,
            color='green',
            hatch='x',  # this one defines the fill pattern
            label="Prediction",
            edgecolor='black')

    plt.bar(0.7, df_merged['tripinfo_timeLoss_o'], width,
            alpha=1,
            color='red',
            hatch='o',
            label="No prediction",
            edgecolor='black')

    plt.bar(1, df_merged['tripinfo_waitingTime_p'] , width,
            alpha=1,
            color='green',
            hatch='x',  # this one defines the fill pattern
            label="Prediction",
            edgecolor='black')

    plt.bar(1.2, df_merged['tripinfo_waitingTime_o'], width,
            alpha=1,
            color='red',
            hatch='o',
            label="No prediction",
            edgecolor='black')
    ax.set_xticks( [0.1,0.6,1.1], ['Trip Duration', 'Timeloss','Waiting Time'])
    ax.set_ylabel(r'Time [s]')
    ax.legend(['Prediction', 'No Prediction'], loc='lower center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.show()


def emissions_summary(df_merged):
# Setting the positions and width for the bars
    pos = list(range(2))
    width = 0.2  # the width of a bar

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(8, 5))
    print(df_merged.keys())
    plt.bar(0, df_merged['emissions_NOx_abs_p'], width,
           alpha=1,
           color='green',
           hatch='x',  # this one defines the fill pattern
           label="Prediction",
           edgecolor='black')

    plt.bar(0.2, df_merged['emissions_NOx_abs_o'], width,
               alpha=1,
               color='red',
               hatch='o',
               label="No prediction",
               edgecolor='black')


    plt.bar(0.5, df_merged['emissions_CO2_abs_p']/1000, width,
               alpha=1,
               color='green',
               hatch='x',  # this one defines the fill pattern
               label="Prediction",
               edgecolor='black')

    plt.bar(0.7, df_merged['emissions_CO2_abs_o']/1000, width,
               alpha=1,
               color='red',
               hatch='o',
               label="No prediction",
               edgecolor='black')
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    plt.bar(1, df_merged['emissions_fuel_abs_p'] , width,
               alpha=1,
               color='green',
               hatch='x',  # this one defines the fill pattern
               label="Prediction",
               edgecolor='black')

    plt.bar(1.2, df_merged['emissions_fuel_abs_o'] , width,
               alpha=1,
               color='red',
               hatch='o',
               label="No prediction",
               edgecolor='black')

    ax.set_xticks( [0.1,0.6,1.1], ['NOx', 'CO2','Fuel'])
    ax.set_ylabel(r'Emissions [ug]')
    ax2.set_ylabel(r'Fuel consumption [ml]')
    ax.legend(['Prediction', 'No Prediction'], loc='lower center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.show()

def ev(df_merged):
    # Setting the positions and width for the bars
    pos = list(range(2))
    width = 0.2  # the width of a bar

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(8, 5))
    print(df_merged.keys())
    plt.bar(0, df_merged['vehicle_electricity_p'], width,
            alpha=1,
            color='green',
            hatch='x',  # this one defines the fill pattern
            label="Prediction",
            edgecolor='black')

    plt.bar(0.2, df_merged['vehicle_electricity_o'], width,
            alpha=1,
            color='red',
            hatch='o',
            label="No prediction",
            edgecolor='black')


    ax.set_xticks( [0.1,0.6,1.1], ['Trip Duration', 'Timeloss','Waiting Time'])
    ax.set_ylabel(r'Time [s]')
    ax.legend(['Prediction', 'No Prediction'], loc='lower center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.show()


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.0fk' % (x*1e-3)



def plot_train_logs():
    #path_log_csv = "/Users/Pablo/Desktop/RL/plots/logs/train_value_loss_150k.csv"

    path_log_csv = "/Users/Pablo/Desktop/RL/plots/model/plots/train_value_loss.csv"
    value_loss_df = pd.read_csv(path_log_csv)
    formatter = FuncFormatter(millions)

    fig, axes = plt.subplots(figsize=(6, 4))
    value_loss_df.plot.line(subplots=True, grid=False, ax=axes, color='green', x='Step', y='Value')
    axes.get_legend().remove()
    axes.xaxis.set_major_formatter(formatter)
    axes.set_ylabel(r'Value Loss')
    axes.set_xlabel(r'Training Steps')
    fig.tight_layout()
    plt.show()

    ev_path_log_csv = "/Users/Pablo/Desktop/RL/plots/model/plots/train_explained_variance.csv"
    ev_df = pd.read_csv(ev_path_log_csv)

    fig, axes = plt.subplots(figsize=(6, 4))
    ev_df.plot.line(subplots=True, grid=False, ax=axes, color='green', x='Step', y='Value')
    axes.get_legend().remove()
    axes.xaxis.set_major_formatter(formatter)
    axes.set_ylabel(r'Explained Variance')
    axes.set_xlabel(r'Training Steps')
    fig.tight_layout()
    plt.show()


def plot_distance():
    rl_path_file = "/Users/Pablo/Desktop/RL/plots/distance/rl.csv"
    rl_distance_df = pd.read_csv(rl_path_file, names=['Distance', 'Time'])

    no_path_file = "/Users/Pablo/Desktop/RL/plots/distance/norl.csv"
    no_distance_df = pd.read_csv(no_path_file, names=['Distance', 'Time'])

    #df_merged = pd.merge(rl_path_file, o_emissions_df, suffixes=('_p', '_o'), on='timestep_time', how='outer').replace(np.nan, 0).reset_index()

    fig, axes = plt.subplots(figsize=(8, 5))
    no_distance_df.plot.line(subplots=True, grid=False,ax=axes,color='red',x='Time',y='Distance', label='No prediction')
    rl_distance_df.plot.line(subplots=True, grid=False,ax=axes,color='green', x='Time',y='Distance',label='Prediction')

    axes.set_ylabel(r'Distance to TL [m]')
    axes.set_xlabel(r'Time [s]')
    #axes.set_title('CO2 emissions')

    plt.show()


def plot_reward():
    path_10k_file = "/Users/Pablo/Desktop/RL/plots/model/10k/reward.csv"
    df_10k = pd.read_csv(path_10k_file, names=['Reward', 'Penalty', 'Time'])

    path_100k_file = "/Users/Pablo/Desktop/RL/plots/model/100k/reward.csv"
    df_100k = pd.read_csv(path_100k_file, names=['Reward', 'Penalty', 'Time'])
    """
    fig, axes = plt.subplots(figsize=(8, 5))
    df_10k.plot.line(subplots=True, grid=False, ax=axes, color='red', x='Time', y='Reward',
                             label='No prediction')
    df_100k.plot.line(subplots=True, grid=False, ax=axes, color='green', x='Time', y='Reward',
                     label='prediction')


    axes.set_ylabel(r'PPO Max Reward')
    axes.set_xlabel(r'Time [s]')
    # axes.set_title('CO2 emissions')
    #plt.show()
    """
    sns.lineplot(x="Time", y="Reward", data=df_100k)
    sns.lineplot(x="Time", y="Reward", data=df_10k)
    plt.show()

def plot_total_energy():

    path_energy_file_rl20 = "/Users/Pablo/Desktop/RL/plots/energy/rl/modelrl20.csv"
    energy_rl_df20 = pd.read_csv(path_energy_file_rl20)
    energy_rl_df20 = energy_rl_df20.filter(items=['vehicle_acceleration', 'timestep_time','vehicle_energyConsumed',  'vehicle_id', 'vehicle_maximumBatteryCapacity', 'vehicle_speed', 'vehicle_timeStopped', 'vehicle_energyCharged'])
    energy_rl_df20.dropna(inplace=True)

    path_energy_file_rl50 = "/Users/Pablo/Desktop/RL/plots/energy/rl/modelrl50.csv"
    energy_rl_df50 = pd.read_csv(path_energy_file_rl50)
    energy_rl_df50 = energy_rl_df50.filter(items=['vehicle_acceleration', 'timestep_time','vehicle_energyConsumed',  'vehicle_id', 'vehicle_maximumBatteryCapacity', 'vehicle_speed', 'vehicle_timeStopped', 'vehicle_energyCharged'])
    energy_rl_df50.dropna(inplace=True)

    path_energy_file_rl70 = "/Users/Pablo/Desktop/RL/plots/energy/rl/modelrl70.csv"
    energy_rl_df70 = pd.read_csv(path_energy_file_rl70)
    energy_rl_df70 = energy_rl_df70.filter(items=['vehicle_acceleration', 'timestep_time', 'vehicle_energyConsumed', 'vehicle_id', 'vehicle_maximumBatteryCapacity', 'vehicle_speed', 'vehicle_timeStopped', 'vehicle_energyCharged'])
    energy_rl_df70.dropna(inplace=True)

    path_energy_file_no_rl70 = "/Users/Pablo/Desktop/RL/plots/energy/norl/norl70.csv"
    energy_no_rl_df70 = pd.read_csv(path_energy_file_no_rl70)
    energy_no_rl_df70 = energy_no_rl_df70.filter(items=['vehicle_acceleration','timestep_time','vehicle_energyConsumed',  'vehicle_id', 'vehicle_maximumBatteryCapacity', 'vehicle_speed', 'vehicle_timeStopped', 'vehicle_energyCharged'])
    energy_no_rl_df70.dropna(inplace=True)

    path_energy_file_no_rl20 = "/Users/Pablo/Desktop/RL/plots/energy/norl/norl20.csv"
    energy_no_rl_df20 = pd.read_csv(path_energy_file_no_rl20)
    energy_no_rl_df20 = energy_no_rl_df20.filter(items=['vehicle_acceleration','timestep_time','vehicle_energyConsumed',  'vehicle_id', 'vehicle_maximumBatteryCapacity', 'vehicle_speed', 'vehicle_timeStopped', 'vehicle_energyCharged'])
    energy_no_rl_df20.dropna(inplace=True)

    path_energy_file_no_rl50 = "/Users/Pablo/Desktop/RL/plots/energy/norl/norl50.csv"
    energy_no_rl_df50 = pd.read_csv(path_energy_file_no_rl50)
    energy_no_rl_df50 = energy_no_rl_df50.filter(items=['vehicle_acceleration','timestep_time','vehicle_energyConsumed',  'vehicle_id', 'vehicle_maximumBatteryCapacity', 'vehicle_speed', 'vehicle_timeStopped', 'vehicle_energyCharged'])
    energy_no_rl_df50.dropna(inplace=True)

    df_merged = pd.merge(energy_rl_df50, energy_no_rl_df50, suffixes=('_p', '_o'), on='timestep_time',
                    how='outer').replace(np.nan, 0).reset_index()
    df_merged = df_merged.filter(items=['vehicle_timeStopped_p', 'vehicle_timeStopped_o','vehicle_energyConsumed_p','vehicle_energyConsumed_o','vehicle_speed_o','vehicle_speed_p'])


    df_merged2 = pd.merge(energy_rl_df70, energy_no_rl_df70, suffixes=('_p', '_o'), on='timestep_time',
                         how='outer').replace(np.nan, 0).reset_index()
    df_merged2 = df_merged2.filter(items=['vehicle_timeStopped_p', 'vehicle_timeStopped_o', 'vehicle_energyConsumed_p', 'vehicle_energyConsumed_o',
               'vehicle_speed_o', 'vehicle_speed_p'])

    df_merged3 = pd.merge(energy_rl_df20, energy_no_rl_df20, suffixes=('_p', '_o'), on='timestep_time',
                          how='outer').replace(np.nan, 0).reset_index()
    df_merged3 = df_merged3.filter(
        items=['vehicle_timeStopped_p', 'vehicle_timeStopped_o', 'vehicle_energyConsumed_p', 'vehicle_energyConsumed_o',
               'vehicle_speed_o', 'vehicle_speed_p'])

    df_merged['vehicle_speed_o_km'] =  df_merged['vehicle_speed_o'] * 3.6
    df_merged['vehicle_speed_p_km'] = df_merged['vehicle_speed_p'] * 3.6

    df_merged2['vehicle_speed_o_km'] = df_merged2['vehicle_speed_o'] * 3.6
    df_merged2['vehicle_speed_p_km'] = df_merged2['vehicle_speed_p'] * 3.6

    df_merged3['vehicle_speed_o_km'] = df_merged3['vehicle_speed_o'] * 3.6
    df_merged3['vehicle_speed_p_km'] = df_merged3['vehicle_speed_p'] * 3.6

    num_rows = df_merged.shape[0]

    total_df =  df_merged.sum() #50
    total_df2 = df_merged2.sum() #70
    total_df3 = df_merged2.sum()  # 20

    df_merged.to_csv('/Users/Pablo/Desktop/RL/plots/energy/merged.csv')

    pos = list(range(2))
    width = 0.2  # the width of a bar

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(8, 5))
    print(df_merged.keys())
    #### predittion 20
    plt.bar(0, total_df3['vehicle_energyConsumed_p'], width,
            alpha=1,
            color='limegreen',
            hatch='x',  # this one defines the fill pattern
            label="Prediction",
            edgecolor='black')

    #### predittion 50
    plt.bar(0.2, total_df['vehicle_energyConsumed_p'], width,
            alpha=1,
            color='green',
            hatch='/',  # this one defines the fill pattern
            label="Prediction",
            edgecolor='black')
    #### predittion 70
    plt.bar(0.4, total_df2['vehicle_energyConsumed_p'], width,
            alpha=1,
            color='blue',
            hatch='|',
            #label="Prediction",
            edgecolor='black')

    #### original 20
    plt.bar(0.6, total_df3['vehicle_energyConsumed_o'], width,
            alpha=1,
            color='aqua',
            hatch='-',
            # label="Prediction",
            edgecolor='black')

    #### original 50
    plt.bar(0.8, total_df['vehicle_energyConsumed_o'], width,
            alpha=1,
            color='dodgerblue', hatch=".",
            #label="Prediction",
            edgecolor='black')
    #### original 70
    plt.bar(1, total_df2['vehicle_energyConsumed_o'], width,
            alpha=1,
            color='red',
            hatch='o',
            label="No prediction",
            edgecolor='black')


    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    #20
    plt.bar(1.4, total_df3['vehicle_speed_p_km']/num_rows , width,
            alpha=1,
            color='limegreen',
            hatch='x',  # this one defines the fill pattern
            edgecolor='black')

    #50
    plt.bar(1.6, total_df['vehicle_speed_p_km']/num_rows , width,
            alpha=1,
            color='green',
            hatch='/',  # this one defines the fill pattern
            label="Prediction",
            edgecolor='black')
    #70
    plt.bar(1.8, total_df2['vehicle_speed_p_km']/num_rows, width,
            alpha=1,
            color='blue',
            hatch='|',
            #label="Prediction",
            edgecolor='black')
    # 20
    plt.bar(2, total_df3['vehicle_speed_o_km'] / num_rows, width,
            alpha=1,
            color='aqua',
            hatch='-',
            label="No prediction",
            edgecolor='black')
    #50
    plt.bar(2.2, total_df['vehicle_speed_o_km']/num_rows , width,
            alpha=1,
            color='dodgerblue',
            hatch='.',
            label="No prediction",
            edgecolor='black')
    #70
    plt.bar(2.4, total_df2['vehicle_speed_o_km']/num_rows , width,
            alpha=1,
            color='red',
            hatch='o',
            label="No prediction",
            edgecolor='black')

    ax.set_xticks([0.0, 0.2, 0.4, 0.6], ['','', '', ''])
    ax.set_ylabel(r'Total energy consumption [Wh]')
    ax2.set_ylabel(r'Mean speed [km/h]')
    ax2.legend(['P - 20 km/h', 'P - 50 km/h', 'P - 70 km/h', 'NP - 20 km/h', 'NP - 50 km/h', 'NP - 70 km/h'], loc='lower right',
             fancybox=False, shadow=False)
    plt.show()


def plot_energy():
    path_energy_file_rl50 = "/Users/Pablo/Desktop/RL/plots/energy/rl/modelrl50.csv"
    energy_rl_df50 = pd.read_csv(path_energy_file_rl50)
    energy_rl_df50 = energy_rl_df50.filter(items=['vehicle_acceleration', 'timestep_time','vehicle_energyConsumed',  'vehicle_id', 'vehicle_maximumBatteryCapacity', 'vehicle_speed', 'vehicle_timeStopped'])
    energy_rl_df50.dropna(inplace=True)

    path_energy_file_rl70 = "/Users/Pablo/Desktop/RL/plots/energy/rl/modelrl70.csv"
    energy_rl_df70 = pd.read_csv(path_energy_file_rl70)
    energy_rl_df70 = energy_rl_df70.filter(items=['vehicle_acceleration', 'timestep_time','vehicle_energyConsumed',  'vehicle_id', 'vehicle_maximumBatteryCapacity', 'vehicle_speed', 'vehicle_timeStopped'])
    energy_rl_df70.dropna(inplace=True)


    path_energy_file_no_rl = "/Users/Pablo/Desktop/RL/plots/energy/norl/norl.csv"
    energy_no_rl_df = pd.read_csv(path_energy_file_no_rl)
    energy_no_rl_df = energy_no_rl_df.filter(items=['vehicle_acceleration','timestep_time','vehicle_energyConsumed',  'vehicle_id', 'vehicle_maximumBatteryCapacity', 'vehicle_speed', 'vehicle_timeStopped'])
    energy_no_rl_df.dropna(inplace=True)


    fig, axes = plt.subplots(figsize=(8, 5))
    energy_rl_df50.plot.line(subplots=True, grid=False,ax=axes,color='green',x='timestep_time',y='vehicle_speed', label='Prediction')

    energy_rl_df70.plot.line(subplots=True, grid=False, ax=axes, color='blue', x='timestep_time', y='vehicle_speed',
                             label='Prediction')

    energy_no_rl_df.plot.line(subplots=True, grid=False, ax=axes, color='red', x='timestep_time', y='vehicle_speed',
                           label='No Prediction')
    #energy_rl_df.plot.line(subplots=True, grid=False,ax=axes,color='red',x='timestep_time',y='vehicle_energyConsumed', label='Prediction')
    #rl_distance_df.plot.line(subplots=True, grid=False,ax=axes,color='green', x='Time',y='Distance',label='Prediction')

    #axes.set_ylabel(r'Energy consumption [Wh]')
    axes.set_ylabel(r'Speed [m/s]')
    axes.set_xlabel(r'Time [s]')
    #axes.set_title('CO2 emissions')
    plt.show()


def plot_summary_file(path_summary_file):
    # READ DATA
    df_summ_file = pd.read_csv(path_summary_file)
    df_summ_file = df_summ_file.filter(items=['step_time','step_running','step_waiting'])
    df_summ_file.dropna(inplace=True)

    # PLOT
    fig, axes = plt.subplots(figsize=(8, 5))
    df_summ_file.plot.line(subplots=True, grid=False,ax=axes,color='green',x='step_time',y='step_running', label='# vehicles')

    #energy_rl_df70.plot.line(subplots=True, grid=False, ax=axes, color='blue', x='timestep_time', y='vehicle_speed',
    #                         label='Prediction')

    #energy_no_rl_df.plot.line(subplots=True, grid=False, ax=axes, color='red', x='timestep_time', y='vehicle_speed',
    #                       label='No Prediction')

    #energy_rl_df.plot.line(subplots=True, grid=False,ax=axes,color='red',x='timestep_time',y='vehicle_energyConsumed', label='Prediction')
    #rl_distance_df.plot.line(subplots=True, grid=False,ax=axes,color='green', x='Time',y='Distance',label='Prediction')

    axes.set_ylabel(r'Running Vehicles')
    axes.set_xlabel(r'Time [s]')
    plt.show()


def plot_generic_fundamental_statistics(list_of_agg_dfs, y,factor,ylabel):
    # DATA
    a_25, a_50, a_75, a_100 = list_of_agg_dfs[0], list_of_agg_dfs[1], list_of_agg_dfs[2], list_of_agg_dfs[3]
    # PLOT TRIP TIME FOR DIFERENT DENSITIES
    width = 0.1
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [0.0, 0.2]
    plt.bar(x[0], np.mean(a_50[y] / factor), yerr=np.std(a_50[y] / factor), width=width,align='center', alpha=0.5, ecolor='black', capsize=10, color='red', hatch='o', label='Low Density',edgecolor='black')
    plt.bar(x[1], np.mean(a_50[y] / factor), yerr=np.std(a_50[y] / factor), width=width,align='center', alpha=0.5, ecolor='black', capsize=10, color='green', hatch='-', label='Peak Hour',edgecolor='black')
    ax.set_xticks(x, [])
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_tripinfo_file(list_dfs_tripinfo):
    # READ DATA
    filter_items = ["emissions_NOx_abs", "tripinfo_timeLoss", "tripinfo_vType", "tripinfo_routeLength", "tripinfo_duration",
                    "tripinfo_waitingTime", "emissions_CO2_abs", "emissions_fuel_abs", 'emissions_electricity_abs','tripinfo_vType','tripinfo_depart']
    list_of_agg_dfs = filter_function(list_dfs_tripinfo, filter_items)

    # DATA
    a_25, a_50, a_75, a_100 = list_of_agg_dfs[0], list_of_agg_dfs[1], list_of_agg_dfs[2], list_of_agg_dfs[3]

    # TODOS LOS PLOTS SON DE 50% PENETRATION
    #plot trip time [min]
    plot_generic_fundamental_statistics(list_of_agg_dfs,'tripinfo_duration',60,'Mean Trip Duration [min]')
    # plot route length [km]
    plot_generic_fundamental_statistics(list_of_agg_dfs, 'tripinfo_routeLength', 1000, 'Mean Router Length [km]')


def filter_function(list_of_df, filter_list):
    # funcion que me permite pasar una lista de elementos a filtrar en una lista de dfs
    list_filtered_df = []
    for df in list_of_df:
        list_filtered_df.append(df.filter(items=filter_list))
    return list_filtered_df


def count_total_vehicles_emissions(list_df_emissions):
    # filtramos emissions file
    filter_items = ['timestep_time', 'vehicle_id', 'vehicle_type']
    list_of_filtered_dfs = filter_function(list_df_emissions, filter_items)

    # total vehicles list [25,50,75,100]
    list_total_vehicles = []
    list_total_ev_vehicles = []
    list_total_gas_vehicles = []

    for df in list_of_filtered_dfs:
        list_total_vehicles.append(df['vehicle_id'].nunique())
        # filter evs
        df_ev = df[df['vehicle_type'] == 'ev']
        list_total_ev_vehicles.append(df_ev['vehicle_id'].nunique())
        # filter gas
        df_gas = df[df['vehicle_type'] == 'gas']
        list_total_gas_vehicles.append(df_gas['vehicle_id'].nunique())

    print(list_total_vehicles,list_total_ev_vehicles,list_total_gas_vehicles)

    # PLOT
    # Setting the positions and width for the bars
    width = 0.1  # the width of a bar

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(9, 6))

    plt.bar(0.0, list_total_gas_vehicles[0],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='gray',
            hatch='-',  # this one defines the fill pattern
            label="25% EV's Penetration",
            edgecolor='black')
    plt.bar(0.1, list_total_gas_vehicles[1],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='red',
            hatch='o',  # this one defines the fill pattern
            label="50% EV's Penetration",
            edgecolor='black')

    plt.bar(0.2, list_total_gas_vehicles[2],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='blue',
            hatch='x',  # this one defines the fill pattern
            label="75% EV's Penetration",
            edgecolor='black')

    plt.bar(0.3, list_total_gas_vehicles[3],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='green',
            hatch='|',  # this one defines the fill pattern
            label="100% EV's Penetration",
            edgecolor='black')

    plt.bar(0.5, list_total_ev_vehicles[0],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='gray',
            hatch='-',  # this one defines the fill pattern
            #label="25% EV's Penetration",
            edgecolor='black')
    plt.bar(0.6, list_total_ev_vehicles[1],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='red',
            hatch='o',  # this one defines the fill pattern
            #label="50% EV's Penetration",
            edgecolor='black')

    plt.bar(0.7, list_total_ev_vehicles[2],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='blue',
            hatch='x',  # this one defines the fill pattern
            #label="75% EV's Penetration",
            edgecolor='black')

    plt.bar(0.8, list_total_ev_vehicles[3],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='green',
            hatch='|',  # this one defines the fill pattern
            #label="100% EV's Penetration",
            edgecolor='black')

    plt.bar(1, list_total_vehicles[0],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='gray',
            hatch='-',  # this one defines the fill pattern
            #label="25% EV's Penetration",
            edgecolor='black')
    plt.bar(1.1, list_total_vehicles[1],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='red',
            hatch='o',  # this one defines the fill pattern
            #label="50% EV's Penetration",
            edgecolor='black')

    plt.bar(1.2, list_total_vehicles[2],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='blue',
            hatch='x',  # this one defines the fill pattern
            #label="75% EV's Penetration",
            edgecolor='black')

    plt.bar(1.3, list_total_vehicles[3],
            width=width,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color='green',
            hatch='|',  # this one defines the fill pattern
            #label="100% EV's Penetration",
            edgecolor='black')

    ax.set_xticks( [0.15,0.65, 1.15], ['GAS', 'EVs', 'Total Vehicles'])
    ax.set_ylabel(r'# of vehicles')
    ax.legend(loc='lower center', bbox_to_anchor = (0.5, 1.0), ncol=2, fancybox=True, shadow=False)
    #plt.tight_layout()
    plt.show()


def count_vehicles_emissions(path_emissions_file):
    # READ DATA
    df_emissions_file = pd.read_csv(path_emissions_file)
    filter_items = ['timestep_time','vehicle_id','vehicle_type']
    df_emissions_file = df_emissions_file.filter(items=filter_items)
    df_emissions_file.dropna(inplace=True)
    df_total = df_emissions_file.groupby('timestep_time', as_index=False).count()

    # fitral por type of vehicle
    df_ev = df_emissions_file[df_emissions_file['vehicle_type'] == 'ev']
    df_ev = df_ev.groupby('timestep_time', as_index=False).count()

    # fitral por type of vehicle
    df_gas = df_emissions_file[df_emissions_file['vehicle_type'] == 'gas']
    df_gas = df_gas.groupby('timestep_time', as_index=False).count()

    # PLOT  fuel
    fig, axes = plt.subplots(figsize=(8, 5))
    df_ev.plot.line(subplots=True, grid=False, ax=axes, color='green', x='timestep_time', y='vehicle_id',
                        label='EV')
    df_gas.plot.line(subplots=True, grid=False, ax=axes, color='red', x='timestep_time', y='vehicle_id',
                    label='Gas')
    df_total.plot.line(subplots=True, grid=False, ax=axes, color='blue', x='timestep_time', y='vehicle_id',
                     label='Total')
    axes.set_ylabel(r'# vehicles')
    axes.set_xlabel(r'Time [s]')
    plt.show()


def plots_instantaneous_emissions_generic(list_of_agg_dfs,x,y,ylabel):
    #DATA
    a_25, a_50, a_75, a_100 = list_of_agg_dfs[0], list_of_agg_dfs[1], list_of_agg_dfs[2], list_of_agg_dfs[3]
    l25, l50, l75, l100 = "25% EV's Penetration", "50% EV's Penetration", "75% EV's Penetration", "100% EV's Penetration"
    #PLOT
    fig, axes = plt.subplots(figsize=(7, 5))
    #a_25.plot.line(subplots=True, marker='o', markerfacecolor='gray', markersize=6, grid=False, ax=axes, color='gray', x=x, y=y, label=l25)
    a_25.plot.line(subplots=True, grid=False, ax=axes, color='gray',x=x, y=y, label=l25)
    a_50.plot.line(subplots=True, grid=False, ax=axes, color='red', x=x, y=y, label=l50)
    a_75.plot.line(subplots=True, grid=False, ax=axes, color='blue', x=x, y=y, label=l75)
    a_100.plot.line(subplots=True,grid=False, ax=axes, color='green', x=x, y=y, label=l100)
    axes.set_ylabel(ylabel)
    axes.set_xlabel(r'Time [min]')
    plt.show()


def plots_total_emissions_generic(list_of_agg_dfs,y,ylabel,factor):
    #DATA
    a_25, a_50, a_75, a_100 = list_of_agg_dfs[0], list_of_agg_dfs[1], list_of_agg_dfs[2], list_of_agg_dfs[3]
    l25, l50, l75, l100 = "25% EV's Penetration", "50% EV's Penetration", "75% EV's Penetration", "100% EV's Penetration"
    #PLOT
    width=0.1

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(7, 5))
    x = [0.0,0.1,0.2,0.3]
    plt.bar(x[0],a_25[y].sum()/factor,width=width,align='center',alpha=0.5,ecolor='black',capsize=10,color='gray',hatch='-',label=l25,edgecolor='black')
    plt.bar(x[1],a_50[y].sum()/factor,width=width,align='center',alpha=0.5,ecolor='black',capsize=10,color='red',hatch='o', label=l50, edgecolor='black')
    plt.bar(x[2],a_75[y].sum()/factor,width=width,align='center',alpha=0.5,ecolor='black',capsize=10,color='blue',hatch='x', label=l75, edgecolor='black')
    plt.bar(x[3],a_100[y].sum()/factor,width=width,align='center',alpha=0.5,ecolor='black',capsize=10,color='green',hatch='|', label=l100, edgecolor='black')
    ax.set_xticks(x,[])
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_emissions_file(list_df_emissions):
    # filtramos emissions file
    filter_items = ['timestep_time',"vehicle_CO2","vehicle_NOx",'vehicle_fuel','vehicle_eclass','vehicle_electricity','vehicle_speed']
    list_of_filtered_dfs = filter_function(list_df_emissions, filter_items)

    # total vehicles list [25,50,75,100]
    list_of_agg_dfs = []
    list_of_agg_dfs_total = []

    for df in list_of_filtered_dfs:
        # agrupar por timestep y aggregar por suma
        df_emissions_aggr = df.groupby('timestep_time', as_index=False).sum()
        # CO2 en [g]
        df_emissions_aggr['vehicle_CO2_g'] = df_emissions_aggr['vehicle_CO2']/1000
        # convertimos de mg a ml  2[mg] = 1[ml]
        # convertir mg a ml divido para 2 y de ml a l /1000
        df_emissions_aggr['vehicle_fuel_l'] = (df_emissions_aggr['vehicle_fuel']/2)/1000
        # sec to min
        df_emissions_aggr['time_min'] = df_emissions_aggr['timestep_time'] / 60
        # Wh/s to kWh/s
        df_emissions_aggr['vehicle_electricity_kwh'] = df_emissions_aggr['vehicle_electricity'] / 1000

        # AGREGATE TOTAL
        list_of_agg_dfs_total.append(df_emissions_aggr)

        # Sampling
        rate = 30
        df_subset = df_emissions_aggr[::rate]
        # AGREGATE SAMPLING
        list_of_agg_dfs.append(df_subset)

    # Instantaneous PLOTS
    # PLOT  CO2[g]
    plots_instantaneous_emissions_generic(list_of_agg_dfs,'time_min','vehicle_CO2_g',r'CO$_{2}$ [g]')
    # PLOT  NOX [mg]
    plots_instantaneous_emissions_generic(list_of_agg_dfs, 'time_min', 'vehicle_NOx', 'NOx [mg]')
    # PLOT  fuel [kL]
    plots_instantaneous_emissions_generic(list_of_agg_dfs, 'time_min', 'vehicle_fuel_l', 'Fuel consumption [L]')
    # PLOT  energy consumption [kWh]
    plots_instantaneous_emissions_generic(list_of_agg_dfs, 'time_min', 'vehicle_electricity_kwh', 'Battery Energy Consumption [kWh]')

    # Totalized PLOTS
    # PLOT  CO2[t]
    plots_total_emissions_generic(list_of_agg_dfs_total,'vehicle_CO2_g','CO$_{2}$ [t]',1e6) #medimos en kg el total
    # PLOT  NOx[g]
    plots_total_emissions_generic(list_of_agg_dfs_total,'vehicle_NOx','NOx [t]',1e6) #medimos en kg el total
    # PLOT  CO2[g]
    plots_total_emissions_generic(list_of_agg_dfs_total,'vehicle_fuel_l','Fuel consumption [kL]',1e3) #medimos en kg el total
    # PLOT  energy [kWh]
    plots_total_emissions_generic(list_of_agg_dfs_total, 'vehicle_electricity_kwh', 'Battery Energy Consumption [MWh]',1e3)  # medimos en kg el total


def read_data(general_path):
    df_t_25 = pd.read_csv(os.path.join(general_path,'25', 'origin_destination_tripinfo_0.csv'))
    #df_s_25 = pd.read_csv(os.path.join(general_path,'25', 'origin_destination_summary_0.csv'))
    df_e_25 = pd.read_csv(os.path.join(general_path,'25', 'origin_destination_emission_0.csv'))

    df_t_50 = pd.read_csv(os.path.join(general_path, '50', 'origin_destination_tripinfo_0.csv'))
    #df_s_50 = pd.read_csv(os.path.join(general_path, '50', 'origin_destination_summary_0.csv'))
    df_e_50 = pd.read_csv(os.path.join(general_path, '50', 'origin_destination_emission_0.csv'))

    df_t_75 = pd.read_csv(os.path.join(general_path, '75', 'origin_destination_tripinfo_0.csv'))
    #df_s_75 = pd.read_csv(os.path.join(general_path, '75', 'origin_destination_summary_0.csv'))
    df_e_75 = pd.read_csv(os.path.join(general_path, '75', 'origin_destination_emission_0.csv'))

    df_t_100 = pd.read_csv(os.path.join(general_path, '100', 'origin_destination_tripinfo_0.csv'))
    #df_s_100 = pd.read_csv(os.path.join(general_path, '100', 'origin_destination_summary_0.csv'))
    df_e_100 = pd.read_csv(os.path.join(general_path, '100', 'origin_destination_emission_0.csv'))

    dict = {'tripinfo':[df_t_25,df_t_50,df_t_75,df_t_100],'emissions':[df_e_25,df_e_50,df_e_75,df_e_100],}
    return dict

#plot_total_energy()
#plot_energy()  # plot speed, energy consumption

#df_merged = read_tripinfo_file_TL()
#triptime(df_merged)

#emissions()
#df_merged = read_tripinfo_file_TL()
#triptime(df_merged)
#emissions_summary(df_merged)

#plot_train_logs()
#plot_reward()
#plot_distance()

# READ SUMO OUTPUT FILES
general_path = '/Users/Pablo/Dropbox/Mac/Documents/PROYECTO_SUMO/DATA/HD'
dict = read_data(general_path)

# Plot the number of running vehicles at each simulation time
#plot_summary_file(os.path.join(general_path,'origin_destination_summary_0.csv'))

# SI - Plot traffic statistics - FALTA OTRA DENSIDAD MAS ALTA PARA COMPARAR
#plot_tripinfo_file(dict['tripinfo'])

# SI - Plot time evolution emissions, fuel
#plot_emissions_file(dict['emissions'])

# NO - cuenta el numero de ev o gas vehicles durante el tiempo
#count_vehicles_emissions(os.path.join(general_path,'origin_destination_emission_0.csv'))

# SI - cuenta el numero total de ev o gas vehicles
#count_total_vehicles_emissions(dict['emissions'])
