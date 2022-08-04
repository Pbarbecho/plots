import os
import matplotlib.pyplot as plt  # noqa
import numpy as np
import pandas as pd
import seaborn as sns

parent_dir = '/Users/Pablo/Desktop/RL/statistics'
cpu_statistics = os.path.join(parent_dir, 'cpu')
memory_statistics = os.path.join(parent_dir, 'memory')
disk_statistics = os.path.join(parent_dir, 'disk')
results_dir = os.path.join(parent_dir, 'results')
plots_dir = os.path.join(parent_dir, 'plots')
begin = 0
end = 30


def prepare_memory_df(tool):
    if tool == 'SMO':
        tool_path = 'osm'
    elif tool == 'OMNeT++':
        tool_path = 'omnet'
    else:
        print('ERROOOOR !!!!!')

    str_file = os.path.join(memory_statistics, tool_path)
    memory_statistics_files = os.listdir(str_file)
    temp_df = pd.DataFrame()

    if 'DS_Store' in memory_statistics_files:
        memory_statistics_files.remove('.DS_Store') # mac problem

    for index_a, f in enumerate(memory_statistics_files):
        f_name = f.split('.')
        curr_file = os.path.join(str_file, f)
        file = pd.read_fwf(curr_file)

        # prepare dataframe
        #file.drop(file.tail(1).index, inplace=True)  # drop last n rows
        data = file.to_numpy()
        body = data[0:, :][0]  # filter first rows header

        header = file.keys()
        header = header[1:]

        new_body = []
        for e in body:
            if e != 'Mem:':
                new_body.append(e)

        # build dataframe
        df = pd.DataFrame(data=[new_body], columns=header)

        # replace , x . column
        df['tool'] = tool
        df['iteration'] = index_a
        df['file'] = f_name[0]
        #df['%memused'] = [ele.replace(',', '.') for ele in df['%memused']]
        #df['%commit'] = [ele.replace(',', '.') for ele in df['%commit']]
        # build df
        temp_df = temp_df.append(df, sort=True, ignore_index=True, verify_integrity=True)
    # data type
    elem = list(temp_df.keys())
    #elem.remove('Time')
    elem.remove('tool')
    temp_df[elem] = temp_df[elem].apply(pd.to_numeric)
    #temp_df['Time'] = temp_df['Time'].astype('datetime64[ns]').dt.time

    # filter
    memory = temp_df
    memory_data = memory.filter(['available', 'free', 'file', 'used','tool'], axis=1)
    # add column of real time
    sorted_df = memory_data.sort_values(by='file')
    realtime = [x for x in range(0, ((len(sorted_df.file)) * 2), 2)]
    sorted_df['RealTime'] = realtime

    sorted_df['percentage'] = (sorted_df['used'] / sorted_df['available']) * 100

    print(sorted_df)

    sorted_df.to_csv(os.path.join(results_dir, 'memory', '{}.csv'.format(tool)))


def prepare_cpu_df(tool):
    if tool == 'SMO':
        tool_path = 'osm'
    elif tool == 'OMNeT++':
        tool_path = 'omnet'
    else:
        print('ERROOOOR !!!!!')

    str_file = os.path.join(cpu_statistics, tool_path)
    cpu_statistics_files = os.listdir(str_file)
    #os.system('rm {}/{}/*.*'.format(cpu_statistics, tool))
    temp_df = pd.DataFrame()

    for index_a, f in enumerate(cpu_statistics_files):
        f_name = f.split('.')
        curr_file = os.path.join(str_file, f)

        file = pd.read_fwf(curr_file)

        # prepare dataframe
        file.drop(file.tail(1).index, inplace=True)  # drop last n rows
        data = file.to_numpy()

        body = data[0:, :]  # filter first 2 rows ALL CPUs

        #header = data[:1, :]
        #header = header[0]

        header = file.keys()
        new_header = []
        for index, i in enumerate(header):
            if index == 0:
                new_header.append('Time')
            else:
                new_header.append(i)

        # build dataframe
        df = pd.DataFrame(data=body, columns=new_header)

        # replace , x . column
        df['tool'] = tool
        df['iteration'] = index_a
        df['file'] = f_name[0]
        """
        df['%idle'] = [ele.replace(',', '.') for ele in df['%idle']]
        df['%iowait'] = [ele.replace(',', '.') for ele in df['%iowait']]
        df['%nice'] = [ele.replace(',', '.') for ele in df['%nice']]
        df['%steal'] = [ele.replace(',', '.') for ele in df['%steal']]
        df['%system'] = [ele.replace(',', '.') for ele in df['%system']]
        df['%user'] = [ele.replace(',', '.') for ele in df['%user']]
        """
        temp_df = temp_df.append(df, sort=True, ignore_index=True, verify_integrity=True)

    # data type
    elem = list(temp_df.keys())
    elem.remove('AM')
    elem.remove('Time')
    elem.remove('tool')
    elem.remove('CPU')
    temp_df[elem] = temp_df[elem].apply(pd.to_numeric)
    temp_df['Time'] = temp_df['Time'].astype('datetime64[ns]').dt.time
    #filter columns
    cpu = temp_df
    cpu_data = cpu.filter(['CPU', '%idle', 'iteration', 'tool', 'Time', 'file'], axis=1)
    filter_string = ['all']
    cpu_data = cpu_data[cpu_data.CPU.isin(filter_string)]
    # add column of real time
    sorted_df = cpu_data.sort_values(by='file')

    realtime = [x for x in range(0, (len(sorted_df.file)*2), 2)]
    sorted_df['RealTime'] = realtime
    sorted_df.to_csv(os.path.join(results_dir, 'cpu', '{}.csv'.format(tool)))


def prepare_disk_df(tool):
    if tool == 'SMO':
        tool_path = 'osm'
    elif tool == 'OMNeT++':
        tool_path = 'omnet'
    else:
        print('ERROOOOR !!!!!')

    str_file = os.path.join(disk_statistics, tool_path)
    disk_statistics_files = os.listdir(str_file)

    temp_df = pd.DataFrame()

    data_list = []
    for index_a, f in enumerate(disk_statistics_files):
        f_name = f.split('.')
        curr_file = os.path.join(str_file, f)
        file = pd.read_fwf(curr_file)

        # prepare dataframe
        disk = list(file.keys())
        data_list.append([disk[0], f_name[0], tool])

    new_header = ['size', 'file', 'tool']
    # build dataframe
    temp_df = pd.DataFrame(data=data_list, columns=new_header)

    # data type
    temp_df[['size', 'file']] = temp_df[['size', 'file']].apply(pd.to_numeric)

    # add column of real time
    sorted_df = temp_df.sort_values(by='file')
    realtime = [x for x in range(0, ((len(sorted_df.file)) * 2), 2)]
    sorted_df['RealTime'] = realtime
    sorted_df.to_csv(os.path.join(results_dir, 'disk', '{}.csv'.format(tool)))


def plot_cpu():
    cpu_dir = os.path.join(results_dir, 'cpu')
    result_files = os.listdir(cpu_dir)

    for f in result_files:
        name = f.split('.')

        if name[0] == 'SMO':
            df_osm = pd.read_csv(os.path.join(cpu_dir, f))
            print(os.path.join(cpu_dir, f))

        elif name[0] == 'OMNeT++':
            df_omnet = pd.read_csv(os.path.join(cpu_dir, f))
        else:
            print('ERROR TOOL NAME !!!!!!!!')

    df_osm = df_osm[df_osm.file >= begin]
    df_omnet = df_omnet[df_omnet.file >= begin]

    df_osm = df_osm[df_osm.file <= end]
    df_omnet = df_omnet[df_omnet.file <= end]

    fig, axes = plt.subplots(figsize=(4, 3))

    df_osm['Time'] = df_osm['RealTime']  # en minutos
    df_omnet['Time'] = df_omnet['RealTime']  # en minutos

    df_osm['idle'] = 100-df_osm['%idle'] # uso cpu no idle
    df_omnet['idle'] = 100 - df_omnet['%idle']  # uso cpu no idle


    df_osm.plot.line(subplots=True, grid=False, ax=axes, color='green', x='Time', y='idle',
                             label='Prediction')
    df_omnet.plot.line(subplots=True, grid=False, ax=axes, color='red', x='Time', y='idle',
                             label='No prediction')

    axes.set_ylabel(r'% of CPU usage')
    axes.set_xlabel(r'Time [s]')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_memory():
    memory_dir = os.path.join(results_dir, 'memory')
    result_files = os.listdir(memory_dir)

    for f in result_files:
        name = f.split('.')
        if name[0] == 'SMO':
            df_osm = pd.read_csv(os.path.join(memory_dir, f))
        elif name[0] == 'OMNeT++':
            df_omnet = pd.read_csv(os.path.join(memory_dir, f))
        else:
            print('ERROR TOOL NAME !!!!!!!!')

    df_osm = df_osm[df_osm.file >= begin]
    df_omnet = df_omnet[df_omnet.file >= begin]

    df_osm = df_osm[df_osm.file <= end]
    df_omnet = df_omnet[df_omnet.file <= end]

    data = pd.concat([df_osm, df_omnet], ignore_index=True)

    fig, axes = plt.subplots(figsize=(4, 3))

    df_osm['Time'] = df_osm['RealTime']  # en minutos
    df_omnet['Time'] = df_omnet['RealTime']  # en minutos

    df_omnet.plot.line(subplots=True, grid=False, ax=axes, color='green', x='Time', y='percentage',
                             label='Prediction')
    df_osm.plot.line(subplots=True, grid=False, ax=axes, color='red', x='Time', y='percentage',
                             label='No prediction')

    axes.set_ylabel(r'% of memory usage')
    axes.set_xlabel(r'Time [s]')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_disk():
    disk_dir = os.path.join(results_dir, 'disk')
    result_files = os.listdir(disk_dir)

    for f in result_files:
        name = f.split('.')
        if name[0] == 'SMO':
            df_osm = pd.read_csv(os.path.join(disk_dir, f))
        elif name[0] == 'OMNeT++':
            df_omnet = pd.read_csv(os.path.join(disk_dir, f))
        else:
            print('ERROR TOOL NAME !!!!!!!!')

    df_osm = df_osm[df_osm.file >= begin]
    df_omnet = df_omnet[df_omnet.file >= begin]

    df_osm = df_osm[df_osm.file <= end]
    df_omnet = df_omnet[df_omnet.file <= end]

    data = pd.concat([df_osm, df_omnet], ignore_index=True)

    # setup style
    sns.set(font_scale=1.2, rc={'text.usetex': True}, style="whitegrid",
            palette=sns.palplot(sns.color_palette('Paired')), color_codes=True)

    f, axes = plt.subplots(1, 1, figsize=(5, 5), sharex=False, sharey=False)
    sns.despine(left=False, top=False, right=False)

    # Line PLOT
    ###############################
    # 1 bit = 0.125 bytes
    f1 = sns.lineplot(data=data, x=data[r'RealTime']/60, y=(data['size']*1024)/1000000000, hue='tool')
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles=handles[1:], labels=labels[1:])

    # Axes names

    f1.set(ylabel=r'Disk usage~[GB]', xlabel=r'Time[min]')
    plt.tight_layout()
    f.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'disk.pdf'), bbox_inches="tight")


if __name__ == '__main__':
    tools = ['SMO', 'OMNeT++']
    for t in tools:
        prepare_cpu_df(t)
        #prepare_memory_df(t)
        #prepare_disk_df(t)

    plot_cpu()
    plot_memory()
    #plot_disk()
