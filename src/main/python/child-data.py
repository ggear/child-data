# -*- coding: utf-8 -*-

import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# TODO:
#   - Decide on size 30x30, 30x48, 40x60, 48x60, 60x60 print
#   - Only save variations at high res, scale to print size

DAYS_TO_INCLUDE = 475


def save_plot(child, data, path_output, background_colour, foreground_colour, label_colour, label_print, stats_print):
    figure = plt.figure(figsize=(2.67, 4.5))
    font = mfm.FontProperties(size=3, fname='/System/Library/Fonts/Apple Symbols.ttf')
    axes = figure.add_subplot(111, projection='polar')
    axes.set_facecolor(background_colour)
    axes.axes.get_xaxis().set_visible(False)
    axes.axes.get_yaxis().set_visible(False)
    axes.spines['polar'].set_visible(False)
    print('Plotting [{}] data ... '.format(data[0].shape[0]))

    for row in range(data[0].shape[0]):
        axes.bar(
            x=(-(data[0].loc[data[0].index[row], 'Minutes'] +
                 data[0].loc[data[0].index[row], 'Duration'] / 2) / 60 / 24 * 2 * np.pi) - np.pi / 2,
            height=1,
            width=data[0].loc[data[0].index[row], 'Duration'] / 60 / 24 * 2 * np.pi,
            bottom=data[0].loc[data[0].index[row], 'Radial'],
            color=foreground_colour)

    file_name = '{}_{}_{}_{}.png' \
        .format(path_output, background_colour, foreground_colour, 'None' if label_colour is None else label_colour)
    print('Saving figure [{}] ... '.format(file_name))

    if label_colour is not None:
        ticks = np.linspace(0, np.pi * 2, num=25)
        hours = ['6pm', '5pm', '4pm', '3pm', '2pm', '1pm', '12pm',
                 '11am', '10am', '9am', '8am', '$7am$', '6am',
                 '5am', '4am', '3am', '2am', '1am', '12am',
                 '11pm', '10pm', '9pm', '8pm', '$7pm$', '']
        hours = ['', '', '', '', '', '', '',
                 '', '', '', '', '7am', '',
                 '', '', '', '', '', '',
                 '', '', '', '', '7pm', '']
        for i in range(len(ticks)):
            if hours[i] != '':
                plt.plot((0, ticks[i]), (0, data[0]['Radial'].max() + 1), color=label_colour, linewidth=0.1, alpha=0.5)
            if label_print:
                plt.text(ticks[i], data[0]['Radial'].max() + 50, hours[i], ha='center', va='center',
                         color=label_colour, fontproperties=font)
        if stats_print:
            plt.text(4.25, 850, data[1], ha='left', va='center', color=label_colour, linespacing=1.8, fontproperties=font)

    plt.ylim(ymax=data[0]['Radial'].max() + 1)
    plt.savefig(file_name, facecolor=background_colour, dpi=900)
    plt.close('all')
    print('Released resources for figure\n')


def get_data(child, path_input, path_output, activity):
    df = pd.read_csv(path_input)
    df = df.loc[df['Activity'] == activity]
    df = df.set_index(pd.to_datetime(df['Date and Time'], format='%Y-%m-%d %H:%M:%S'))

    with pd.option_context(
            'display.max_rows', 100,
            'display.max_columns', None,
            'display.width', None):

        df['Date'] = df.index.date
        df['Time'] = df.index.time
        df['Duration'] = df['Duration (min)']
        df = df[['Date', 'Time', 'Activity', 'Duration']]
        print('Data pre-processing:\n{}\n'.format(df))
        df.to_csv("{}_{}_{}.csv".format(path_output, 0, "original"))

        low_durations = df.groupby(['Date'])['Duration'].agg('sum')
        low_durations = low_durations[low_durations < 400]
        low_durations_list = [date.strftime('%Y-%m-%d') for date in low_durations.index.tolist()]
        df = df[~df['Date'].isin(low_durations.index)]
        print("Found [{}] low duration days {}".format(len(low_durations_list), low_durations_list))
        df.to_csv("{}_{}_{}.csv".format(path_output, 1, "duration"))

        def get_missing_days():
            missing_days = df.set_index(df['Date']).drop_duplicates(subset='Date', keep='first')
            missing_days = missing_days.reindex(pd.date_range(df['Date'].min(), df['Date'].max(), freq='D'))
            missing_days = missing_days.loc[missing_days['Date'].isna()]
            return missing_days

        missing_days_global = get_missing_days()
        for i in range(missing_days_global.shape[0]):
            missing_days_loop = get_missing_days()
            df['Date'] = np.where(df['Date'] >= missing_days_loop.index[0], df['Date'] - pd.Timedelta(days=1), df['Date'])
        missing_days_list = [date.strftime('%Y-%m-%d') for date in missing_days_global.index.date.tolist()]
        print("Found [{}] missing days {}".format(len(missing_days_list), missing_days_list))
        df.to_csv("{}_{}_{}.csv".format(path_output, 2, "missing"))

        trimmed_days = df.set_index(df['Date']).drop_duplicates(subset='Date', keep='first').reset_index(drop=True)
        trimmed_days = trimmed_days[trimmed_days.index >= DAYS_TO_INCLUDE]
        trimmed_days_list = [date.strftime('%Y-%m-%d') for date in trimmed_days['Date'].tolist()]
        df = df[~df['Date'].isin(trimmed_days['Date'])]
        print("Found [{}] excessive days {}".format(len(trimmed_days_list), trimmed_days_list))
        df.to_csv("{}_{}_{}.csv".format(path_output, 3, "trimmed"))

        df['Radial'] = ((df['Date'] - df['Date'].min()).astype(str).str.split(' ').str.get(0)).astype(int)
        df['Minutes'] = df['Time'].astype(str).str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
        duplicates = df[df.index.duplicated()]
        if duplicates.shape[0] > 0:
            raise Exception('Duplicates detected:\n{}'.format(duplicates))
        zero_duration = df.loc[df['Duration'] == 0]
        if zero_duration.shape[0] > 0:
            raise Exception('Zero duration detected:\n{}'.format(zero_duration))
        na_duration = df.loc[df['Duration'].isna()]
        if na_duration.shape[0] > 0:
            raise Exception('None duration detected:\n{}'.format(na_duration))
        print('\nData post-processing:\n{}\n'.format(df))
        df.to_csv("{}_{}_{}.csv".format(path_output, 4, "polar"))

    stats_path = "{}_stats.txt".format(path_output)
    stats = "" \
            "${}'s \\/\\/ first \\/\\/ sleeps$\n\n" \
            "$Start = t_s = {}$\n" \
            "$Finish = t_f = {}$\n" \
            "$Timestamps = T = \\{{t: t\\geq{{t_s}}, \\/ t\\leq{{t_f}}\\}}$\n" \
            "$Sleeps = S = \\{{s_t: t \\/ \\in \\/ T\\}}$\n" \
            "$Epoch = t_e = days(t_f - t_s) = {}\\/days$\n" \
            "$Total = |S| = {}\\/sleeps$\n" \
            "$Average = |S|/t_e = {:.2f}\\/sleeps/day$\n" \
            "$Sum = ΣS/t_e = {:.0f}\\/min/day$\n" \
            "$Mean = \\overline{{s}} = {:.0f}\\/min$\n" \
            "$Median = med(S) = {:.0f}\\/min$\n" \
            "$Maximum = max(S) = {:.0f}\\/min$\n" \
            "$Mininum = min(S) = {:.0f}\\/min$\n" \
            "".format(
        child.title(),
        df.index[0].strftime("%d/%m/%Y\\/\\/%H\\colon%M\\colon%S"),
        df.index[-1].strftime("%d/%m/%Y\\/\\/%H\\colon%M\\colon%S"),
        df.set_index(df['Date']).drop_duplicates(subset='Date', keep='first').shape[0],
        df.shape[0],
        df.shape[0] / df.set_index(df['Date']).drop_duplicates(subset='Date', keep='first').shape[0],
        df.groupby(['Date'])['Duration'].agg('sum').mean(),
        df['Duration'].mean(),
        df['Duration'].median(),
        df['Duration'].max(),
        df['Duration'].min(),
    )
    with open(stats_path, 'w') as stats_file:
        stats_file.write(stats)
    print("Summary stats written to [{}]:\n\n{}".format(stats_path, stats))

    return (df, stats)


metadata_all = {
    'edwin': [
        # ('#F0F6FF', '#A4C9D7', None, False, False),
        ('#F0F6FF', '#A4C9D7', '#5A7B8F', True, True),
        # ('#EAF2FA', '#A4C9D7', '#5A7B8F', False, False),
        # ('#B2D2A4', '#1A4314', '#5A7B8F', False, False),
        # ('#7EC8E3', '#0000FF', '#5A7B8F', False, False),
        # ('#C3E0E5', '#274472', '#5A7B8F', False, False),
    ],
    # 'ada': [
    #     # ('#FFF1F0', '#EFACA7', None, False, False),
    #     ('#FFF1F0', '#EFACA7', '#D08D61', True, True),
    # ]
}

for child in metadata_all.keys():
    data = get_data(child, '../resources/data/cleansed/{}.csv'.format(child), '../resources/data/processed/{}_sleep'.format(child), 'Sleep')
    for metadata in metadata_all[child]:
        save_plot(child, data, '../resources/image/{}_sleep'.format(child), metadata[0], metadata[1], metadata[2], metadata[3], metadata[4])
