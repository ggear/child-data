# -*- coding: utf-8 -*-

import datetime
import os

import PIL
import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

WIDTH_INCH = 27
WIDTH_INCH_SCALE = 27
HEIGHT_INCH = 33 / WIDTH_INCH_SCALE * WIDTH_INCH
DOTS_INCH = 300

MAX_EPOCH = 479
MAX_DURATION = 806
QUANTISED_DURATION = 5
PIL.Image.MAX_IMAGE_PIXELS = None
mfm.fontManager.ttflist.extend(mfm.createFontList(mfm.findSystemFonts(fontpaths='/Users/graham/Library/Fonts')))


def save_plot(label, data, output, background_colour, foreground_colour, highlight_colour, label_colour, label_print, stats_print):
    file_name = '{}_{}_{}_{}_{}' \
        .format(output, background_colour, foreground_colour, highlight_colour, label_colour)

    print('Setup [{}.png] ... '.format(file_name))
    figure_width = WIDTH_INCH + 1 * (WIDTH_INCH + 1) / (WIDTH_INCH_SCALE + 1)
    figure_height = 1.08 * figure_width
    figure_header = 0.3 * figure_width
    figure_margin = 21 / 30 * figure_width / (WIDTH_INCH_SCALE + 1)
    figure_width_subplot = 18 * figure_width / (WIDTH_INCH_SCALE + 1)
    figure_height_subplot = 7.8 * figure_width / (WIDTH_INCH_SCALE + 1)
    figure_line_width = 0.05 * figure_width
    figure_line_style = (6.92, 10)
    figure_alpha = 0.5
    font_size = 1.25 * figure_width
    font_size_small = 0.8 * figure_width
    font_baselineskip = 1.05 * figure_width
    font = 'inconsolata'
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amsfonts}',
        r'\usepackage{booktabs}',
        r'\usepackage{' + font + '}',
    ]
    plt.rcParams.update({
        'font.size': font_size,
        'text.usetex': True,
    })
    figure = plt.figure(figsize=(figure_width, figure_height + figure_header))
    axes = figure.add_subplot(111, projection='polar')
    axes.set_facecolor(background_colour)
    axes.axes.get_xaxis().set_visible(False)
    axes.axes.get_yaxis().set_visible(False)
    axes.spines['polar'].set_visible(False)

    print('Plotting [{}.png] ... '.format(file_name))
    for index, row in data.iterrows():
        start, stop = row.loc[['Start', 'Stop']]
        radial = np.linspace(start, stop, int(10000 * (stop - start)))
        theta = -2 * np.pi * radial - np.pi / 2
        axes.plot(
            theta,
            radial,
            linewidth=figure_line_width,
            color=foreground_colour if row.loc['Period'] == 'Day' else highlight_colour,
            solid_capstyle='butt',
            alpha=figure_alpha
        )

    print('Annotating [{}.png] ... '.format(file_name))
    if label_colour is not None:
        ticks = np.linspace(0, np.pi * 2, num=25)
        hours = ['', '', '', '', '', '', '', '', '', '', '', '7am',
                 '', '', '', '', '', '', '', '', '', '', '', '7pm', '']
        for i in range(len(ticks)):
            if hours[i] != '':
                plt.plot((0, ticks[i]), (0, data['Stop'].max()),
                         linestyle=(figure_line_style[0] / 2, figure_line_style), color=label_colour, linewidth=figure_line_width,
                         alpha=figure_alpha)
            if label_print:
                plt.text(ticks[i], data['Stop'].max() + 50, hours[i], ha='center', va='center', color=label_colour)
        if stats_print:
            df = data
            df_duration = df['Duration']
            df_days_sum = df.groupby(['Date'])['Duration'].agg('sum')
            df_days = df.set_index(df['Date']).drop_duplicates(subset='Date', keep='first')

            def eqs(padding=None):
                return (" " if padding is None else r"\hspace{""" + str(padding * figure_width) + r"sp}") + "= "

            stats = (r"""
                        \begin{table}[h!]
                            \sffamily
                            \fontsize{""" + "{}".format(font_size_small) + r"""}{""" + "{}".format(font_baselineskip) + r"""}\selectfont
                            \setlength\tabcolsep{1ex}
                            \setlength\lightrulewidth{0.1ex}
                            \setlength\heavyrulewidth{0.3ex}
                            \begin{tabular}{@{} *2l @{}}
                                \toprule
                                \textbf{""" + label.title() + r"""'s first sleeps} &  \\
                                \midrule
                                start & $t_s$""" + "{}{}".format(eqs(33250), df.index[0].strftime("%d/%m/%Y %H:%M:%S")) + r""" \\
                                finish & $t_f$""" + "{}{}".format(eqs(), df.index[-1].strftime("%d/%m/%Y %H:%M:%S")) + r""" \\
                                epoch & $t_e$""" + eqs(33500) + r"""$t_f - t_s$ = """ + "{}".format(df_days.shape[0]) + r""" days \\
                                timestamps & $T$""" + eqs(31500) + r"""$\{t: t_s \leq t \leq t_f\}$ \\
                                sleeps & $S$""" + eqs(35500) + r"""$\{s_t: t \in T\}$ \\
                                total & $|S|$ = $|T|$""" + "{}{}".format(eqs(), df.shape[0]) + r""" sleeps \\
                                average & $|S|/t_e$""" + "{}{:.2f}".format(eqs(30250), df.shape[0] / df_days.shape[0]) + r"""
                                sleeps/day \\
                                sum & $\Sigma S/t_e$""" + "{}{:.0f}".format(eqs(), df_days_sum.mean()) + r""" min/day \\
                                mean & $\overline{s}$""" + "{}{:.0f}".format(eqs(), df_duration.mean()) + r""" min \\
                                median & $\widetilde{s}$""" + "{}{:.0f}".format(eqs(), df_duration.median()) + r""" min \\
                                minimum & $\vee(S)$""" + "{}{:.0f}".format(eqs(), df_duration.min()) + r""" min \\
                                maximum & $\wedge(S)$""" + "{}{:.0f}".format(eqs(), df_duration.max()) + r""" min \\
                                standard deviation & $\sigma(S)$""" + "{}{:.0f}".format(eqs(31750), df_duration.std()) + r""" min \\
                                \bottomrule
                            \end{tabular}
                        \end{table}
                    """).replace("\n", "")
            plt.text(4.1, 775, stats, ha='left', va='center', color=label_colour)
    plt.ylim(ymax=df['Stop'].max())

    print('Saving [{}.png] ... '.format(file_name))
    plt.savefig('{}.png'.format(file_name), facecolor=background_colour, tight_layout=True, dpi=DOTS_INCH)
    plt.close('all')

    hist_width = 1
    hist_bins = 100
    hist_bins_range = (0, 850)
    label_x = r"\fontsize{" + "{}".format(font_size_small) + r"}{" + "{}".format(font_baselineskip) + r"}\selectfont{3 hours}"
    label_y = r"\fontsize{" + "{}".format(font_size_small) + r"}{" + "{}".format(font_baselineskip) + r"}\selectfont{125 instances}"
    figure = plt.figure(figsize=(figure_width_subplot, figure_height_subplot))
    axes = figure.add_subplot(111)
    plt.hist(data[data['Period'] == 'Day']['Duration'], facecolor=foreground_colour,
             bins=hist_bins, range=hist_bins_range, rwidth=hist_width, edgecolor=background_colour, alpha=figure_alpha)
    plt.hist(data[data['Period'] == 'Night']['Duration'], facecolor=highlight_colour,
             bins=hist_bins, range=hist_bins_range, rwidth=hist_width, edgecolor=background_colour, alpha=figure_alpha)
    plt.xlim(0, hist_bins_range[1])
    plt.ylim(0, 260)
    plt.axis('off')
    axes.axes.axvline(174.5, 0, 0.85, linestyle=(0, figure_line_style), color=label_colour, linewidth=figure_line_width, alpha=figure_alpha)
    axes.text(0.206, 0.9, label_x, transform=axes.axes.transAxes, ha='center', va='center', color=label_colour)
    axes.axes.axhline(124, 0, 0.77, linestyle=(0, figure_line_style), color=label_colour, linewidth=figure_line_width, alpha=figure_alpha)
    axes.text(0.836, 0.485, label_y, transform=axes.axes.transAxes, ha='center', va='center', color=label_colour)
    plt.savefig('{}_tmp.png'.format(file_name), facecolor=background_colour, tight_layout=True, dpi=DOTS_INCH)
    plt.close('all')

    print('Cropping [{}.png] ... '.format(file_name))
    crop_top_left_x = int(figure_margin * DOTS_INCH)
    crop_top_left_y = int(figure_header * DOTS_INCH)
    crop_bottom_right_x = int(figure_width * DOTS_INCH)
    crop_bottom_right_y = int((figure_height + figure_header) * DOTS_INCH)
    crop_width_offset = (crop_bottom_right_x - crop_top_left_x - WIDTH_INCH * DOTS_INCH) / 2
    crop_height_offset = (crop_bottom_right_y - crop_top_left_y - HEIGHT_INCH * DOTS_INCH) / 2
    with Image.open('{}.png'.format(file_name)) as figure_png:
        with Image.open('{}_tmp.png'.format(file_name)) as tmp_png:
            figure_png = figure_png.crop((
                crop_top_left_x + crop_width_offset, crop_top_left_y + crop_height_offset,
                crop_bottom_right_x - crop_width_offset, crop_bottom_right_y - crop_height_offset
            ))
            tmp_png = tmp_png.crop((
                int((675 - 20) / 300 * DOTS_INCH), int((302 - 60 - 20) / 300 * DOTS_INCH),
                int((4617 + 20) / 300 * DOTS_INCH), int((2003 + 200 + 20) / 300 * DOTS_INCH)
            ))
            figure_png.paste(tmp_png, (int(3168 / 300 * DOTS_INCH), int(7055 / 300 * DOTS_INCH)))
            figure_png.save('{}.png'.format(file_name), dpi=(DOTS_INCH, DOTS_INCH))
            os.remove('{}_tmp.png'.format(file_name))
            figure_png.show()

    print('Completed image processing\n')


def get_data(label, path_input, path_output, activity):
    data = pd.read_csv(path_input)
    data = data.loc[data['Activity'] == activity]
    data = data.set_index(pd.to_datetime(data['Date and Time'], format='%Y-%m-%d %H:%M:%S'))

    with pd.option_context(
            'display.max_rows', 1000,
            'display.max_columns', None,
            'display.width', None):
        data['Date'] = data.index.date
        data['Time'] = data.index.time
        data["Finish"] = data.index + pd.to_timedelta(data['Duration (min)'], unit='m')
        data['Duration'] = data['Duration (min)']
        data = data[['Date', 'Time', 'Finish', 'Duration']]
        print('Data pre-processing:\n{}\n'.format(data))
        data.to_csv('{}_{}_{}.csv'.format(path_output, 0, 'original'))

        low_durations = data.groupby(['Date'])['Duration'].agg('sum')
        low_durations = low_durations[low_durations < 250]
        low_durations_list = sorted([date.strftime('%Y-%m-%d') for date in low_durations.index.tolist()])
        print('Found [{}] low duration days {}'.format(len(low_durations_list), low_durations_list))

        missing_days = data.set_index(data['Date']).drop_duplicates(subset='Date', keep='first')
        missing_days = missing_days.reindex(pd.date_range(data['Date'].min(), data['Date'].max(), freq='D'))
        missing_days = missing_days.loc[missing_days['Date'].isna()]
        missing_days_list = missing_days.index.date.tolist()
        missing_days_conflated = ([], [], [])
        for index, day in enumerate(missing_days_list):
            count = 1
            while index + count < len(missing_days_list) and (day + pd.Timedelta(days=1)) == missing_days_list[index + count]:
                count += 1
            if len(missing_days_conflated[1]) == 0 or missing_days_conflated[2][-1] == 1:
                missing_days_conflated[0].append(day)
                missing_days_conflated[1].append(count)
                missing_days_conflated[2].append(count)
            else:
                missing_days_conflated[2][-1] = missing_days_conflated[2][-1] - 1
        for index in range(1, len(missing_days_conflated[0])):
            for index_offset in range(index, len(missing_days_conflated[0])):
                missing_days_conflated[0][index_offset] = \
                    missing_days_conflated[0][index_offset] - pd.Timedelta(days=missing_days_conflated[1][index - 1])
        for index, day in enumerate(missing_days_conflated[0]):
            data.set_index(np.where(data.index >= pd.to_datetime(day),
                                    data.index - pd.Timedelta(days=missing_days_conflated[1][index]), data.index), inplace=True)
            data['Date'] = data.index.date
            data['Time'] = data.index.time
            data["Finish"] = data.index + pd.to_timedelta(data['Duration'], unit='m')
        missing_days_list = sorted([date.strftime('%Y-%m-%d') for date in missing_days.index.date.tolist()])
        print('Found [{}] missing days {}'.format(len(missing_days_list), missing_days_list))
        data.to_csv('{}_{}_{}.csv'.format(path_output, 1, 'missing'))

        start_unixtime = data.index.astype(np.int64)
        finish_unixtime = data["Finish"].astype(np.int64)
        previous_finish_unixtime = finish_unixtime.shift(1)
        overlapping_sleeps = data[start_unixtime <= previous_finish_unixtime]
        overlapping_sleeps_list = sorted(set([date.strftime('%Y-%m-%d') for date in overlapping_sleeps.index.tolist()]))
        print('Found [{}] overlapping sleeps on days {}'.format(len(overlapping_sleeps_list), overlapping_sleeps_list))

        trimmed_days = data.set_index(data['Date']).drop_duplicates(subset='Date', keep='first').reset_index(drop=True)
        trimmed_days = trimmed_days[trimmed_days.index >= MAX_EPOCH]
        trimmed_days_list = sorted([date.strftime('%Y-%m-%d') for date in trimmed_days['Date'].tolist()])
        data = data[~data['Date'].isin(trimmed_days['Date'])]
        print('Found [{}] trimmed days {}'.format(len(trimmed_days_list), trimmed_days_list))
        data.to_csv('{}_{}_{}.csv'.format(path_output, 2, 'trimmed'))

        data['Period'] = data['Time'].apply(lambda x: 'Day' if x <= datetime.time(17, 0, 0) and x >= datetime.time(8, 0, 0) else 'Night')
        data['Start'] = (data.index - data.index[0].replace(hour=0, minute=0, second=0)) / np.timedelta64(1, 'D')
        data['Stop'] = (data['Finish'] - data.index[0].replace(hour=0, minute=0, second=0)) / np.timedelta64(1, 'D')
        duplicates = data[data.index.duplicated()]
        if duplicates.shape[0] > 0:
            raise Exception('Duplicates detected:\n{}'.format(duplicates))
        zero_duration = data.loc[data['Duration'] == 0]
        if zero_duration.shape[0] > 0:
            raise Exception('Zero duration detected:\n{}'.format(zero_duration))
        na_duration = data.loc[data['Duration'].isna()]
        if na_duration.shape[0] > 0:
            raise Exception('None duration detected:\n{}'.format(na_duration))
        print('\nData post-processing:\n{}\n'.format(data))
        data.to_csv('{}_{}_{}.csv'.format(path_output, 3, 'polar'))

    return data


metadata_all = {
    'edwin': [
        ('#f0f6ff', '#c2e1ec', '#afd0e7', '#5a7b8f', True, True),
    ],
    'ada': [
        ('#fff1f0', '#f4c7c4', '#efaca7', '#a5662f', True, True),
    ]
}

for child in metadata_all.keys():
    data = get_data(child, '../resources/data/cleansed/{}.csv'.format(child),
                    '../resources/data/processed/{}_sleep'.format(child), 'Sleep')
    for metadata in metadata_all[child]:
        save_plot(child, data, '../resources/image/{}_sleep'.format(child),
                  metadata[0], metadata[1], metadata[2], metadata[3], metadata[4], metadata[5])
