# -*- coding: utf-8 -*-

import PIL
import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# TODO:
#   - Decide on size 30x30, 30x48, 40x60, 48x60, 60x60 print
#   - Only save variations at high res, scale to print size

SCALE_FACTOR = 40
DAYS_TO_INCLUDE = 475
PIL.Image.MAX_IMAGE_PIXELS = None
mfm.fontManager.ttflist.extend(mfm.createFontList(mfm.findSystemFonts(fontpaths='/Users/graham/Library/Fonts')))


def save_plot(label, df, output, background_colour, foreground_colour, label_colour, label_print, stats_print):
    file_name = '{}_{}_{}_{}.png' \
        .format(output, background_colour, foreground_colour, 'None' if label_colour is None else label_colour)

    print('Setup [{}] ... '.format(file_name))
    figure_dpi = min(900, int(6000 / SCALE_FACTOR))
    figure_width = 2 * SCALE_FACTOR
    figure_height = 3 * SCALE_FACTOR
    figure_header = 0.8 * SCALE_FACTOR
    font = 'inconsolata'
    font_size = 2.5 * SCALE_FACTOR
    font_baselineskip = 3.125 * SCALE_FACTOR
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
    plot = figure.add_subplot(111, projection='polar')
    plot.set_facecolor(background_colour)
    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    plot.spines['polar'].set_visible(False)

    print('Plotting [{}] image data ... '.format(df.shape[0]))
    for row in range(df.shape[0]):
        plot.bar(
            x=(-(df.loc[df.index[row], 'Minutes'] +
                 df.loc[df.index[row], 'Duration'] / 2) / 60 / 24 * 2 * np.pi) - np.pi / 2,
            height=1,
            width=df.loc[df.index[row], 'Duration'] / 60 / 24 * 2 * np.pi,
            bottom=df.loc[df.index[row], 'Radial'],
            color=foreground_colour)

    print('Annotating [{}] ... '.format(file_name))
    if label_colour is not None:
        ticks = np.linspace(0, np.pi * 2, num=25)
        hours = ['', '', '', '', '', '', '', '', '', '', '', '7am',
                 '', '', '', '', '', '', '', '', '', '', '', '7pm', '']
        for i in range(len(ticks)):
            if hours[i] != '':
                plt.plot((0, ticks[i]), (0, df['Radial'].max() + 1), color=label_colour, linewidth=0.1 * SCALE_FACTOR, alpha=0.5)
            if label_print:
                plt.text(ticks[i], df['Radial'].max() + 50, hours[i], ha='center', va='center', color=label_colour)
        if stats_print:
            df_duration = df['Duration']
            df_days_sum = df.groupby(['Date'])['Duration'].agg('sum')
            df_days = df.set_index(df['Date']).drop_duplicates(subset='Date', keep='first')

            def eqs(padding=None):
                return (" " if padding is None else r"\hspace{""" + str(padding * SCALE_FACTOR) + r"sp}") + "= "

            stats = (r"""
                        \begin{table}[h!]
                            \sffamily
                            \fontsize{""" + "{}".format(font_size) + r"""}{""" + "{}".format(font_baselineskip) + r"""}\selectfont
                            \setlength\tabcolsep{1ex}
                            \setlength\lightrulewidth{0.1ex}
                            \setlength\heavyrulewidth{0.3ex}
                            \begin{tabular}{@{} *2l @{}}
                                \toprule
                                \textbf{""" + label.title() + r"""'s first sleeps} &  \\
                                \midrule
                                start & $t_s$""" + "{}{}".format(eqs(66500), df.index[0].strftime("%d/%m/%Y %H:%M:%S")) + r""" \\
                                finish & $t_f$""" + "{}{}".format(eqs(), df.index[-1].strftime("%d/%m/%Y %H:%M:%S")) + r""" \\
                                epoch & $t_e$""" + eqs(67000) + r"""$t_f - t_s$ = """ + "{}".format(df_days.shape[0]) + r""" days \\
                                timestamps & $T$""" + eqs(62000) + r"""$\{t: t_s \leq t \leq t_f\}$ \\
                                sleeps & $S$""" + eqs(70000) + r"""$\{s_t: t \in T\}$ \\
                                total & $|S|$ = $|T|$""" + "{}{}".format(eqs(), df.shape[0]) + r""" sleeps \\
                                average & $|S|/t_e$""" + "{}{:.2f}".format(eqs(61000), df.shape[0] / df_days.shape[0]) + r""" sleeps/day \\
                                sum & $\Sigma S/t_e$""" + "{}{:.0f}".format(eqs(), df_days_sum.mean()) + r""" min/day \\
                                mean & $\overline{s}$""" + "{}{:.0f}".format(eqs(), df_duration.mean()) + r""" min \\
                                median & $\widetilde{s}$""" + "{}{:.0f}".format(eqs(), df_duration.median()) + r""" min \\
                                minimum & $\vee(S)$""" + "{}{:.0f}".format(eqs(), df_duration.min()) + r""" min \\
                                maximum & $\wedge(S)$""" + "{}{:.0f}".format(eqs(), df_duration.max()) + r""" min \\
                                standard deviation & $\sigma(S)$""" + "{}{:.0f}".format(eqs(63500), df_duration.std()) + r""" min \\
                                \bottomrule
                            \end{tabular}
                        \end{table}
                    """).replace("\n", "")
            plt.text(4.20, 875, stats, ha='left', va='center', color=label_colour)
    plt.ylim(ymax=df['Radial'].max() + 1)

    print('Saving [{}] ... '.format(file_name))
    plt.savefig(file_name, facecolor=background_colour, tight_layout=True, dpi=figure_dpi)
    plt.close('all')

    print('Cropping [{}] ... '.format(file_name))
    figure_png = Image.open(file_name)
    figure_png = figure_png.crop((0, figure_header * figure_dpi, figure_width * figure_dpi, (figure_height + figure_header) * figure_dpi))
    figure_png.save(file_name, dpi=(figure_dpi, figure_dpi))
    figure_png.show()

    print('Completed image processing\n')


def get_data(label, path_input, path_output, activity):
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
        df.to_csv('{}_{}_{}.csv'.format(path_output, 0, 'original'))

        low_durations = df.groupby(['Date'])['Duration'].agg('sum')
        low_durations = low_durations[low_durations < 400]
        low_durations_list = [date.strftime('%Y-%m-%d') for date in low_durations.index.tolist()]
        df = df[~df['Date'].isin(low_durations.index)]
        print('Found [{}] low duration days {}'.format(len(low_durations_list), low_durations_list))
        df.to_csv('{}_{}_{}.csv'.format(path_output, 1, 'duration'))

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
        print('Found [{}] missing days {}'.format(len(missing_days_list), missing_days_list))
        df.to_csv('{}_{}_{}.csv'.format(path_output, 2, 'missing'))

        trimmed_days = df.set_index(df['Date']).drop_duplicates(subset='Date', keep='first').reset_index(drop=True)
        trimmed_days = trimmed_days[trimmed_days.index >= DAYS_TO_INCLUDE]
        trimmed_days_list = [date.strftime('%Y-%m-%d') for date in trimmed_days['Date'].tolist()]
        df = df[~df['Date'].isin(trimmed_days['Date'])]
        print('Found [{}] excessive days {}'.format(len(trimmed_days_list), trimmed_days_list))
        df.to_csv('{}_{}_{}.csv'.format(path_output, 3, 'trimmed'))

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
        df.to_csv('{}_{}_{}.csv'.format(path_output, 4, 'polar'))

    return df


metadata_all = {
    'edwin': [
        # ('#F0F6FF', '#A4C9D7', None, False, False),
        ('#F0F6FF', '#A4C9D7', '#5A7B8F', True, True),
        # ('#EAF2FA', '#A4C9D7', '#5A7B8F', False, False),
        # ('#B2D2A4', '#1A4314', '#5A7B8F', False, False),
        # ('#7EC8E3', '#0000FF', '#5A7B8F', False, False),
        # ('#C3E0E5', '#274472', '#5A7B8F', False, False),
    ],
    'ada': [
        # ('#FFF1F0', '#EFACA7', None, False, False),
        ('#FFF1F0', '#EFACA7', '#D08D61', True, True),
    ]
}

for child in metadata_all.keys():
    data = get_data(child, '../resources/data/cleansed/{}.csv'.format(child), '../resources/data/processed/{}_sleep'.format(child), 'Sleep')
    for metadata in metadata_all[child]:
        save_plot(child, data, '../resources/image/{}_sleep'.format(child), metadata[0], metadata[1], metadata[2], metadata[3], metadata[4])
