#!/usr/bin/env python

"""
@author: Jordan Graesser
"""

import os
import sys
import time
import ast
import argparse
from copy import copy

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

# Matplotlib
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import ticker
except ImportError:
    raise ImportError('Matplotlib must be installed')

# Pandas
try:
    import pandas
except ImportError:
    raise ImportError('Pandas must be installed')

# Statsmodels
try:
    import statsmodels.api as sm
except:
    pass

try:
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
except:
    pass

# SciPy
try:
    from scipy import stats
except ImportError:
    raise ImportError('SciPy must be installed')

# PySAL
try:
    import pysal
except:
    print('PySAL is not installed')


def _handle_axis_frame(axis, turn_off=['left', 'right', 'top', 'bottom']):

    for location in turn_off:
        axis.spines[location].set_visible(False)

    return axis


def _handle_axis_ticks(axis, turn_off_xaxis=True, turn_off_yaxis=True):

    if turn_off_xaxis:
        for tic in axis.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False

    if turn_off_yaxis:
        for tic in axis.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False

    return axis


def _handle_tick_labels(axis, xtick_labels=[], ytick_labels=[]):

    axis.set_xticks(xtick_labels)
    axis.set_yticks(ytick_labels)

    return axis


def _handle_axis_labels(axis, xlabel=None, ylabel=None, xlabel_pad=0, ylabel_pad=0):

    if xlabel:
        axis.set_xlabel(xlabel)
        axis.xaxis.labelpad = xlabel_pad

    if ylabel:
        axis.set_ylabel(ylabel)
        axis.yaxis.labelpad = ylabel_pad

    return axis


def _handle_axis_spines(axis, axis_color='#424242'):

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_edgecolor(axis_color)
    axis.spines['bottom'].set_edgecolor(axis_color)

    return axis


def _add_axis_commas(axis, xaxis=True, yaxis=True):

    if xaxis:
        axis.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    if yaxis:
        axis.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, p: format(int(y), ',')))

    return axis


class plot(object):

    def __init__(self):

        self.time_stamp = time.asctime(time.localtime(time.time()))

    def setup_plot(self, output, figsize=(8, 8), dpi=200, background_color='#2E2E2E',
                   font_color='#F2F2F2', transparent=False, x_axis_is_year=False):

        self.background_color = background_color
        self.font_color = font_color
        self.transparent = transparent
        self.dpi = dpi

        if self.transparent:
            frameon = False
        else:
            frameon = True

        if isinstance(output, str):

            mpl.rcParams['font.size'] = 18.
            mpl.rcParams['axes.labelsize'] = 18.
            mpl.rcParams['xtick.labelsize'] = 14.
            mpl.rcParams['ytick.labelsize'] = 16.

        else:

            mpl.rcParams['font.size'] = 10.
            mpl.rcParams['axes.labelsize'] = 10.
            mpl.rcParams['xtick.labelsize'] = 7.
            mpl.rcParams['ytick.labelsize'] = 7.

        mpl.rcParams['font.family'] = 'Calibri'
        mpl.rcParams['axes.facecolor'] = self.background_color
        mpl.rcParams['axes.edgecolor'] = self.background_color
        mpl.rcParams['axes.labelcolor'] = self.font_color
        mpl.rcParams['figure.facecolor'] = self.background_color
        mpl.rcParams['figure.edgecolor'] = self.background_color
        mpl.rcParams['ytick.color'] = self.font_color
        mpl.rcParams['xtick.color'] = self.font_color

        mpl.rcParams['savefig.transparent'] = self.transparent
        mpl.rcParams['savefig.dpi'] = self.dpi

        # plt.rc('text', usetex=True)

        fig = plt.figure(figsize=figsize, frameon=frameon, facecolor=self.background_color)
        self.ax = fig.add_subplot(111)

        self.ax.get_xaxis().tick_bottom()
        self.ax.get_yaxis().tick_left()

        self.ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        if not x_axis_is_year:
            self.ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        self.ax = _handle_axis_spines(self.ax, axis_color=self.font_color)

        plt.tick_params(\
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top='off',
            right='off',
            bottom='off',      # ticks along the bottom edge are off
            left='off')         # ticks along the top edge are off

    def load_data(self, data, keyword_filter={}):

        try:
            self.df = pandas.read_csv(data)
        except:
            self.df = pandas.read_excel(data)

        if keyword_filter:

            for kw, fv in keyword_filter.iteritems():

                self.df = self.df.loc[self.df[kw] == fv, :]

    def load_data_from_store(self, kw, fv):

        self.df = self.store[kw].loc[self.store[kw]['Variable'] == fv, :]

    def merge_data(self, merge_on):

        for k, v in self.store.iteritems():

            print(k, v)

    def load_shapefile(self, shape_name):

        if shape_name == 'argentina':
            shapefile1 = '/Volumes/SEAGATE-DOS/GIS/political/countries/argentina/ARG/departamentos_v2.shp'
            shapefile2 = '/Users/Safflower/Documents/GIS/political/countries/argentina/ARG/departamentos_v2.shp'
            self.unq_shp_idx = 7
        elif shape_name == 'brazil':
            shapefile1 = '/Volumes/SEAGATE-DOS/GIS/political/countries/brazil/BRA/adm4_v2_WGS84.shp'
            shapefile2 = '/Users/Safflower/Documents/GIS/political/countries/brazil/BRA/adm4_v2_WGS84.shp'
            self.unq_shp_idx = 1
        elif shape_name == 'paraguay':
            shapefile1 = '/Volumes/SEAGATE-DOS/GIS/political/countries/paraguay/PRY_adm1.shp'
            shapefile2 = '/Users/Safflower/Documents/GIS/political/countries/paraguay/PRY_adm1.shp'
            self.unq_shp_idx = 3
        elif shape_name == 'colombia':
            shapefile1 = '/Volumes/SEAGATE-DOS/GIS/political/countries/colombia/COL_adm1.shp'
            shapefile2 = '/Users/Safflower/Documents/GIS/political/countries/colombia/COL_adm1.shp'
            self.unq_shp_idx = 3

        try:
            self.shapefile_dbf = pysal.open(shapefile1.replace('.shp', '.dbf'), 'r')
            self.the_shapefile = copy(shapefile1)
        except:
            self.shapefile_dbf = pysal.open(shapefile2.replace('.shp', '.dbf'), 'r')
            self.the_shapefile = copy(shapefile2)

        self.shapefile_dbf = dict([(col, np.array(self.shapefile_dbf.by_col(col))) for col in self.shapefile_dbf.header])
        self.shapefile_dbf = pandas.DataFrame(self.shapefile_dbf)

    def scatter(self, x, y, title='Title', size=20, alpha=.5, color='#819FF7',
                line_color='#FE9A2E', line_width=2, confidence=.95, ax=None):

        if not ax:
            self.ax = self.ax
        else:
            self.ax = ax

        self.x = self.df[x].values
        self.y = self.df[y].values

        x_ = sm.add_constant(self.x)
        model = sm.OLS(self.y, x_).fit()

        # model, __, __, fitted_data, __ = least_squares.fit(df=self.df, y_label=y, x_label=x)
        # model.predict(exog=self.df[[x]])

        x_pred = np.linspace(self.x.min(), self.x.max(), 50)
        x_pred2 = sm.add_constant(x_pred)
        y_pred = model.predict(x_pred2)

        # Regression confidence interval. Plot the
        # probability that the true regression lies
        # within the confidence interval.
        y_err = self.y - model.predict(x_)
        mean_x = x_.T[1].mean()
        n = len(x_)
        dof = n - model.df_model - 1
        t = stats.t.ppf(confidence, df=dof)
        s_err = np.sum(np.power(y_err, 2))
        conf = t * np.sqrt((s_err / (n - 2)) * (1. / n + (np.power((x_pred-mean_x), 2) /
                                                          ((np.sum(np.power(x_pred, 2))) - n * (np.power(mean_x, 2))))))
        upper = y_pred + abs(conf)
        lower = y_pred - abs(conf)

        # The prediction interval. Plot the probability of
        # the real value (y) lies within the prediction
        # confidence interval.
        __, lower_pr, upper_pr = wls_prediction_std(model, exog=x_pred2, alpha=1.-confidence)

        self.ax.scatter(self.x, self.y, s=size, alpha=alpha, color=color, edgecolor='none')
        self.ax.plot(x_pred, y_pred, '-', color=line_color, linewidth=line_width)
        self.ax.fill_between(x_pred, lower, upper, color='#888888', alpha=.4)
        self.ax.fill_between(x_pred, lower_pr, upper_pr, color='#888888', alpha=.1)

        self.ax.set_title(title, color=self.font_color)
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)

        # self.ax.text(self.x[-1], self.y[-1], r'$test$')

        return self.ax

    def boxplot(self, output, rows, fields, plot_by, **kwargs):

        area_threshold = 1000

        # if fields:
        #     self.df_sub = self.df.loc[:, fields]

        try:
            marker_size = kwargs['marker_size']
        except:
            marker_size = 100

        try:
            marker_colors = kwargs['marker_colors']
        except:
            marker_colors = ['#2EFEF7', '#FE9A2E']

        try:
            line_weight = kwargs['line_weight']
        except:
            line_weight = 2

        try:
            line_style = kwargs['line_style']
        except:
            line_style = ':'

        if rows:

            for k, v in rows.iteritems():

                self.df_sub = self.df.loc[self.df[k] == v, :]

                # get unique names
                if plot_by == 'ecoregion':
                    unique_name = 'ECO_NAME'
                    uniques = np.unique(self.df_sub.ECO_NAME)
                elif plot_by == 'country':
                    unique_name = 'COUNTRY'
                    uniques = np.unique(self.df_sub.COUNTRY)

                # remove unwanted
                uniques = [u for u in uniques if not isinstance(u, float)]

                x = range(1, (len(uniques)+1)*2)

                x_min = 1000
                x_max = -1000

                labels_xs = []
                labels = []

                # pre-processing
                for ui, unique in enumerate(uniques):

                    self.df_sub_unique = self.df_sub.loc[self.df_sub[unique_name] == unique, :]

                    label = self.df_sub_unique.iloc[0, :].ECO_NAME

                    if 'rock' in label.lower():
                        uniques = np.delete(uniques, list(uniques).index(label))
                        continue

                    for fi, field in enumerate(fields):

                        y = self.df_sub_unique[field].values

                        if (np.max(np.abs(y)) < area_threshold) or (len(y) < 5):

                            uniques = np.delete(uniques, list(uniques).index(label))
                            break

                for ui, unique in enumerate(uniques):

                    self.df_sub_unique = self.df_sub.loc[self.df_sub[unique_name] == unique, :]

                    label = self.df_sub_unique.iloc[0, :].ECO_NAME
                    label = label.decode('ascii', 'ignore')
                    label = label.replace(' ', '\n')

                    labels.append(label)

                    offsets = []
                    fi = -.2
                    for field in fields:
                        offsets.append(fi)
                        fi += .4

                    medians = []
                    lw = []
                    up = []
                    xs = []
                    label_mean = 0.

                    for fi, field in enumerate(fields):

                        y = self.df_sub_unique[field].values

                        medians.append(np.median(y))
                        lw.append(np.percentile(y, 25))
                        up.append(np.percentile(y, 75))

                        x_sub = np.zeros(len(y), dtype='float32')

                        # x_sub = []
                        # xi = 0
                        # for f in xrange(0, len(y)):
                        #     x_sub.append(xi)
                        #     xi += 2

                        # x_sub = np.asarray(x_sub).astype(np.float32)
                        x_sub += (ui + 2)
                        x_sub += offsets[fi]

                        x_min = min(x_min, np.min(x_sub))
                        x_max = max(x_max, np.max(x_sub))

                        label_mean += x_sub[0]

                        xs.append(x_sub[0])

                        # plot the data
                        self.ax.scatter(x_sub, y, s=marker_size, edgecolor=marker_colors[fi], color='none', alpha=.5, \
                                        lw=1)

                        # plot the mean to N standard deviations
                        n_std = 1.
                        self.ax.plot([x_sub[0], x_sub[0]], [y.mean(), y.mean()+y.std()*n_std], '-', \
                                     color=self.font_color, lw=line_weight, alpha=.5)

                        self.ax.plot([x_sub[0], x_sub[0]], [y.mean(), y.mean()-y.std()*n_std], '-', \
                                     color=self.font_color, lw=line_weight, alpha=.5)

                    x1s = [xs[0]]
                    x1s.insert(0, xs[0]-.2)
                    x1s.append(xs[0]+.2)

                    x2s = [xs[-1]]
                    x2s.insert(0, xs[-1]-.2)
                    x2s.append(xs[-1]+.2)

                    self.ax.plot(x1s, [medians[0]]*len(x1s), color='#FA5858', lw=line_weight)#marker_colors[0])
                    self.ax.plot(x2s, [medians[-1]]*len(x2s), color='#FA5858', lw=line_weight)#marker_colors[1])

                    self.ax.plot(x1s, [lw[0]]*len(x1s), color=self.font_color, lw=line_weight-.25, ls=line_style)#marker_colors[0])
                    self.ax.plot(x2s, [lw[-1]]*len(x2s), color=self.font_color, lw=line_weight-.25, ls=line_style)#marker_colors[1])

                    self.ax.plot(x1s, [up[0]]*len(x1s), color=self.font_color, lw=line_weight-.25, ls=line_style)#marker_colors[0])
                    self.ax.plot(x2s, [up[-1]]*len(x2s), color=self.font_color, lw=line_weight-.25, ls=line_style)#marker_colors[1])

                    label_mean /= 2

                    labels_xs.append(label_mean)

                self.ax.set_xlim(x_min-.5, x_max+.5)
                self.ax.set_ylim(-10001, 10000)

                self.ax.set_xticks(labels_xs)
                self.ax.set_xticklabels(labels)

                self.ax.set_ylabel('Change rate (ha/year)')

            self.save_or_show(output)

    def join_dataframes(self, df1, df2, field1, field2):

        # convert join fields to string
        df1[field1] = df1[field1].astype('str')
        df2[field2] = df2[field2].astype('str')

        self.df_join = pandas.merge(df1, df2, left_on=field1, right_on=field2, how='inner')

    def open_hdf_store(self, hdf_store):

        self.store = pandas.HDFStore(hdf_store)

    def close_hdf_store(self):

        self.store.close()

    def merge_dataframes(self):

        si = 0

        for kw, fv in self.store.iteritems():

            current_df = self.store[kw[1:]]
            current_df = current_df.loc[current_df['Variable'] == 'Planted_ha', :]

            rename_dict = {}
            for y in xrange(1969, 2014):
                rename_dict[str(y)] = '%s_%d' % (kw[1:], y)

            current_df.rename(columns=rename_dict, inplace=True)

            if si == 0:
                self.df_merge = current_df
            else:
                self.df_merge = pandas.merge(self.df_merge, current_df, on='GeoId', how='outer')

            si += 1

        self.df_merge = self.df_merge.replace([np.inf, -np.inf, np.nan], 0.)

    def open_crop(self, crop2open, variable2open):

        self.df_crop = self.store[crop2open]
        self.df_crop = self.df_crop.loc[self.df_crop['Variable'] == variable2open, :]

    def save_or_show(self, output):

        if not output:
            plt.show()
        else:
            plt.savefig(output)


def manage_plot(input=None, output=None, type='scatter', x=None, y=None, rows={}, data_fields=[], unique_fields=[], \
                plot_by='ecoregion', shape_name='argentina', hdf_stores=[], keyword_filter={}):

    # sets the class object
    mpp = plot()

    mpp.setup_plot(output)

    if input:

        # loads the table into a dataframe
        mpp.load_data(input, keyword_filter=keyword_filter)

    if type == 'box':

        mpp.boxplot(output, rows, data_fields, plot_by)

    elif type == 'trends':

        for hdf_store in hdf_stores:

            mpp.open_hdf_store(hdf_store=hdf_store)

            mpp.merge_dataframes()

            year_range = range(1969, 2014)
            columns1 = ['Soybeans_%d' % yr for yr in year_range]
            columns2 = ['Sunflower_%d' % yr for yr in year_range]

            for row in xrange(0, mpp.df_merge.shape[0]):

                color = np.random.rand(3,)
                # mpp.ax.plot(year_range, mpp.df_merge.loc[row, columns1], '-.', color=color, lw=1)
                mpp.ax.plot(year_range, mpp.df_merge.loc[row, columns2], '-', color=color, lw=1, alpha=.5)

                # if row == 110:
                #     break

            mpp.close_hdf_store()

        mpp.ax.set_xlim(1969.5, 2013.5)
        # mpp.ax.set_ylim(0, 10000)

        mpp.save_or_show(output)

    elif type == 'scatter':

        colors = ['purple', 'orange', 'green']

        if hdf_stores:

            for color, hdf_store in zip(colors, hdf_stores):

                mpp.open_hdf_store(hdf_store=hdf_store)

                colors = ['purple', 'orange', 'green']

                ci = 0

                for kw, fv in keyword_filter.iteritems():

                    color = colors[ci]

                    mpp.load_data_from_store(kw, fv)

                    mpp.ax.scatter(mpp.df[data_fields[0]].values, mpp.df[data_fields[1]].values, s=30, alpha=.5, \
                                   color=color, edgecolor='none')

                    ci += 1

                mpp.close_hdf_store()

            mpp.ax.set_xlim(0, 100000)
            mpp.ax.set_ylim(0, 100000)

        else:

            mpp.scatter(output, x, y)

        mpp.save_or_show(output)

    else:

        # loads the shapefile dbf into a dataframe
        mpp.load_shapefile(shape_name=shape_name)

        # join the shapefile dbf with the data table
        mpp.join_dataframes(mpp.shapefile_dbf, mpp.df, unique_fields[0], unique_fields[1])

        mpp.open_hdf_store(hdf_store=hdf_store)

        # join the census data with the data table
        mpp.open_crop('Soybeans', 'Planted_ha')
        mpp.join_dataframes(mpp.df_join, mpp.df_crop, unique_fields[0], 'GeoId')

        mpp.close_hdf_store()

        plt.scatter(mpp.df_join['ANL_CHR1_crop'].values, mpp.df_join['ANL_CHR2_crop'].values, s=2, alpha=.5)

        mpp.save_or_show(output)


def dict_split(dictionary):

    dictionary = dictionary.split(',')

    info_dict = '{'

    for i_counter, row in enumerate(dictionary):

        row_split = row.split(':')

        if (i_counter + 1) == len(dictionary):
            info_dict = "%s'%s':'%s'" % (info_dict, row_split[0], row_split[1])
        else:
            info_dict = "%s'%s':'%s'," % (info_dict, row_split[0], row_split[1])

    info_dict = '%s}' % info_dict

    return ast.literal_eval(info_dict)


def _examples():

    sys.exit("""\
    plot.py -i /table.csv --type box -r COUNTRY:Argentina -k SIG_POLY_crop:yes -f ANL_CHR1_crop,ANL_CHR2_crop -b ecoregion -o /figure.png
    plot.py -i /table.csv --type other -r COUNTRY:Argentina -f ANL_CHR1_crop,ANL_CHR2_crop -u cod_depto,cod_depto -s argentina -hdf /ARG.h5 -o /figure.png
    plot.py --type trends -r COUNTRY:Argentina -s argentina -hdf /ARG.h5 -o /figure.png\
    plot.py --type scatter -f 1990,2008 -k Soybeans:Planted_ha -hdf /ARG.h5,/PRY.h5 -o /figure.png

    plot.py --type scatter -x cattle -y soy -i /table.csv
    """)


def main():

    parser = argparse.ArgumentParser(description='Plots data', \
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input', help='The input data table', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output figure', default=None)
    parser.add_argument('-t', '--type', dest='type', help='The type of plot', default='scatter', \
                        choices=['scatter', 'bar', 'box', 'trends', 'other'])
    parser.add_argument('-x', dest='x', help='The x field to plot with scatter', default=None)
    parser.add_argument('-y', dest='y', help='The y field to plot with scatter', default=None)
    parser.add_argument('-f', '--fields', dest='fields', help='The fields to plot', default=[], nargs='+')
    parser.add_argument('-u', '--unique_fields', dest='unique_fields', help='The unique fields list (dbf, data table)', \
                        default=[], nargs='+')
    parser.add_argument('-r', '--rows', dest='rows', help='A label to identifer dictionary for rows to plot', \
                        default=None)
    parser.add_argument('-b', '--plot_by', dest='plot_by', help='How to plot the data', default='ecoregions', \
                        choices=['ecoregion', 'admin', 'country'])
    parser.add_argument('-s', '--shape_name', dest='shape_name', help='The shapefile to use', default='argentina')
    parser.add_argument('-hdf', '--hdf_stores', dest='hdf_stores', help='The HDF stores to use', default=[], nargs='+')
    parser.add_argument('-k', '--keyword_filter', dest='keyword_filter', \
                        help='A keyword dictionary to filter the data table', default=None)

    args = parser.parse_args()

    if args.examples:
        _examples()

    if args.rows:
        args.rows = dict_split(args.rows)

    if args.keyword_filter:
        args.keyword_filter = dict_split(args.keyword_filter)

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    manage_plot(args.input, output=args.output, type=args.type, x=args.x, y=args.y, rows=args.rows, \
                data_fields=args.fields, unique_fields=args.unique_fields, plot_by=args.plot_by, \
                shape_name=args.shape_name, hdf_stores=args.hdf_stores, keyword_filter=args.keyword_filter)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' % \
          (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
