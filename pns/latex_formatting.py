#!/usr/bin/python
"""Contains helper functions for saving, loading and input/output."""

import numpy as np
import pandas as pd


def latex_sf(value, start_end_sf=(2, -2), dp=4):
    if value != 0:
        power = int(np.log10(abs(value)))
    else:
        power = 0
    if power >= start_end_sf[0] or power <= start_end_sf[1]:
        value = value * (10 ** (- power))
    else:
        power = False
    output = round(value, dp)
    if dp == 0:
        output = int(output)
    output = str(output)
    if power is not False and power != 0:
        output += 'cdot 10{' + str(power) + '}'
    return output


def latex_form(value_in, error_in, start_end_sf=(2, -2), dp=4):
    try:
        if value_in == 0:
            power = 0
        else:
            power = int(np.log10(abs(value_in)))
        if power >= start_end_sf[0] or power <= start_end_sf[1]:
            value = value_in * (10 ** (- power))
        else:
            value = value_in
            power = 0
        output = '{:.{prec}f}'.format(value, prec=dp)
        # output = str(round(value, dp))
        error = error_in / (10 ** (power - dp))
        if error < 0:
            print('latex_form: warning: error on final digit=' + str(error) +
                  ' < 0')
        if error == 0:
            output += '(0)'
        else:
            output += '('
            if error > 1:
                error_dp = 0
            else:
                error_dp = int(np.ceil((-1.0) * np.log10(error)))
            output += '{:.{prec}f}'.format(error, prec=error_dp)
            output += ')'
        if power is not False and power != 0:
            output += ' cdot 10{' + str(power) + '}'
        return output
    except (ValueError, OverflowError):
        return str(value_in) + '(' + str(error_in) + ')'


def latex_form_percent(value_in, error_in, start_end_sf=(2, -2), dp=4):
    return latex_form(value_in * 100, error_in * 100,
                      start_end_sf=start_end_sf, dp=(dp - 2)) + '\\%'


def latex_format_df(df, cols=None, rows=None, dp_list=None):
    if cols is None:
        cols = [n for n in list(df.columns) if (len(n) <= 3 or
                                                n[-4:] != '_unc')]
    if rows is None:
        rows = list(df.index)
    if dp_list is None:
        dp_list = [4] * len(rows)
    latex_dict = {}
    for c in cols:
        latex_dict[c] = []
        for i, r in enumerate(rows):
            temp = latex_form(df[c][r], df[c + '_unc'][r], dp=dp_list[i],
                              start_end_sf=[4, -4])
            latex_dict[c].append(temp)
    latex_df = pd.DataFrame(latex_dict, index=rows)
    return latex_df
