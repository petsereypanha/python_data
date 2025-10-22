import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def apply_theme():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.float_format', '{:,.3f}'.format)

    flatui = ["#2e86de", "#ff4757", "#feca57", "#2ed573", "#ff7f50", "#00cec9", "#fd79a8", "#a4b0be"]
    flatui_palette = sns.color_palette(flatui)
    sns.palplot(flatui_palette)
    sns.set_palette(flatui_palette)

    sns.set_style("darkgrid", {
        'axes.edgecolor': '#2b2b2b',
        'axes.facecolor': '#2b2b2b',
        'axes.labelcolor': '#919191',
        'figure.facecolor': '#2b2b2b',
        'grid.color': '#545454',
        'patch.edgecolor': '#2b2b2b',
        'text.color': '#bababa',
        'xtick.color': '#bababa',
        'ytick.color': '#bababa'
    })