"""Utility functions."""

import pandas as pd
import rospy

from datetime import datetime

def get_param(name):
    """Return parameter value specified in ROS launch file or via command line, e.g. agent:=DDPG."""
    return rospy.get_param(name)


def get_timestamp(t=None, format='%Y-%m-%d_%H-%M-%S'):
    """Return timestamp as a string; default: current time, format: YYYY-DD-MM_hh-mm-ss."""
    if t is None:
        t = datetime.now()
    return t.strftime(format)


def plot_stats(csv_filename, columns=['total_reward'], **kwargs):
	"""Plot specified columns from CSV file."""
	df_stats = pd.read_csv(csv_filename)
	df_stats[columns].plot(**kwargs)
