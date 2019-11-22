import sys
import os
import time

def generate_save_date_name(file_ending, prefix="", suffix=""):
    time_str = str(time.strftime("%Y%m%d-%H%M%S"))
    full_suffix = ( prefix+"_" if prefix!="" else "") + time_str + ( "_"+suffix if suffix!="" else "") + file_ending
    return full_suffix