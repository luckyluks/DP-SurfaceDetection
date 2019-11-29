import sys
import os
import time
import shutil

def generate_save_date_name(file_ending, prefix="", suffix=""):
    time_str = str(time.strftime("%Y%m%d-%H%M%S"))
    full_suffix = ( prefix+"_" if prefix!="" else "") + time_str + ( "_"+suffix if suffix!="" else "") + file_ending
    return full_suffix

def generate_save_date_name_short(file_ending, prefix="", suffix=""):
    time_str = str(time.strftime("%b%d"))
    full_suffix = ( prefix+"_" if prefix!="" else "") + time_str + ( "_"+suffix if suffix!="" else "") + file_ending
    return full_suffix

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)

def get_disk_space(path="/", format="converted"):
    total, used, free = shutil.disk_usage(path)
    # total = (total // (2**30))
    # used = (used // (2**30))
    # free = (free // (2**30))
    if not format=="raw":
        total = convert_bytes(total)
        used = convert_bytes(used)
        free = convert_bytes(free)

    return total, used, free




def main():
    print(os.path.join("directory",generate_save_date_name('.mp4', 'object_detection_30s')))


if __name__ == '__main__':
    main()