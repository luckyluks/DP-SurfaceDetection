###################################################################################################
# author: Lukas Rauh
# description:  this is a wrapper around the realsensedatacollector.py file 
#               to record multiple clips in one take, from comandline or as script
###################################################################################################
import argparse
import time

from realsensedatacollector import *
from utils import *

def main():

    # use setup section if not run through command line
    if (args.runs==None): 
        args.runs = 2
        args.clip = 30
        args.warmup = 10
        args.fps = 6
        args.directory = "recordings"
        args.file_name_prefix = "darklab"      #add room here
        args.file_name_suffix = "{}s-{}fps".format(args.clip, args.fps)

    # display setup
    estimated_required_space = args.clip/2*args.runs * 1049000
    _, _, free = get_disk_space(format="raw")
    setup_message = "data collection wrapper setup:" \
                    +"\nsave directory: {}".format(args.directory) \
                    +"\nestimated required space: {}".format(convert_bytes(estimated_required_space)) \
                    +"\nfree disk space in \"/\": {}".format(convert_bytes(free)) \
                    +"\nsingle clip length: {}s".format(args.clip) \
                    +"\nsingle warmup length: {}s".format(args.warmup) \
                    +"\nrecording fps: {}s".format(args.fps) \
                    +"\nplanned runs: {}".format(args.runs) \
                    +"\nplanned runing time: {:.2f}min".format(args.runs*args.clip/60)
    print("-"*110+"\n"+setup_message)

    # check free space
    if int(estimated_required_space) >= int(free):
        print("ERROR: estimated required space \"{}\" is more than remaining free space \"{}\"! Execution stoped".format(convert_bytes(estimated_required_space), convert_bytes(free)))
        exit(1)

    start_time = time.time()

    # run recordings
    for run_index in range(args.runs):
        print("-"*110+"\nrun {} out of {}".format(run_index, args.runs-1))
        if run_index==0:
            record_runs(is_test_run=False, 
                        is_long_warmup_run=True, 
                        clip_length=args.clip, 
                        warmup_length=args.warmup, 
                        fps=args.fps, 
                        directory=args.directory,
                        file_name_prefix=args.file_name_prefix, 
                        file_name_suffix=args.file_name_suffix)
        else:
            record_runs(is_test_run=False, 
                        is_long_warmup_run=False, 
                        clip_length=args.clip, 
                        warmup_length=args.warmup, 
                        fps=args.fps, 
                        directory=args.directory,
                        file_name_prefix=args.file_name_prefix, 
                        file_name_suffix=args.file_name_suffix)
    
    print("-"*110+"\nall {} runs done in {:.2f}min!\n".format(args.runs, (time.time() - start_time)/60 )+"-"*110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", type=int, help="Runs to record")
    parser.add_argument("-c", "--clip", type=int, help="Clip length to record")
    parser.add_argument("-f", "--fps", type=int, help="FPS to record")
    parser.add_argument("-w", "--warmup", type=int, help="Warmup length to record")
    parser.add_argument("-d", "--directory", type=str, help="Path to save the images")
    args = parser.parse_args()

    main()