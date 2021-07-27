""" Main function to launch a full scale simulation from the command line using the configuration files.
"""

import sys, getopt
from pathlib import Path

# https://www.tutorialspoint.com/python/python_command_line_arguments.htm
from lppydsmc import main

def run(argv):
    try:
        opts, args = getopt.getopt(argv,"hp:s", ['path=']) # h for help, s for save
    except getopt.GetoptError:
        print('Usage : ')
        print('run.py -p <path_to_cfg> -s')   
        sys.exit(2)

    if(len(opts)==0):
        print('Usage : ')
        print('run.py -p <path_to_cfg> -s')   
        sys.exit()

    if(opts[0][0] == '-h'):
        print('Usage : ')
        print('run.py -p <path_to_cfg> -s')   
        sys.exit()
    elif opts[0][0] in ('-p','--path'):
        # resolve allows to go in absolute path (and note relative anymore)
        mod_path = Path(__file__).resolve().parents[1]
        cfg_path = (mod_path/Path(opts[0][1])).resolve()
        if(opts[0][0] in ('-s','-save')):
            main(cfg_path, save = True)
        else :
            main(cfg_path, save = False)

    else :
        print('Please precise path to cfg.')
        print('Usage : ')
        print('run.py -p <path_to_cfg> -s')   
        sys.exit()

if __name__ == "__main__":
    run(sys.argv[1:]) # because the first arg is always the name of the file