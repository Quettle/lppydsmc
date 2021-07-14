# local import
from .run import run

# python import
import sys, getopt
from pathlib import Path
# https://www.tutorialspoint.com/python/python_command_line_arguments.htm

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hp:", ['path=']) # h for help
    except getopt.GetoptError:
        print('Usage : ')
        print('main.py -p <path_to_cfg>')   
        sys.exit(2)

    if(len(opts)==0):
        print('Usage : ')
        print('main.py -p <path_to_cfg>')   
        sys.exit()

    if(opts[0][0] == '-h'):
        print('Usage : ')
        print('main.py -t <type> -p <path_to_cfg>')   
        sys.exit()
    elif opts[0][0] in ('-p','--path'):
        # resolve allows to go in absolute path (and note relative anymore)
        mod_path = Path(__file__).resolve().parents[1]
        cfg_path = (mod_path/Path(opts[0][1])).resolve()
        run(cfg_path)
    else :
        print('Please precise path to cfg.')
        print('Usage : ')
        print('main.py -t <type> -p <path_to_cfg>')   
        sys.exit()

if __name__ == "__main__":
    main(sys.argv[1:]) # because the first arg is always the name of the file