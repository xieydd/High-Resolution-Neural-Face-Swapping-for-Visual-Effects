import test
import argparse
import re
import sys
import numpy as np

def main():
        sep_np_list = np.array(sys.argv[1:])
        float_sep_list = sep_np_list.astype(float)
        print(np.std(float_sep_list))



if __name__ == '__main__':
        main()


