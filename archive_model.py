import shutil
import argparse
import os

from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help = 'MILP instance type to process.',
        choices = ['setcover', 'cauctions', 'indset'],
    )
    parser.add_argument(
        'problem_params', 
        help = 'Problem parameters to identify the instances.',
        nargs = '*'
    )
    args = parser.parse_args()
    model_dir = f'models/'+ '_'.join([args.problem] + args.problem_params)
    shutil.make_archive(model_dir + ' ' + datetime.now().strftime('%Y-%m-%d %H%M%S'), 'zip', model_dir)