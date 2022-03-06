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
    parser.add_argument(
        '-rmk', '--remarks',
        help = 'The other remarks you want to add in the file name.',
        nargs = '*'
    )
    args = parser.parse_args()
    model_dir = f'models/'+ '_'.join([args.problem] + args.problem_params)
    if not os.path.isdir('models/archive'):
        os.makedirs('models/archive')
    shutil.make_archive('models/archive/' + '_'.join([args.problem] + args.problem_params) + ' ' + datetime.now().strftime('%Y-%m-%d %H%M%S') + (' '+'_'.join(args.remarks) if args.remarks is not None else ''), 'zip', model_dir)