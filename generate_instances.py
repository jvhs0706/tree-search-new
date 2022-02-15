import os
import argparse
import numpy as np
import scipy.sparse
from itertools import combinations

from benchmarks.setcover import *
from benchmarks.indset import *
from benchmarks.cauctions import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    if args.problem == 'setcover':
        nrows = 500
        ncols = 1000
        dens = 0.05
        max_coef = 100

        filenames = []
        nrowss = []
        ncolss = []
        denss = []

        # train instances
        try:
            n = 10000
            lp_dir = f'data/instances/setcover/train_{nrows}r_{ncols}c_{dens}d'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nrowss.extend([nrows] * n)
            ncolss.extend([ncols] * n)
            denss.extend([dens] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # validation instances
        try:
            n = 2000
            lp_dir = f'data/instances/setcover/valid_{nrows}r_{ncols}c_{dens}d'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nrowss.extend([nrows] * n)
            ncolss.extend([ncols] * n)
            denss.extend([dens] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # medium transfer instances
        try:
            n = 100
            nrows = 1000
            lp_dir = f'data/instances/setcover/transfer_{nrows}r_{ncols}c_{dens}d'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nrowss.extend([nrows] * n)
            ncolss.extend([ncols] * n)
            denss.extend([dens] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # big transfer instances
        try:
            n = 100
            nrows = 2000
            lp_dir = f'data/instances/setcover/transfer_{nrows}r_{ncols}c_{dens}d'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nrowss.extend([nrows] * n)
            ncolss.extend([ncols] * n)
            denss.extend([dens] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # test instances
        try:
            n = 2000
            nrows = 500
            ncols = 1000
            lp_dir = f'data/instances/setcover/test_{nrows}r_{ncols}c_{dens}d'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nrowss.extend([nrows] * n)
            ncolss.extend([ncols] * n)
            denss.extend([dens] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # medium test instances
        try:
            n = 100
            nrows = 1000
            lp_dir = f'data/instances/setcover/test_{nrows}r_{ncols}c_{dens}d'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nrowss.extend([nrows] * n)
            ncolss.extend([ncols] * n)
            denss.extend([dens] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # big test instances
        try:
            n = 100
            nrows = 2000
            lp_dir = f'data/instances/setcover/test_{nrows}r_{ncols}c_{dens}d'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nrowss.extend([nrows] * n)
            ncolss.extend([ncols] * n)
            denss.extend([dens] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # actually generate the instances
        for filename, nrows, ncols, dens in zip(filenames, nrowss, ncolss, denss):
            print(f'  generating file {filename} ...')
            generate_setcover(nrows=nrows, ncols=ncols, density=dens, filename=filename, rng=rng, max_coef=max_coef)

        print('done.')

    elif args.problem == 'indset':
        number_of_nodes = 500
        affinity = 4

        filenames = []
        nnodess = []

        # train instances
        try:
            n = 10000
            lp_dir = f'data/instances/indset/train_{number_of_nodes}_{affinity}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nnodess.extend([number_of_nodes] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # validation instances
        try:
            n = 2000
            lp_dir = f'data/instances/indset/valid_{number_of_nodes}_{affinity}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nnodess.extend([number_of_nodes] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # medium transfer instances
        try:
            n = 100
            number_of_nodes = 1000
            lp_dir = f'data/instances/indset/transfer_{number_of_nodes}_{affinity}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nnodess.extend([number_of_nodes] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # big transfer instances
        try:
            n = 100
            number_of_nodes = 1500
            lp_dir = f'data/instances/indset/transfer_{number_of_nodes}_{affinity}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nnodess.extend([number_of_nodes] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')


        # test instances
        try:
            n = 2000
            number_of_nodes = 500
            lp_dir = f'data/instances/indset/test_{number_of_nodes}_{affinity}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nnodess.extend([number_of_nodes] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # medium test instances
        try:
            n = 100
            number_of_nodes = 1000
            lp_dir = f'data/instances/indset/test_{number_of_nodes}_{affinity}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nnodess.extend([number_of_nodes] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # big test instances
        try:
            n = 100
            number_of_nodes = 1500
            lp_dir = f'data/instances/indset/test_{number_of_nodes}_{affinity}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nnodess.extend([number_of_nodes] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # actually generate the instances
        for filename, nnodes in zip(filenames, nnodess):
            print(f"  generating file {filename} ...")
            graph = Graph.barabasi_albert(nnodes, affinity, rng)
            generate_indset(graph, filename)

        print("done.")

    elif args.problem == 'cauctions':
        number_of_items = 100
        number_of_bids = 500
        filenames = []
        nitemss = []
        nbidss = []

        # train instances
        try:
            n = 10000
            lp_dir = f'data/instances/cauctions/train_{number_of_items}_{number_of_bids}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nitemss.extend([number_of_items] * n)
            nbidss.extend([number_of_bids ] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # validation instances
        try:
            n = 2000
            lp_dir = f'data/instances/cauctions/valid_{number_of_items}_{number_of_bids}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nitemss.extend([number_of_items] * n)
            nbidss.extend([number_of_bids ] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # medium transfer instances
        try:
            n = 100
            number_of_items = 200
            number_of_bids = 1000
            lp_dir = f'data/instances/cauctions/transfer_{number_of_items}_{number_of_bids}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nitemss.extend([number_of_items] * n)
            nbidss.extend([number_of_bids ] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # big transfer instances
        try:
            n = 100
            number_of_items = 300
            number_of_bids = 1500
            lp_dir = f'data/instances/cauctions/transfer_{number_of_items}_{number_of_bids}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nitemss.extend([number_of_items] * n)
            nbidss.extend([number_of_bids ] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # test instances
        try:
            n = 2000
            number_of_items = 100
            number_of_bids = 500
            lp_dir = f'data/instances/cauctions/test_{number_of_items}_{number_of_bids}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nitemss.extend([number_of_items] * n)
            nbidss.extend([number_of_bids ] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # medium test instances
        try:
            n = 100
            number_of_items = 200
            number_of_bids = 1000
            lp_dir = f'data/instances/cauctions/test_{number_of_items}_{number_of_bids}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nitemss.extend([number_of_items] * n)
            nbidss.extend([number_of_bids ] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # big test instances
        try:
            n = 100
            number_of_items = 300
            number_of_bids = 1500
            lp_dir = f'data/instances/cauctions/test_{number_of_items}_{number_of_bids}'
            print(f"{n} instances in {lp_dir}")
            os.makedirs(lp_dir)
            filenames.extend([f'{lp_dir}/instance_{i+1}.lp' for i in range(n)])
            nitemss.extend([number_of_items] * n)
            nbidss.extend([number_of_bids ] * n)
        except FileExistsError:
            print(f'{lp_dir} already exists!')

        # actually generate the instances
        for filename, nitems, nbids in zip(filenames, nitemss, nbidss):
            print(f"  generating file {filename} ...")
            generate_cauctions(rng, filename, n_items=nitems, n_bids=nbids, add_item_prob=0.7)

        print("done.")

    