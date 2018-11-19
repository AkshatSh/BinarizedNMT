import argparse
import os
import mmap
from tqdm import tqdm
import csv

from utils import get_num_lines


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Shard a large file.')

    # Parser data loader options
    parser.add_argument('--shard_size', type=int, default=100000, help='the number of sentences in each shard')
    parser.add_argument('--output_dir', type=str, default=None, help='the name of the output directory')
    parser.add_argument('--src_file', type=str, required=True, help='the name of the input file to be sharded')
    parser.add_argument('--target_file', type=str, required=True, help='the name of the translated file to be sharded')
    parser.add_argument('--output_prefix', type=str, default='shard', help='the prefix of each shard file')

    return parser

def get_shard_file_name(output_dir: str, prefix: str, shard_idx: int) -> str:
    return os.path.join(
        output_dir,
        "{}_{}.shard".format(prefix, shard_idx)
    )

def shard_file(
    src_filename: str,
    target_filename: str,
    output_dir: str,
    prefix: str,
    shard_size: int,
) -> list:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shard_idx = 0
    total_count = 0
    result_shards = []
    curr_file = None
    csv_writer = None
    with open(src_filename) as src_file, open(target_filename) as trg_file:

        src_total_lines = get_num_lines(src_filename)
        targ_total_lines = get_num_lines(target_filename)

        print(
            "Src lines: {}\nTarg Lines: {}".format(
                src_total_lines,
                targ_total_lines,
            )
        )

        # must match up exactly
        assert(src_total_lines == targ_total_lines)

        for src, trg in tqdm(zip(src_file, trg_file), total=src_total_lines):
            if total_count % shard_size == 0:
                # completed shard
                result_shards.append(curr_file.name) if curr_file is not None else None
                curr_file.close() if curr_file is not None else None
                shard_idx += 1
                curr_file = open(get_shard_file_name(output_dir, prefix, shard_idx), 'w')
                csv_writer = csv.writer(curr_file, delimiter=',')
            
            # remove whitespace
            src = src.strip()
            trg = trg.strip()
            if src and trg:
                csv_writer.writerow([src, trg])
            total_count += 1
    
    curr_file.close() if curr_file is not None else None

    print("Finished")
    print(
        "Processed {line} lines \n with {shard_size} shards".format(
            line=total_count,
            shard_size=len(result_shards)
        )
    )

    return result_shards

def main():
    args = get_args().parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.src_file + "_shard"

    shard_file(
        src_filename=args.src_file,
        target_filename=args.target_file,
        output_dir=output_dir,
        prefix=args.output_prefix,
        shard_size=args.shard_size,
    )

if __name__ == "__main__":
    main()