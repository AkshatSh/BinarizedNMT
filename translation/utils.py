import os
import mmap

def get_num_lines(file_path: str) -> int:
    '''
    returns the number of lines in a file
    '''
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines