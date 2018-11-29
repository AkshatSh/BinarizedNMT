import time
import string
import sys
import subprocess

"""
Logs CPU, GPU, and memory usage about a process, very basic implementation, need to put this in a wrapper later
"""

GPU_PER = "Gpu"
GPU_MEM = "Used GPU Memory"

def format_bytes(bytes):
    if abs(bytes) < 1000:
        return str(bytes)+"B"
    elif abs(bytes) < 1e6:
        return str(round(bytes/1e3,2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"

def get_cpumem(pid):
    process = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)
    out, err = process.communicate()
    out = out.decode("utf-8")
    d = [i for i in out.split('\n')
        if len(i) > 0 and i.split()[1] == str(pid)]
    return (float(d[0].split()[2]), float(d[0].split()[3]), format_bytes(int(d[0].split()[5]))) if d else None

def get_gpumem(pid):
    process = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    out = out.decode("utf-8")
    out_dict = {}
    for item in out.split('\n'):
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except:
            pass

    x = out_dict[GPU_PER].replace(' ', '')
    x = float(x.replace('%', ''))
    y = out_dict[GPU_MEM].replace(' ', '')  
    return (x, y) 

if __name__ == '__main__':
    if not len(sys.argv) == 2 or not all(i in string.digits for i in sys.argv[1]):
        print("usage: {} PID".format(sys.argv[0]))
        exit(2)
    print("%CPU\t%MEM\tMEM\t%GPU\tGPU")
    try:
        while True:
            x,y,z = get_cpumem(sys.argv[1])
            i,j = get_gpumem(sys.argv[1]) 
            if not x:
                print("no such process")
                exit(1)
            print("{}\t{}\t{}\t{}\t{}".format(x,y,z,i,j))
            time.sleep(10)
    except KeyboardInterrupt:
        print
        exit(0)
