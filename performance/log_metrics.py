import time
import string
import sys
import subprocess

"""
Logs CPU, GPU, and memory usage about a process, very basic implementation
"""

GPU_PER = "Gpu"
GPU_MEM = "Used GPU Memory"

def format_bytes(bytes):
    if abs(bytes) < (1024):
        return str(bytes)+"B"
    elif abs(bytes) < (2 << 20):
        return str(round(bytes/1024,2)) + "kiB"
    elif abs(bytes) < (2 << 30):
        return str(round(bytes / (2 << 20), 2)) + "MiB"
    else:
        return str(round(bytes / (2<<20), 2)) + "GiB"

def get_cpumem(pid):
    process = subprocess.Popen(['ps', 'ux'], stdout=subprocess.PIPE)
    out, err = process.communicate()
    out = out.decode("utf-8")
    d = [i for i in out.split('\n') if len(i) > 0 and i.split()[1] == str(pid)]
    if d:
        return (float(d[0].split()[2]), float(d[0].split()[3]), format_bytes(int(d[0].split()[5])))
    else:
        return None

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

def log(pid, duration=1800, file=None):
    print("%CPU\t%MEM\tMEM\t%GPU\tGPU")
    try:
        while True:
            res = get_cpumem(pid)
            if res is None:
                print("no such process")
                exit(1)   
            cpu_per,mem_per,mem = res       
            gpu_per,gpu = get_gpumem(pid) 
            print("{}\t{}\t{}\t{}\t{}".format(cpu_per,mem_per,mem,gpu_per,gpu))
            time.sleep(duration)
    except KeyboardInterrupt:
        print()
        exit(0)