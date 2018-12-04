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
    try:
        process = subprocess.Popen(['ps', 'ux'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except e:
        #print(e.output)
        return None
    out, err = process.communicate()
    if out is None:
        return None
    out = out.decode("utf-8")
    d = [i for i in out.split('\n') if len(i) > 0 and i.split()[1] == str(pid)]
    if d:
        return (float(d[0].split()[2]), float(d[0].split()[3]), format_bytes(int(d[0].split()[5])))
    else:
        return None

def get_gpumem(pid):
    try:
        process = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        #print(e.output)
        return None
    out, err = process.communicate()
    if out is None:
        return None
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

def log(pid, poll_period=1800, file_path=None):
    if file_path is None:
        print("%CPU,%MEM,MEM,%GPU,GPU")
    else:
        f = open(file_path, "a+")
        f.write("%CPU,%MEM,MEM,%GPU,GPU\n")
    try:
        while True:
            cpu_res = get_cpumem(pid)
            if cpu_res is None:
                print("cpu information not found")  
                cpu_per,mem_per,mem = None, None, None
            else:
                cpu_per,mem_per,mem = cpu_res
            gpu_res =  get_gpumem(pid)
            if gpu_res is None:
                print("gpu information not found")
                gpu_per,gpu = None, None
            else:
                gpu_per,gpu = gpu_res
            if gpu_res is None and cpu_res is None:
                print("no such process")
                exit(1)    
            message = "{},{},{},{},{}\n".format(cpu_per,mem_per,mem,gpu_per,gpu)
            if file_path is None:
               print(message)
            else:
                f.write(message)
            time.sleep(duration)
    except KeyboardInterrupt:
        close(f)
        print()
        exit(0)