import sys
sys.path.append('..')
import time
import string
import sys
import subprocess
from tensor_logger import Logger

"""
Logs CPU, GPU, and memory usage about a process, very basic implementation
"""

GPU_PER = "Gpu"
GPU_MEM = "Used GPU Memory"

def get_cpumem(pid):
    try:
        process = subprocess.Popen(['ps', 'ux'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except e:
        return None
    out, err = process.communicate()
    if out is None:
        return None
    out = out.decode("utf-8")
    d = [i for i in out.split('\n') if len(i) > 0 and i.split()[1] == str(pid)]
    if d:
        return (float(d[0].split()[2]), float(d[0].split()[3]), int(d[0].split()[5]))
    else:
        return None

def get_gpumem(pid):
    try:
        process = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
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
    y = out_dict[GPU_MEM].replace(' ', '')  # need to convert this from whatever suffix to bytes
    return (x, y) 

def log(pid, poll_period=1800, log_dir=None):
    logger = None
    if log_dir is None:
        print("%CPU,%MEM,MEM,%GPU,GPU")
    else:
        logger = Logger(log_dir)
    try:
        while True:
            cpu_res = get_cpumem(pid)
            if cpu_res is None: 
                cpu_per,mem_per,mem = None, None, None
            else:
                cpu_per,mem_per,mem = cpu_res

            gpu_res =  get_gpumem(pid)
            if gpu_res is None:
                gpu_per,gpu = None, None
            else:
                gpu_per,gpu = gpu_res

            if gpu_res is None and cpu_res is None:
                exit(1)

            if logger is None:
               print("{},{},{},{},{}".format(cpu_per,mem_per,mem,gpu_per,gpu))
            else:
                timestamp = time.time()
                logger.scalar_summary(
                    "cpu_percentage",
                    cpu_per,
                    timestamp,
                )
                logger.scalar_summary(
                    "memory_percentage",
                    mem_per,
                    timestamp,
                )
                logger.scalar_summary(
                    "memory",
                    mem,
                    timestamp,
                )
                logger.scalar_summary(
                    "gpu_percentage",
                    gpu_per,
                    timestamp,
                )
                logger.scalar_summary(
                    "gpu",
                    gpu,
                    timestamp,
                )
            time.sleep(duration)
    except KeyboardInterrupt:
        print()
        exit(0)