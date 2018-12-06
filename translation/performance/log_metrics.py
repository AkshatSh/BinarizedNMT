import sys
sys.path.append('..')
import time
import string
import sys
import subprocess
import re
from tensor_logger import Logger

"""
Logs CPU, GPU, and memory usage about a process, very basic implementation
"""

GPU_PER = "Gpu"
GPU_MEM = "Used GPU Memory"

def get_cpumem(pid):
    try:
        process = subprocess.Popen(['ps', 'ux'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except e:
        return None
    out, err = process.communicate()
    if out is None:
        return None
    out = out.decode("utf-8")
    d = [i for i in out.split('\n') if len(i) > 0 and i.split()[1] == str(pid)]
    if d:
        # (CPU%, MEM%, MEM)
        # MEM is in term of Kilobytes (2^10)
        return (float(d[0].split()[2]), float(d[0].split()[3]), int(d[0].split()[5]) * 1000)
    else:
        return None

def get_gpumem(pid):
    try:
        process = subprocess.Popen(['gpustat', '-cp'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except:
        return None
    out, err = process.communicate()
    if out is None:
        return None
    out = out.decode("utf-8")
    pid_tag = "python/" + str(pid)
    gpu_per, gpu = None, None
    for item in out.split('\n'):
        if pid_tag in item:
            try:
                gpu_per = [int(s) for s in item.split("|")[1].split(",")[1].split() if s.isdigit()][0]
                gpu = (re.search(r'\((.*?)\)', item).group(1))
            except:
                pass
    return (gpu_per, int(gpu[:len(gpu)-1])*(2**20)) # GPU memory is in term of Mibibytes (2^20)

def log(pid, poll_period=1800, log_dir=None):
    logger = None
    if log_dir is None:
        print("%CPU,%MEM,MEM,%GPU,GPU")
    else:
        logger = Logger(log_dir)
    try:
        while True:
            # Get CPU and MEM stats
            cpu_res = get_cpumem(pid)
            if cpu_res is None: 
                cpu_per,mem_per,mem = 0, 0, 0
            else:
                cpu_per,mem_per,mem = cpu_res
            # Get GPU stats
            gpu_res =  get_gpumem(pid)
            if gpu_res is None:
                gpu_per,gpu = 0, 0
            else:
                gpu_per,gpu = gpu_res
            # Exit if nothing is gotten
            if gpu_res is None and cpu_res is None:
                exit(1)

            # log stats
            if logger is None:
               print("{},{},{},{},{}\n".format(cpu_per,mem_per,mem,gpu_per,gpu))
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
            time.sleep(poll_period)
    except KeyboardInterrupt:
        print()
        exit(0)