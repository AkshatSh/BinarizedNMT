from multiprocessing import Process, Queue
import functools, time
from log_metrics import log

def profile(f: callable) -> callable:
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        f_process = Process(target = f, args = args, kwargs=kwargs)
        f_process.start()
        pid = f_process.pid
        log_process = Process(target = log, args = (pid,))
        log_process.start()
        f_process.join()
        log_process.join()
    return wrapper
