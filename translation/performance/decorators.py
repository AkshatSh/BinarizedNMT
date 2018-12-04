from multiprocessing import Process, Queue
import functools, time
from log_metrics import log

POLL_PERIOD = 'poll_period'
LOG_DIR = 'log_dir'

def profile(_func: callable = None, *, poll_period: int = 1800, log_dir: str = None) -> callable:
    def decorater_profile(_func):
        @functools.wraps(_func)
        def wrapper(*args, **kwargs):
            func_process = Process(target = _func, args = args, kwargs=kwargs)
            func_process.start()
            pid = func_process.pid
            log_process = Process(target = log, args = (pid,),
                                kwargs = {POLL_PERIOD: poll_period, LOG_DIR: log_dir})
            log_process.start()
            func_process.join()
            log_process.join()
        return wrapper

    if _func is None:
        return decorater_profile
    else:
        return decorater_profile(_func)