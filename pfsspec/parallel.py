# from: https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class

"""
The ``processes`` module provides some convenience functions
for using parallel processes in python.

Adapted from http://stackoverflow.com/a/16071616/287297

Example usage:

    print prll_map(lambda i: i * 2, [1, 2, 3, 4, 6, 7, 8], 32, verbose=True)

Comments:

"It spawns a predefined amount of workers and only iterates through the input list
 if there exists an idle worker. I also enabled the "daemon" mode for the workers so
 that KeyboardInterupt works as expected."

Pitfalls: all the stdouts are sent back to the parent stdout, intertwined.

Alternatively, use this fork of multiprocessing:
https://github.com/uqfoundation/multiprocess
"""

# Modules #
from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing
import numpy as np
from tqdm import tqdm

def apply_function(x, init_func, worker_func, queue_in, queue_out):
    logger = multiprocessing.log_to_stderr()
    #logger.setLevel(logging.DEBUG)
    np.random.seed()
    if init_func is not None:
        init_func(x)
    while not queue_in.empty():
        num, obj = queue_in.get()
        queue_out.put((num, worker_func(obj)))

def prll_map(init_func, worker_func, items, cpus=None, verbose=False):
    # Number of processes to use #
    if cpus is None: cpus = min(multiprocessing.cpu_count(), 32)
    # Create queues #
    q_in  = multiprocessing.Queue()
    q_out = multiprocessing.Queue()
    # Process list #
    new_proc  = lambda t, a: multiprocessing.Process(target=t, args=a)
    processes = [new_proc(apply_function, (x, init_func, worker_func, q_in, q_out)) for x in range(cpus)]
    # Put all the items (objects) in the queue #
    sent = [q_in.put((i, x)) for i, x in enumerate(items)]
    # Start them all #
    for proc in processes:
        proc.daemon = True
        proc.start()

    # Display progress bar or not
    if verbose:
        return q_out, tqdm(range(len(sent)))
    else:
        return q_out, range(len(sent))

    #if verbose:
    #    results = [q_out.get() for x in tqdm(range(len(sent)))]
    #else:
    #    results = [q_out.get() for x in range(len(sent))]

    # Wait for them to finish #
    #for proc in processes: proc.join()

    # Return results #
    #return [x for i, x in sorted(results)]

def srl_map(init_func, worker_func, items, verbose=False):
    if init_func is not None:
        init_func(0)

    results = []
    if verbose:
        for i in tqdm(items):
            results.append(worker_func(i))
    else:
        for i in items:
            results.append(worker_func(i))
    return results

class SmartParallel():
    def __init__(self, initializer=None, verbose=False, parallel=True):
        self.pool = None
        self.initializer = initializer
        self.verbose = verbose
        self.parallel = parallel

    def __enter__(self):
        if self.parallel:
            logging.info("Starting parallel execution.")
        else:
            logging.info("Starting serial execution.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool is not None:
            self.pool.shutdown()
        if self.parallel:
            logging.info("Finished parallel execution.")
        else:
            logging.info("Finished serial execution.")
        return False

    def __del__(self):
        pass

    def map(self, worker, items):
        if self.parallel:
            self.pool = ProcessPoolExecutor(initializer=self.initializer)
            m = self.pool.map(worker, items)
        else:
            self.pool = None
            if self.initializer is not None:
                self.initializer()
            m = map(worker, items)
        if self.verbose:
            m = tqdm(m, total=len(items))
        return m