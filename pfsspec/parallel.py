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
import os, sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import logging
import multiprocessing
from multiprocessing.queues import Queue
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

class IterableQueue():
    def __init__(self, queue, length):
        self.logger = logging.getLogger()
        self.queue = queue
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.length > 0:
            self.length -= 1
            o = self.queue.get()
            if isinstance(o, Exception):
                print(o, file=sys.stderr)
                self.logger.error(str(o))
                raise o
            else:
                return o
        else:
            raise StopIteration()
    
class SmartParallel():
    def __init__(self, initializer=None, verbose=False, parallel=True, threads=None):
        if threads is not None:
            self.cpus = threads
        elif 'SLURM_CPUS_PER_TASK' in os.environ:
            self.cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            self.cpus = multiprocessing.cpu_count() // 2
        
        self.logger = logging.getLogger()
        
        self.processes = []
        self.queue_in = None
        self.queue_out = None
        self.initializer = initializer
        self.verbose = verbose
        self.parallel = parallel

    def __enter__(self):
        if self.parallel:
            self.logger.debug("Starting parallel execution on {} CPUs.".format(self.cpus))
        else:
            self.logger.debug("Starting serial execution.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parallel:
            self.logger.debug("Joining worker processes.")
            for p in self.processes:
                p.join()
            self.logger.debug("Finished parallel execution.")
        else:
            self.logger.debug("Finished serial execution.")
        return False

    def __del__(self):
        pass

    @staticmethod
    def pool_worker(initializer, worker, queue_in, queue_out):
        if initializer is not None:
            initializer()
        while True:
            i = queue_in.get()
            if isinstance(i, StopIteration):
                return
            else:
                try:
                    o = worker(i)
                    queue_out.put(o)
                except Exception as e:
                    queue_out.put(e)

    def map(self, worker, items):
        if self.parallel:
            self.queue_in = multiprocessing.Queue()
            self.queue_out = multiprocessing.Queue()

            pool_size = self.cpus

            for i in range(pool_size):
                target = SmartParallel.pool_worker
                args = (self.initializer, worker, self.queue_in, self.queue_out)
                p = multiprocessing.Process(target=target, args=args)
                p.daemon = True
                p.start()
                self.processes.append(p)

            for i in items:
                self.queue_in.put(i)

            for i in range(len(self.processes)):
                self.queue_in.put(StopIteration())

            m = IterableQueue(self.queue_out, len(items))
        else:
            if self.initializer is not None:
                self.initializer()
            m = map(worker, items)

        if self.verbose:
            m = tqdm(m, total=len(items))

        return m