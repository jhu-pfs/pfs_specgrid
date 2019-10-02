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
import logging
import multiprocessing
import numpy as np
from tqdm import tqdm

def apply_function(func_to_apply, queue_in, queue_out):
    logger = multiprocessing.log_to_stderr()
    #logger.setLevel(logging.DEBUG)
    np.random.seed()
    while not queue_in.empty():
        num, obj = queue_in.get()
        queue_out.put((num, func_to_apply(obj)))

def prll_map(func_to_apply, items, cpus=None, verbose=False):
    # Number of processes to use #
    if cpus is None: cpus = min(multiprocessing.cpu_count(), 32)
    # Create queues #
    q_in  = multiprocessing.Queue()
    q_out = multiprocessing.Queue()
    # Process list #
    new_proc  = lambda t, a: multiprocessing.Process(target=t, args=a)
    processes = [new_proc(apply_function, (func_to_apply, q_in, q_out)) for x in range(cpus)]
    # Put all the items (objects) in the queue #
    sent = [q_in.put((i, x)) for i, x in enumerate(items)]
    # Start them all #
    for proc in processes:
        proc.daemon = True
        proc.start()
    # Display progress bar or not #
    if verbose:
        results = [q_out.get() for x in tqdm(range(len(sent)))]
    else:
        results = [q_out.get() for x in range(len(sent))]

    # Wait for them to finish #
    #for proc in processes: proc.join()

    # Return results #
    return [x for i, x in sorted(results)]

def srl_map(func_to_apply, items, verbose=False):
    results = []
    if verbose:
        for i in tqdm(items):
            results.append(func_to_apply(i))
    else:
        for i in items:
            results.append(func_to_apply(i))
    return results