import time
import logging

class Timer:
    def __init__(self, message, level=logging.DEBUG, print_to_console=False, print_to_log=True):
        self.message = message
        self.level = level
        self.print_to_console = print_to_console
        self.print_to_log = print_to_log
        self.logger = logging.getLogger() if self.print_to_log else None

        self.start = time.perf_counter()

    def __enter__(self):
        self.banner()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stamp()

    def banner(self):
        if self.print_to_console:
            print(self.message)
        if self.logger:
            self.logger.log(self.level, self.message)

    def stamp(self):
        elapsed = time.perf_counter() - self.start
        msg = '... done in {} sec.'.format(elapsed)
        if self.print_to_console:
            print(msg)
        if self.logger:
            self.logger.log(self.level, msg)
