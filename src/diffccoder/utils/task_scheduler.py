from multiprocessing import Process
from multiprocessing import Event
from typing import Callable

from loguru import logger


class Scheduler(Process):
    def __init__(self, timeout: int, function: Callable, args=None, kwargs=None):
        super(Scheduler, self).__init__()
        self.timeout = timeout
        self.function = function
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self.finished_event = Event()

    def cancel(self):
        """Stop the timer if it hasn't finished yet"""
        logger.info('Closing submitted task...')
        self.finished_event.set()

    def run(self):
        self.finished_event.wait(self.timeout)
        if not self.finished_event.is_set():
            self.function(*self.args, **self.kwargs)
        self.finished_event.set()


class RepeatingScheduler(Scheduler):
    def __init__(self, interval: int, function: Callable):
        super(RepeatingScheduler, self).__init__(interval, function)

    def run(self):
        while not self.finished_event.is_set():
            # function callback could set the stop event
            self.function(*self.args, **self.kwargs)
            self.finished_event.wait(self.timeout)
