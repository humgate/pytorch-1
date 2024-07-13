from timeit import default_timer as timer


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start_timer(self):
        self.start_time = timer()

    def stop_timer(self):
        self.end_time = timer()

    def print_elapsed_time(self):
        if self.start_time is not None and self.end_time is not None:
            print(f"Elapsed time: {self.end_time-self.start_time:.3f} seconds")
