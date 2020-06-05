import ctypes
import ctypes.wintypes
import time


class QueryPerformanceCounter:
    def __init__(self):
        self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        self.starting_time = ctypes.wintypes.LARGE_INTEGER()
        self.ending_time = ctypes.wintypes.LARGE_INTEGER()
        self.elapsed_microseconds = ctypes.wintypes.LARGE_INTEGER()
        self.frequency = ctypes.wintypes.LARGE_INTEGER()

    def start(self):
        self.kernel32.QueryPerformanceFrequency(ctypes.byref(self.frequency))
        self.kernel32.QueryPerformanceCounter(ctypes.byref(self.starting_time))

    def end(self, program_name="This program"):
        self.kernel32.QueryPerformanceCounter(ctypes.byref(self.ending_time))

        self.elapsed_microseconds = self.ending_time.value - self.starting_time.value
        self.elapsed_microseconds *= 1000000
        self.elapsed_microseconds /= self.frequency.value

        print('{} elapsed time : {} ms'.format(program_name, self.elapsed_microseconds))


if __name__ == '__main__':
    performance_counter = QueryPerformanceCounter()

    performance_counter.start()
    # Activity to be timed, e.g.
    time.sleep(2)
    performance_counter.end()


