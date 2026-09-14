# Note that this is a "static" constant class
from tools.distributed import Distributed_utils as du


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    # below are the ending sequence, either regular, bold, or underline
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Color_string:
    @staticmethod
    def OK(s):
        return bcolors.OKCYAN + s + bcolors.ENDC

    @staticmethod
    def warning(s):
        return bcolors.WARNING + s + bcolors.ENDC

    @staticmethod
    def fail(s):
        return bcolors.FAIL + s + bcolors.ENDC

    @staticmethod
    def header(s):
        return bcolors.HEADER + s + bcolors.ENDC


# the "real" class
class Printer:
    _tabs = 0

    @staticmethod
    def print_base(s):
        if du.is_master_proc():
            print("\t" * Printer._tabs + s)

    @staticmethod
    def print_OK(s):
        Printer.print_base(Color_string.OK(s))

    @staticmethod
    def print_WARNING(s):
        Printer.print_base(Color_string.warning(s))

    @staticmethod
    def print_FAIL(s):
        Printer.print_base(Color_string.fail(s))

    @staticmethod
    def print_HEADER(s):
        Printer.print_base(Color_string.header(s))

    @staticmethod
    def add_tab(times=1):
        if du.is_master_proc():
            Printer._tabs += times

    @staticmethod
    def remove_tab(times=1):
        if du.is_master_proc():
            Printer._tabs -= times
