"""
Implement some auxiliary routines to create/load the different loggers.
"""

from colorama import Fore, Back, Style, init
import logging


# Prepare the colored formatter
init(autoreset=True)
terminal_colors = {"DEBUG": Fore.BLUE,  # "INFO": Fore.CYAN,
                   "WARNING": Fore.YELLOW, "ERROR": Fore.RED,
                   "CRITICAL": Fore.WHITE + Back.RED + Style.BRIGHT}
html_colors = {"DEBUG": 'BLUE',  # "INFO": Fore.CYAN,
               "WARNING": 'YELLOW', "ERROR": "ORANGE",
               "CRITICAL": "RED"}


class SingletonMeta(type):
    """
    Implementation of the Singleton pattern as Python metaclass.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                SingletonMeta, cls).__call__(*args, **kwargs)

        return cls._instances[cls]
        # else:
        # to run __init__ every time the class is called...
        #     cls._instances[cls].__init__(*args, **kwargs)


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if record.levelname in terminal_colors:
            msg = terminal_colors[record.levelname] + msg + \
                  Fore.RESET + Back.RESET + Style.RESET_ALL
        return msg


class Logger(logging.Logger, metaclass=SingletonMeta):
    def __init__(self):
        fmt: str = "%(asctime)s %(levelname)8s >> %(message)s"
        # fmt: str = "%(asctime)s %(levelname)8s: %(name)18s >> %(message)s"
        date_format: str = "%m/%d/%Y %I:%M:%S %p"
        colored_formatter = ColoredFormatter(fmt, datefmt=date_format)
        bw_formatter = logging.Formatter(fmt, datefmt=date_format)

        # now we set the main / base logger called 'parent'
        super().__init__('parent', logging.DEBUG)

        # now we add a handler to print all the debug in the terminal
        from sys import stdout
        handler = logging.StreamHandler(stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(colored_formatter)
        self.addHandler(handler)

        # & finally another one where to save all the info in a file...
        f_handler = logging.FileHandler('summary.log')
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(bw_formatter)
        self.addHandler(f_handler)

    def _log(self, level: int, msg: object, args, **kwargs) -> None:
        """
        Log a message at a given level and always including the module's name
        from where the call was made.
        """
        # module_name = sys.modules[__name__].__spec__.name
        # msg = f'{self.module_name}: {msg}'
        logging.Logger.__dict__['_log'](self, level, msg, args, **kwargs)
