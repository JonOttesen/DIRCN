from pathlib import Path
import logging

save_path = './docs/'

Path(save_path).mkdir(parents=True, exist_ok=True)


logging.basicConfig(filename=save_path + 'training.log')

levels = {0: logging.NOTSET,
          1: logging.DEBUG,
          2: logging.INFO,
          3: logging.WARNING,
          4: logging.ERROR,
          5: logging.CRITICAL
        }

def get_logger(name, level=2):
    if not isinstance(level, int):
        raise TypeError('level must be of type int not {}'.format(type(level)))
    elif level not in levels.keys():
        raise ValueError('level must be {}, not {}'.format(levels.keys(), level))

    logger = logging.getLogger(name)
    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(levels[level])

    return logger
