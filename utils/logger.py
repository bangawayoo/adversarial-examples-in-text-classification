"""
From repository of "Mozes, Maximilian, et al. "Frequency-guided word substitutions for detecting textual adversarial examples." EACL (2021)."
"""

import logging
import sys

class Logger:
    def __init__(self, log_path):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=f"{log_path}/out.log", level=logging.INFO, filemode="w")
        streamHandler = logging.StreamHandler(sys.stdout)
        self.log = logging.getLogger('log')
        self.log.addHandler(streamHandler)
        self.log_path = log_path


