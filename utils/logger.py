"""
From repository of "Mozes, Maximilian, et al. "Frequency-guided word substitutions for detecting textual adversarial examples." EACL (2021)."
"""

import logging
import sys
import csv
import os

class Logger:
    def __init__(self, log_path, seed):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=f"{log_path}/out.log", level=logging.INFO, filemode="w")
        streamHandler = logging.StreamHandler(sys.stdout)
        self.log = logging.getLogger('log')
        self.log.addHandler(streamHandler)
        self.log.info(f"Seed={seed}")
        self.log_path = log_path
        self.seed=seed
        self.metric = {}
        self.metric_names = ['tpr', 'fpr', 'f1', 'auc']

    def log_metric(self, *args):
        assert len(args) == 4, "Check if four metrics were given"
        for v, k in zip(args, self.metric_names):
            self.metric[k] = v

    def save_metric(self):
        csv_path = os.path.join(self.log_path, "results.csv")
        exist_flag = os.path.isfile(csv_path)
        headers = ['seed'] + self.metric_names
        with open(csv_path, 'a', newline='') as f :
            wr = csv.writer(f)
            if not exist_flag:
                wr.writerow(headers)
            data = [self.metric[self.metric_names[i]] for i in range(len(self.metric_names))]
            data.insert(0, self.seed)
            wr.writerow(data)


