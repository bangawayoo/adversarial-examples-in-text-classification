"""
From repository of "Mozes, Maximilian, et al. "Frequency-guided word substitutions for detecting textual adversarial examples." EACL (2021)."
"""

import logging
import sys
import csv
import os

class Logger:
    def __init__(self, log_path):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=f"{log_path}/out.log", level=logging.INFO, filemode="a")
        streamHandler = logging.StreamHandler(sys.stdout)
        self.log = logging.getLogger('log')
        self.log.addHandler(streamHandler)
        self.log_path = log_path
        self.seed = None
        self.metric = {}
        self.metric_names = ['tpr', 'fpr', 'f1', 'auc', 'topk', 'naive_tpr']

    def set_seed(self, seed):
        self.log.info(f"Seed={seed}")
        self.seed = seed

    def log_metric(self, metric_dict):
        for key, value in metric_dict.items():
            assert key in self.metric_names, f"Trying to log {key}: Not in {self.metric_names} "
            self.metric[key] = value

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

    def save_custom_metric(self, filename, metric, header):
        csv_path = os.path.join(self.log_path, f"{filename}.csv")
        exist_flag = os.path.isfile(csv_path)
        headers = ['seed'] + header
        with open(csv_path, 'a', newline='') as f :
            wr = csv.writer(f)
            if not exist_flag:
                wr.writerow(headers)
            data = metric
            data.insert(0, self.seed)
            wr.writerow(data)


