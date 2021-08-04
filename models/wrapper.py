"""
Based on repository of "Mozes, Maximilian, et al. "Frequency-guided word substitutions for detecting textual adversarial examples." EACL (2021)."
Parts based on https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX
"""
from utils.preprocess import *

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch

class BertWrapper:
    def __init__(self, config, logger):
        self.config = config
        num_classes = {"imdb":2, "sst2":2, "agnews":4}
        use_text_attack = True if 'textattack' in config.target_model else False
        logger.log.info(f"Loading {config.target_model}")
        if use_text_attack:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.target_model,
                num_labels=num_classes[config.dataset],
                output_attentions=False,
                output_hidden_states=False)
            self.tokenizer = AutoTokenizer.from_pretrained(config.target_model, use_fast=True)

        else:
            ckpt_path = config.target_model

            self.tokenizer = RobertaTokenizer.from_pretrained(
                "roberta-base", do_lower_case=True)

            self.model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=num_classes[config.dataset],
                output_attentions=False,
                output_hidden_states=False,
            )
            checkpoint = torch.load(ckpt_path)["model_state_dict"]
            self.model.load_state_dict(checkpoint)

        if len(config.gpu) > 1 :
            device = [int(gpu_id) for gpu_id in config.gpu.split()]
            self.model = torch.nn.DataParallel(self.model, device_ids=device)
        elif len(config.gpu) == 1:
            device = torch.device(int(config.gpu))
            self.model = self.model.to(device)

    def __pre_process(self, text):
        assert isinstance(text, list)
        if isinstance(text[0], list):
            text = [" ".join(t) for t in text]

        if self.config.preprocess == 'fgws':
            text = [fgws_preprocess(t) for t in text]
        elif self.config.preprocess == 'standard':
            pass

        return text

    def inference(self, text, output_hs=False, output_attention=False):
        text = self.__pre_process(text)
        x = self.tokenizer(text, max_length=256, add_special_tokens=True, padding=True, truncation=True,
                      return_attention_mask=True, return_tensors='pt')
        output = self.model(input_ids=x['input_ids'].to(self.model.device), attention_mask=x['attention_mask'].to(self.model.device),
                       token_type_ids=(x['token_type_ids'].to(self.model.device) if 'token_type_ids' in x else None),
                       output_hidden_states=output_hs, output_attentions=output_attention)
        return output

    def get_att_mask(self, text):
        text = self.__pre_process(text)
        x = self.tokenizer(text, max_length=256, add_special_tokens=True, padding=True, truncation=True,
                      return_attention_mask=True, return_tensors='pt')
        return x['attention_mask'].to(self.model.device)