import re
from spacy.lang.en import English

spacy_tokenizer = English().tokenizer

def clean_str(string, tokenizer=spacy_tokenizer):
    """
    Parts adapted from https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/mydatasets.py
    """
    assert isinstance(string, str)
    string = string.replace("<br />", "")
    string = re.sub(r"[^a-zA-Z0-9.]+", " ", string)

    return (
        string.strip().lower().split()
        if tokenizer is None
        else [t.text.lower() for t in tokenizer(string.strip())]
    )

def pad(max_len, seq, token):
	assert isinstance(seq, list)
	abs_len = len(seq)
	if abs_len > max_len:
		seq = seq[:max_len]
	else:
		seq += [token] * (max_len - abs_len)
	return seq

def fgws_preprocess(sentence):
	sentence = clean_str(sentence)
	sentence = pad(200, sentence, "<pad>")
	sentence = ' '.join(sentence)
	return sentence