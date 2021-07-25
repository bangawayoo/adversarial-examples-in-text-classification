"""
Word Swap by Gradient
============================================

"""
import os
import sys
sys.path.append('/')

import torch
import transformers

import textattack
from textattack.shared import utils, WordEmbedding
from textattack.shared.validators import validate_model_gradient_word_swap_compatibility
from textattack.transformations.word_swaps import WordSwap

from copy import deepcopy
import pdb
from models.decoder import Decoder

class PSGradientBased(WordSwap):
    """
    Arguments:
        model (nn.Module): The model to attack. Model must have a
            `word_embeddings` matrix and `convert_id_to_word` function.
        top_n (int): the number of top words to return at each index
        iterations (int) : the number of backprop iterations

    Returns:
        list of all possible transformations for `text`
    """

    def __init__(self, model_wrapper, top_n=1, iterations=100, lr=1e-1):
        # Unwrap model wrappers. Need raw model for gradient.
        if not isinstance(model_wrapper, textattack.models.wrappers.ModelWrapper):
            raise TypeError(f"Got invalid model wrapper type {type(model_wrapper)}")
        self.model = model_wrapper.model
        self.model_for_bp = deepcopy(model_wrapper.model)
        self.model_wrapper = model_wrapper
        self.tokenizer = self.model_wrapper.tokenizer # AutoTokenizer from TextAttack
        self.syn_embedding = WordEmbedding.counterfitted_GLOVE_embedding()
        self.syn_top_n = 1
        self.iterations = iterations
        self.lr = lr
        self.verbose = True
        self.latent_emb = torch.load("./ckpt/latent-ep50.emb").cuda()
        self.decoder = Decoder(3, 128, 768, 768).cuda()
        self.decoder.load_state_dict(torch.load("./ckpt/decoder-ep50.pt"))
        self.latent_dim = self.latent_emb.embedding_dim
        self.target_dim = self.model.get_input_embeddings().embedding_dim

        # Make sure this model has all of the required properties.
        if not hasattr(self.model, "get_input_embeddings"):
            raise ValueError(
                "Model needs word embedding matrix for gradient-based word swap"
            )
        if not hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id:
            raise ValueError(
                "Tokenizer needs to have `pad_token_id` for gradient-based word swap"
            )

        self.top_n = top_n
        self.is_black_box = False

    def _get_replacement_words(self, word, word_idx, text_input):
        # Find list of synonym words using HowNet/ WordNet/ GloveEmbedding and their embedding
        candidates = []
        # Tokenize current word
        if word_idx != 0 and isinstance(self.tokenizer.tokenizer, transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast): # If word is not the first word of sentence
            word2tokenize = " " + word # Add space for Roberta tokenizer
        else :
            word2tokenize = word

        word_token = self.tokenizer.tokenizer.tokenize(word2tokenize) # list of tokens

        # If word is tokenized into more than two tokens, skip
        if len(word_token) > 1 :
            return []
        word_token = word_token[0]

        embedding_layer = self.model_for_bp.get_input_embeddings()
        try :
            word_id = self.syn_embedding.word2index(word.lower())
        except KeyError:
            return []

        nnids = self.syn_embedding.nearest_neighbours(word_id, self.syn_top_n)
        nbr_word = [self.syn_embedding.index2word(id_) for id_ in nnids]
        emb_id = [self.tokenizer.encode(w)['input_ids'][1] for w in nbr_word] # Not handles problem with word with multiple tokens
        candidate_emb = self.model.get_input_embeddings().weight.detach()[emb_id,:] #Tensor of shape (# of candidates, hidden_dim)

        str_input = ' '.join(text_input)
        input_tokens = self.tokenizer.tokenizer.tokenize(str_input)
        input_ids = self.tokenizer.encode(str_input)['input_ids']
        # Get copy of the original embedding
        emb_copy = deepcopy(self.model_for_bp.get_input_embeddings())
        # Select embedding for forwarding. This is the embedding of the original input that will be modified.
        emb_forward = emb_copy.weight[input_ids, :].detach().unsqueeze(0)  # Shape : (1, Sequence_length , hidden_dim)
        emb_forward.requires_grad = True

        #Find word_idx in input_ids (original word_idx is index in the sentence)
        word_indices = [idx for idx, val in enumerate(text_input) if val == word] # Find indices of all 'word' in sentence
        word_idx = word_indices.index(word_idx) # Find which 'word' out of multiple words. If only single 'word' is in sentence, this will be 0
        word_idx = [idx for idx, val in enumerate(input_tokens) if val == word_token][word_idx] + 1
        if word_idx > 255:
            return []

        # For comparison log the original word embedding
        word_emb = emb_forward[0, word_idx:word_idx + 1, :].clone()
        original = emb_forward[0, word_idx:word_idx+1,:].clone()
        word_id = self.tokenizer.encode(word)['input_ids'][1]
        # print(f"{word} {word_emb.shape}")
        # if word_emb.shape[0] == 0:
        #     print(text_input)
        #     pdb.set_trace()

        # Toy example
        # error = torch.rand_like(word_emb)
        # error = torch.where(error > 0.5, 1, -1)
        # error = error * 0.001
        # top_candidate_emb = word_emb + error
        with torch.no_grad():
            predictions = self.model_for_bp(inputs_embeds=emb_forward).logits
            labels = predictions.argmax(dim=1)

        for n_iter in range(self.iterations):
            euclidean_distance = torch.norm(candidate_emb - word_emb, p=2, dim=1)
            nn_word_idx = torch.argsort(euclidean_distance, descending=False)
            nn_word_idx = [emb_id[i] for i in nn_word_idx.tolist()]
            adv_word = self.tokenizer.convert_ids_to_tokens(nn_word_idx[0])

            dist_to_original = torch.norm(original-emb_forward[0, word_idx:word_idx+1,:], p=2)
            dist_to_closest = torch.norm(embedding_layer.weight[nn_word_idx[0]] - word_emb, p=2)
            distance_to_all = torch.norm(embedding_layer.weight - word_emb, p=2, dim=1).mean()

            if self.verbose:
                print(f"ITER{n_iter} : {euclidean_distance.mean().item()}")
                print(f"  {word.lower()} --> {adv_word}")
                print(f"  dist. to closest: {dist_to_closest.item()}")
                print(f"  dist to original emb. {dist_to_original}")
                print(f"  dist. to all: {distance_to_all.item()}")
                if distance_to_all > 10 :
                    pdb.set_trace()

            # pdb.set_trace()
            self.model_for_bp.eval()
            self.model_for_bp.zero_grad()
            emb_copy.zero_grad()
            emb_forward.retain_grad()

            loss = self.model_for_bp(inputs_embeds=emb_forward, labels=labels)[0]
            loss.backward()

            with torch.no_grad():
                euclidean_distance = torch.norm(candidate_emb - word_emb, p=2, dim=1)
                nn_word_idx = torch.argsort(euclidean_distance, descending=False)
                top_n = 1
                top_candidate_emb = candidate_emb[nn_word_idx[:top_n], : ]
                if top_candidate_emb.dim() == 1:
                    top_candidate_emb.unsqueeze_(0)

                # grad w.r.t to word embeddings
                grads = emb_forward.grad[0, word_idx:word_idx + 1, :]  # gradient of current word ; Shape (1, 768)

                #Scale grad here#
                inverse_scale = torch.mean(torch.abs(top_candidate_emb - word_emb), dim=0)
                scale = inverse_scale
                if self.verbose:
                    print(f"  loss: {loss.item()}")
                    print(f"  grad: {grads.norm(p=2)}")
                    print(f"  scale : {scale.mean()}")

                # Scale proportionally to the distance from the candidate words
                grads *= scale
                #Update word embedding here according to gradient
                word_emb =  word_emb + self.lr * grads
                emb_forward[0, word_idx:word_idx + 1, :] = word_emb

            del emb_forward.grad  # Delete gradients so as to prevent accumulation

            # pdb.set_trace()
        # Find nearest neighbor in the embedding
        euclidean_distance = torch.norm(candidate_emb - word_emb, p=2, dim=1)
        nn_word_idx = torch.argsort(euclidean_distance, descending=False)
        nn_word_idx = [emb_id[i] for i in nn_word_idx.tolist()]
        adv_word = self.tokenizer.convert_ids_to_tokens(nn_word_idx[0])
        # print(f"{word.lower()} --> {adv_word}")
        candidates.extend(self.tokenizer.convert_ids_to_tokens(nn_word_idx))
        distance_with_all = torch.norm(embedding_layer.weight - word_emb, p=2, dim=1)

        return candidates

    def _get_replacement_words_on_latent(self, word, word_idx, text_input, use_latent=True):
        # Find list of synonym words using HowNet/ WordNet/ GloveEmbedding and their embedding
        candidates = []
        # Tokenize current word
        if word_idx != 0 and isinstance(self.tokenizer.tokenizer, transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast): # If word is not the first word of sentence
            word2tokenize = " " + word # Add space for Roberta tokenizer
        else :
            word2tokenize = word

        word_token = self.tokenizer.tokenizer.tokenize(word2tokenize) # list of tokens

        # If word is tokenized into more than two tokens, skip
        if len(word_token) > 1 :
            return []
        word_token = word_token[0]

        embedding_layer = self.latent_emb
        try :
            word_id = self.syn_embedding.word2index(word.lower())
        except KeyError:
            return []
        pdb.set_trace()
        nnids = self.syn_embedding.nearest_neighbours(word_id, self.syn_top_n)
        nbr_word = [self.syn_embedding.index2word(id_) for id_ in nnids]
        emb_id = [self.tokenizer.encode(w)['input_ids'][1] for w in nbr_word] # Not handles problem with word with multiple tokens
        candidate_emb = self.latent_emb.weight.detach()[emb_id,:] #Tensor of shape (# of candidates, hidden_dim)

        str_input = ' '.join(text_input)
        input_tokens = self.tokenizer.tokenizer.tokenize(str_input)
        input_ids = self.tokenizer.encode(str_input)['input_ids']
        # Get copy of the original embedding
        emb_copy = deepcopy(self.model_for_bp.get_input_embeddings())
        # Select embedding for forwarding. This is the embedding of the original input that will be modified.
        latent_emb = deepcopy(self.latent_emb).weight[input_ids, :].detach().unsqueeze(0)  # Shape : (1, Sequence_length , hidden_dim)
        emb = emb_copy.weight[input_ids, :].detach().unsqueeze(0)  # Shape : (1, Sequence_length , hidden_dim)
        emb.requires_grad = True
        latent_emb.requires_grad = True

        #Find word_idx in input_ids (original word_idx is index in the sentence)
        word_indices = [idx for idx, val in enumerate(text_input) if val == word] # Find indices of all 'word' in sentence
        word_idx = word_indices.index(word_idx) # Find which 'word' out of multiple words. If only single 'word' is in sentence, this will be 0
        word_idx = [idx for idx, val in enumerate(input_tokens) if val == word_token][word_idx] + 1
        if word_idx > 255:
            return []

        # For comparison log the original word embedding
        word_emb = latent_emb[0, word_idx:word_idx + 1, :].clone()
        original = latent_emb[0, word_idx:word_idx+1,:].clone()
        word_id = self.tokenizer.encode(word)['input_ids'][1]
        # print(f"{word} {word_emb.shape}")
        # if word_emb.shape[0] == 0:
        #     print(text_input)
        #     pdb.set_trace()

        with torch.no_grad():
            predictions = self.model_for_bp(inputs_embeds=emb.cuda()).logits
            labels = predictions.argmax(dim=1)  # adv. samples are generated only for correct samples (pred==labels)

        for n_iter in range(self.iterations):
            euclidean_distance = torch.norm(candidate_emb - word_emb, p=2, dim=1)
            nn_word_idx = torch.argsort(euclidean_distance, descending=False)
            nn_word_idx = [emb_id[i] for i in nn_word_idx.tolist()]
            adv_word = self.tokenizer.convert_ids_to_tokens(nn_word_idx[0])

            dist_to_original = torch.norm(original-latent_emb[0,word_idx:word_idx+1,:], p=2)
            dist_to_closest = torch.norm(embedding_layer.weight[nn_word_idx[0]] - word_emb, p=2)
            distance_to_all = torch.norm(embedding_layer.weight - word_emb, p=2, dim=1).mean()

            if self.verbose:
                print(f"ITER{n_iter} : {euclidean_distance.mean().item()}")
                print(f"  {word.lower()} --> {adv_word}")
                print(f"  dist. to closest: {dist_to_closest.item()}")
                print(f"  dist to original emb. {dist_to_original}")
                print(f"  dist. to all: {distance_to_all.item()}")

            # pdb.set_trace()
            self.model_for_bp.eval()
            self.model_for_bp.zero_grad()
            self.decoder.zero_grad()

            seq_length = latent_emb.shape[1]
            latent_emb = latent_emb.view(-1, self.latent_dim)
            latent_emb.retain_grad()
            decoded_target = self.decoder(latent_emb.cuda()).view(seq_length, self.target_dim).unsqueeze(0)
            loss = self.model_for_bp(inputs_embeds=decoded_target, labels=labels)[0]
            loss.backward()

            with torch.no_grad():
                euclidean_distance = torch.norm(candidate_emb - word_emb, p=2, dim=1)
                nn_word_idx = torch.argsort(euclidean_distance, descending=False)
                top_n = 1
                top_candidate_emb = candidate_emb[nn_word_idx[:top_n], : ]
                if top_candidate_emb.dim() == 1:
                    top_candidate_emb.unsqueeze_(0)

                # grad w.r.t to word embeddings
                grads = latent_emb.grad[word_idx:word_idx + 1, :]  # gradient of current word ; Shape (1, embedding_dim)

                #Scale grad here#
                inverse_scale = torch.mean(torch.abs(top_candidate_emb - word_emb), dim=0)
                scale = inverse_scale
                if self.verbose:
                    print(f"  loss: {loss.item()}")
                    print(f"  grad: {grads.norm(p=2)}")
                    print(f"  scale : {scale.mean()}")

                # Scale proportionally to the distance from the candidate words
                grads *= scale
                #Update word embedding here according to gradient
                word_emb = word_emb + self.lr * grads
                latent_emb.unsqueeze_(0)
                latent_emb[:, word_idx:word_idx + 1, :] = word_emb
                pdb.set_trace()
            del latent_emb.grad  #Delete gradients so as to prevent accumulation

            # pdb.set_trace()
        # Find nearest neighbor in the embedding
        euclidean_distance = torch.norm(candidate_emb - word_emb, p=2, dim=1)
        nn_word_idx = torch.argsort(euclidean_distance, descending=False)
        nn_word_idx = [emb_id[i] for i in nn_word_idx.tolist()]
        adv_word = self.tokenizer.convert_ids_to_tokens(nn_word_idx[0])
        # print(f"{word.lower()} --> {adv_word}")
        candidates.extend(self.tokenizer.convert_ids_to_tokens(nn_word_idx))
        distance_with_all = torch.norm(embedding_layer.weight - word_emb, p=2, dim=1)

        return candidates

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        transformed_texts = []
        # print(current_text)
        for i in indices_to_modify:
            if i < 256: # longest sequence is 256 for this model
                word_to_replace = words[i]
                replacement_words = self._get_replacement_words_on_latent(word_to_replace, i, current_text.words)
                transformed_texts_idx = []
                for r in replacement_words:
                    if r == word_to_replace:
                        continue
                    transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
                transformed_texts.extend(transformed_texts_idx)
        # pdb.set_trace()
        return transformed_texts

    def extra_repr_keys(self):
        return ["top_n"]
