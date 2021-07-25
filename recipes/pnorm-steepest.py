import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('//recipes')

from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import BeamSearch
from textattack.shared.attack import Attack

from utils.WordSwapPSGradientBased import PSGradientBased
from textattack.transformations import WordSwapGradientBased


def PNormSteepest(model):
    transformation = PSGradientBased(model, top_n=1)
    # transformation = WordSwapGradientBased(model, top_n=1)

    # Don't modify the same word twice or stopwords
    constraints = [RepeatModification(), StopwordModification()]
    #
    # 0. "We were able to create only 41 examples (2% of the correctly-
    # classified instances of the SST test set) with one or two flips."
    #
    constraints.append(MaxWordsPerturbed(max_num_words=2))
    #
    # 1. "The cosine similarity between the embedding of words is bigger than a
    #   threshold (0.8)."
    #
    constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))
    #
    # 2. "The two words have the same part-of-speech."
    #
    constraints.append(PartOfSpeech())
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model)
    #
    # "HotFlip ... uses a beam search to find a set of manipulations that work
    # well together to confuse a classifier ... The adversary uses a beam size
    # of 10."
    #
    search_method = BeamSearch(beam_width=10)

    return Attack(goal_function, constraints, transformation, search_method)

attack = PNormSteepest