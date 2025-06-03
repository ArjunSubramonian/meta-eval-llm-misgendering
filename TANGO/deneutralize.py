# This code is from: https://github.com/googleinterns/they-them-theirs.
# Changes were made from the original.

# Please see the license here: https://github.com/googleinterns/they-them-theirs?tab=Apache-2.0-1-ov-file#readme, and below:
# Copyright 2022 Tony Sun

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import torch
import pandas as pd
from string import punctuation

# SpaCy: lowercase is for dependency parser, uppercase is for part-of-speech tagger
from spacy.symbols import nsubj, nsubjpass, conj, poss, obj, iobj, pobj, dobj, VERB, AUX, NOUN
from spacy.tokens import Token, Doc

from constants import *

# load SpaCy's "en_core_web_sm" model
# English multi-task CNN trained on OntoNotes
# Assigns context-specific token vectors, POS tags, dependency parse and named entities
# https://spacy.io/models/en
import en_core_web_sm

nlp = en_core_web_sm.load()

# SpaCy: lowercase is for dependency parser, uppercase is for part-of-speech tagger
from spacy.symbols import nsubj, nsubjpass, conj, poss, obj, iobj, pobj, dobj, VERB, AUX, NOUN
from spacy.tokens import Token, Doc

pronouns = pd.read_csv('tango-pronouns.csv')

def revert(sentence: str) -> str:
    """
    convert a sentence to singular form
    :param sentence: sentence meeting SNAPE criteria (meaning 1 entity and 1 gender)
    :return: sentence in singular form
    """

    # replace pronouns that have a clear mapping
    # cannot directly modify "doc" object, so we will instead create a "replacement" attribute
    # in the end, we will create a new doc from original doc and any replacements
    Token.set_extension("simple_replace", getter=simple_replace, force=True)

    # Doc is a SpaCy container comprised of a sequence of Token objects
    doc = nlp(sentence)

    # create a dictionary mapping verbs in sentence from third-person plural to third-person singular
    verbs_auxiliaries = identify_verbs_and_auxiliaries(doc=doc)
    verbs_replacements = singularize_verbs(verbs_auxiliaries)

    # create a new doc with replacements for pronouns, verbs
    new_sentence = create_new_doc(doc, verbs_replacements)

    return new_sentence


def simple_replace(token: Token):
    """
    mainly deals with straightforward cases of pronoun replacement using a lookup
    :param token: SpaCy token
    :return: the token's text replacement (if it exists) as a string.
    """
    text = token.text
    if text == "'re" or text == "'ve":
        return "'s"

    mask = (pronouns.iloc[:, 2:] == text.lower()).any(axis=0)
    cols = pronouns.columns[2:]

    if len(cols[mask]) == 0:
        return None

    return mask_capitalization_helper(original=text,
                                 replacement=cols[mask][0])


def mask_capitalization_helper(original: str, replacement: str) -> str:
    # check for capitalization
    if original.istitle():
        return '{' + replacement.capitalize() + '}'
    elif original.isupper():
        return '{' + replacement.upper() + '}'

    # otherwise, return the default replacement
    return '{' + replacement + '}'

def capitalization_helper(original: str, replacement: str) -> str:
    """
    helper function to return appropriate capitalization
    :param original: original word from the sentence
    :param replacement: replacement for the given word
    :return: replacement word matching the capitalization of the original word
    """
    # check for capitalization
    if original.istitle():
        return replacement.capitalize()
    elif original.isupper():
        return replacement.upper()

    # otherwise, return the default replacement
    return replacement


def identify_verbs_and_auxiliaries(doc: Doc) -> dict:
    """
    identify the root verbs and their corresponding auxiliaries with 'they' as the subject
    :param doc: input Doc object
    :return: dictionary with verbs (SpaCy Token) as keys, auxiliaries as values (SpaCy Token)
    """
    # no need to include uppercase pronouns bc searching for potential_subject checks lower-cased version of each token
    SUBJECT_PRONOUNS = ['they']

    # identify all verbs
    verbs = set()
    # this deals with repeating verbs, e.g. "He sings and sings."
    # verb Token with same text will have different position (makes them unique)
    for possible_subject in doc:
        is_subject = (
                (possible_subject.dep == nsubj or
                 possible_subject.dep == nsubjpass) and  # current token is a subject
                # head of current token is a verb
                (possible_subject.head.pos == VERB or possible_subject.head.pos == AUX) and
                possible_subject.text.lower() in SUBJECT_PRONOUNS  
        )
        if is_subject:
            verbs.add(possible_subject.head)

    # identify all conjuncts and add them to set of verbs
    # e.g. he dances and prances --> prances would be a conjunct
    for possible_conjunct in doc:
        is_conjunct = (
                possible_conjunct.dep == conj and  # current token is a conjunct
                possible_conjunct.head in verbs  
        )
        if is_conjunct:
            verbs.add(possible_conjunct)

    verbs_auxiliaries = dict()
    for verb in verbs:
        verbs_auxiliaries[verb] = list()
    for possible_aux in doc:
        is_auxiliary = (
                possible_aux.pos == AUX and  # current token is an auxiliary verb
                possible_aux.head in verbs  
        )
        if is_auxiliary:
            verb = possible_aux.head
            verbs_auxiliaries[verb].append(possible_aux)

    return verbs_auxiliaries


def singularize_verbs(verbs_auxiliaries: dict) -> dict:
    """
    map each verb and auxiliary to its singular form
    :param verbs_auxiliaries: dictionary with verbs (SpaCy Token) as keys, auxiliaries as values (SpaCy Token)
    :return: dictionary with verbs and auxiliaries (SpaCy Token) as keys, singular form as values (str or None)
    """
    REV_IRREGULAR_VERBS = {v : k for k, v in IRREGULAR_VERBS.items()}
    
    verbs_replacements = dict()

    for verb, auxiliaries in verbs_auxiliaries.items():
        # verb has no auxiliaries
        if not auxiliaries:
            verbs_replacements[verb] = singularize_single_verb(verb)

        # there are auxiliary verbs
        else:
            verbs_replacements[verb] = None  # do not need to singularize root verb if there are auxiliaries

            # use a lookup to find replacements for auxiliaries
            for auxiliary in auxiliaries:
                text = auxiliary.text
                if text.lower() in REV_IRREGULAR_VERBS.keys():
                    replacement = REV_IRREGULAR_VERBS[text.lower()]
                    verbs_replacements[auxiliary] = capitalization_helper(original=text,
                                                                          replacement=replacement)
                else:
                    verbs_replacements[auxiliary] = None

    return verbs_replacements


def singularize_single_verb(verb: Token):
    """
    singularize a single verb
    :param verb: verb as a SpaCy token
    :return: the singular form of the verb as a str, or None if verb doesn't lend itself to singularization
    """
    verb_text = verb.text

    # check verb tense (expect to be either past simple or present simple)
    if 'Tense' not in verb.morph.to_dict():
        verb_tag = None
    else:
        verb_tag = verb.morph.to_dict()['Tense']

    if verb_tag == 'Past':
        # were is an irregular past tense verb from third-person plural to third-person singular
        if verb_text.lower() == 'were':
            return capitalization_helper(verb_text, 'was')

        # other past-tense verbs remain the same
        else:
            return None

    # oftentimes, if there are 2+ verbs in a sentence, each verb after the first (the conjuncts) will be misclassified
    # the POS of these other verbs are usually misclassified as NOUN
    # does not properly handle conjunct verbs
    elif verb_tag == 'Pres':
        return capitalization_helper(original=verb_text.lower(),
                                     replacement=singularize_present_simple(verb_text))

    return None


def singularize_present_simple(lowercase_verb: str):
    """
    singularize a third-person plural verb in the present simple tense
    :param lowercase_verb: original verb (lower-cased)
    :return: 3rd-person singular verb in the present simple tense
    """
    
    for singular, plural in IRREGULAR_VERBS.items():
        if lowercase_verb == plural:
            return singular

    if lowercase_verb.endswith('y'):
        return lowercase_verb[:-1] + 'ies'

    # -es rule: https://howtospell.co.uk/adding-es-singular-rule
    for suffix in VERB_ES_SUFFIXES:
        if lowercase_verb.endswith(suffix[:-2]):
            return lowercase_verb + 'es'

    return lowercase_verb + 's'


def create_new_doc(doc: Doc, verbs_replacements: dict):
    """
    create a new SpaCy doc using the original doc and a mapping of verbs to their replacements
    :param doc: original doc with simple_replace extension (from simple_replace function)
    :param verbs_replacements: dictionary mapping verbs and auxiliaries to their replacements
    :return: the singular sentence as a str
    """
    token_texts = []
    for token in doc:
        replace_verb = (token in verbs_replacements.keys() and
                        verbs_replacements[token])

        if token._.simple_replace:
            token_texts.append(token._.simple_replace)

        elif replace_verb:
            token_texts.append(verbs_replacements[token])

        else:
            token_texts.append(token.text)
        if token.whitespace_:  # filter out empty strings
            token_texts.append(token.whitespace_)

    new_sentence = ''.join(token_texts)
    return new_sentence