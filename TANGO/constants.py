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

# pronouns
NON_FUNCTION_PRONOUNS = {  # need to change pronoun based on context
    # one-to-many mapping
    # using dependency parser for "her" --> "their" / "them", so not including "her" here
    # using dependency parser for "his" --> "their" / "theirs", so not including "his" here

    # "(s)he's" can be resolved as either "(s)he is" or "(s)he has"
    # mapping "(s)he's" instead of operating on individual tokens "(s)he" and "'s"
    # this is because we use regex to find and replace "'s" before feeding the sentence into LM
    # searching for "(s)he's" instead of "'s" prevents false positives such as "that's" --> "that're"
    "he's": ["they've",
             "they're"],
    "she's": ["they've",
              "they're"]
}

NON_INJECTIVE_PRONOUNS = {
    # many-to-one mapping
    'he': 'they',
    'she': 'they',
    "she's": "they're",
    "he's": "they're",
    'herself': 'themself',
    'himself': 'themself'
}

INJECTIVE_PRONOUNS = {
    # one-to-one mapping
    'him': 'them',  # I talked to him --> I talked to them
    'hers': 'theirs'  # This pen is hers --> This pen is theirs
}

EASY_PRONOUNS = NON_INJECTIVE_PRONOUNS.copy()
EASY_PRONOUNS.update(INJECTIVE_PRONOUNS)  # these pronouns can be replaced directly

IRREGULAR_VERBS = {
    'is': 'are',
    'was': 'were',
    'has': 'have',
    'does': 'do',
    'goes': 'go',
    'quizzes': 'quiz'  # 1-1-1 doubling rule
}

VERB_ES_SUFFIXES = ['ses', 'zes', 'xes', 'ches', 'shes']