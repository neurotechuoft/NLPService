# -*- coding: utf-8 -*-

""" Use torchMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the torchMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division, unicode_literals

import json
import sys
from os.path import abspath, dirname

import numpy as np
from emoji import unicode_codes

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
:pensive: :ok_hand: :blush: :heart: :smirk: \
:grin: :notes: :flushed: :100: :sleeping: \
:relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
:sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
:neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
:v: :sunglasses: :rage: :thumbsup: :cry: \
:sleepy: :yum: :triumph: :hand: :mask: \
:clap: :eyes: :gun: :persevere: :smiling_imp: \
:sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
:wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
:angry: :no_good: :muscle: :facepunch: :purple_heart: \
:sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')

MAXLEN = 30


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


async def predict_sentence_emojies(sentense: str) -> dict:
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, MAXLEN)

    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = torchmoji_emojis(PRETRAINED_PATH)
    print(model)
    print('Running predictions.')
    tokenized, _, _ = st.tokenize_sentences([sentense])
    prob = model(tokenized)[0]

    ind_top = top_elements(prob, 5)
    emojis = list(map(lambda x: EMOJIS[x], ind_top))
    emojis_unicode_escape = [unicode_codes.EMOJI_ALIAS_UNICODE[emoj].encode('unicode-escape') for emoj in emojis]

    emojis_unicode = [unicode_codes.EMOJI_ALIAS_UNICODE[emoj] for emoj in emojis]
    return dict(zip(emojis_unicode, prob[ind_top]))


if __name__ == '__main__':
    print(predict_sentence_emojies("hello there"))