# -*- coding: utf-8 -*-

"""
Use torchMoji to score texts for emoji distribution.
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


async def predict_sentence_emojis(sentence: str, num_to_predict: int = 5) -> dict:
    """
    Predict top n emojis based on the sentence
    :param sentence: sentence used in prediction
    :param num_to_predict: number of top emojis to return
    :return: Dictionary where key is predicted emoji and value is its probability
    """

    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, MAXLEN)

    model = torchmoji_emojis(PRETRAINED_PATH)
    print('Running predictions.')
    tokenized, _, _ = st.tokenize_sentences([sentence])
    prob = model(tokenized)[0]

    ind_top = top_elements(prob, num_to_predict)
    emojis = list(map(lambda x: EMOJIS[x], ind_top))

    # Might be useful if we need to send it this way
    # emojis_unicode_escape = [unicode_codes.EMOJI_ALIAS_UNICODE[emoj].encode('unicode-escape') for emoj in emojis]

    emojis_unicode = [unicode_codes.EMOJI_ALIAS_UNICODE[emoj] for emoj in emojis]
    return dict(zip(emojis_unicode, prob[ind_top]))


if __name__ == '__main__':
    print(predict_sentence_emojies("hello there"))
