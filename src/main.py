from sanic import Sanic
import socketio
import sys

import trie_funcs
import complete
import nlp_setup
import score_texts_emojis

sio = socketio.AsyncServer()

# Only if you want to use a custom server; here I want to us Sanic
app = Sanic()
sio.attach(app)


@sio.on('autocomplete')
async def handle_data(sid, data):
    predictions = await trie_funcs.autocomplete(data)
    return predictions


@sio.on('predict_emojis')
async def handle_data(sid, data):
    predictions = await score_texts_emojis.predict_sentence_emojies(data)
    return predictions


if __name__ == '__main__':
    nlp_setup.perform_setup(True, True)

    if len(sys.argv) == 2:
        app.run(host='localhost', port=sys.argv[1])
    else:
        app.run(host='localhost', port=complete.test_port)
