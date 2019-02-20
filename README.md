# NLPService
Microservice that does word and emoji prediction

## Installation
Using Anaconda or Miniconda, create an environment (replacing with the packages listed in ```environment.yml``` with
```
conda env create -n nlp-service -f environment.yml
```
This creates a new conda environment named ```nlp-service```.

## Usage
Activate the conda environment.

__Server side__: run ```python main.py``` to start the server.

__Client side__: connect to the server with SocketIO client and emit "predict" with a word or phrase as an argument to get word predictions. Example for predicting "a":

```
from socketIO_client import SocketIO

# The code below is to simulate requests from the client
def callback_func(*args):
    print("predictions: ", args)

socket_client = SocketIO("localhost", 8001) # server running on localhost:8001
socket_client.connect()
socket_client.emit("predict", "a", callback_func)
socket_client.wait_for_callbacks(seconds=1)
```
Will print:
```
predictions:  ('and', 'at', 'as')
```

This example was taken from ```complete.py```.
