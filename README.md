# NLPService
Microservice that does word and emoji prediction

## Installation
Using Anaconda or Miniconda, create an environment with the packages listed in ```environment.yml``` with
```
conda env create -n nlp-service -f environment.yml
```
This creates a new conda environment named ```nlp-service```. For emoji prediction support, we are using [TorchMoji](https://github.com/huggingface/torchMoji). Clone the TorchMoji repository into src, and follow their installation instructions for installing PyTorch and TorchMoji dependencies.

## Usage
Activate the conda environment.

__Server side__: run ```python main.py``` to start the server.

__Client side__: connect to the server with SocketIO client and emit "autocomplete" with a word or phrase as an argument to get word predictions. Example for autocompleting the word "a":

```
from socketIO_client import SocketIO

# The code below is to simulate requests from the client
def callback_func(*args):
    print("predictions: ", args)

# server running on localhost:8001
socket_client = SocketIO("localhost", 8001)
socket_client.connect()
socket_client.emit("autocomplete", "a", callback_func)
socket_client.wait_for_callbacks(seconds=1)
```
Will print:
```
predictions:  ('and', 'at', 'as')
```

This example was taken from ```complete.py```.

## Server on Google Cloud Platform

We currently have a server running on GCP with IP ```34.73.165.89``` on port ```8001```. To get a prediction or autocomplete a word, run the code above but with
```
socket_client = SocketIO("34.73.165.89", 8001)
```
