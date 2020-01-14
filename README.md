# Speech Recognition
This is a speech recognition project made using the google's speech recognition challenge dataset.
This speech recogntion code is trained on a subset of the entire the TensorFlow speech recognition challenge.
it includes 65,000 one-second long utterances of 30 short words, by thousands of different people.
We will be targeting only a few of the voice from this dataset namely ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'].

### Depandancies

<li><a href = "https://librosa.github.io/librosa/">Librosa</a></li>
<li><a href = "https://keras.io">Keras</a></li>

### Usage
You can simply download the file ```speechrecog.py``` and the dataset from -- <ul>https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data</ul>
and train the model on your system by running the model on your system.

Alternatively you can use the pre trained model uploaded ```model.pickle``` and use the ```predict.py``` file to run the model.
