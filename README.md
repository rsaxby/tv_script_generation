## RNN - TV Script Generation

This was the third project in the Udacity Deep Learning Nanodegree program. It was the RNN unit capstone project, where we trained an RNN to generate TV scripts from the sitcom Seinfeld. 

### Data

The dataset was taken from a collection of Seinfeld scripts from 9 seasons of the show. It's available for download on [Kaggle](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv).

### Data Preprocessing

I split the text by word, converted to lowercase, and tokenized punctuation. Then I encoded the words to integers. I created dictionaries to convert from words to integers and vice versa, as well as a dictionary to convert a punctuation symbol to its tokenized version. 

### LSTM Hyperparameters

As is common, I determined my hyperparemeters through experimenting. I was surprised by some results, which led to some interesting insights. 

For instance, I found that a smaller sequence length worked best. It seems a bit counterintuitive given that a larger sequence length means the network has more context to learn from. But, considering the size and nature of the data - a script from a TV sitcom, which has mostly short exchanges between characters (~5 words per line), perhaps a smaller context/sequence length is ideal.

For the hidden dimension and number of layers, I also experimented. I found that a large hidden dimension allowed the model to better learn the complex structure of the dialogue. For the learning rate, I settled on 0.001, a smaller lr had a slightly more stable, but much slower convergence. For the embedding dimension, I selected 200, again through experimentation.

I chose to use the following hyper parameters:

* sequence length: 10
* batch size: 128
* embedding dim: 200
* hidden dim: 256
* num layers: 2
* learning rate: 1e-4

I trained for 10 epochs, and achieved a loss of 2.85. The loss was still (generally) decreasing, so I probably could have trained for even longer to achieve better results.