# twitter-sentiment-analysis-with-LSTMs-ELMo
Twitter Sentiment analysis using RNS like LSTMs, GRUs and enhancing the performance with ELMo embeddings and a self-attention model

## About the project
In this project, there is a system for classifying Twitter posts using RNNs. We use different types of RNNs like LSTMs and GRUs and we make an extensive grid search to find the best parameters. Then to enchase the performance, we implement a self attention-model, a variant of the attention mechanism, proposed by  [(Lin et al., 2017)](https://arxiv.org/pdf/1703.03130.pdf).
Finally, we use ELMo embeddings [(Peters at el., 2018)](https://www.aclweb.org/anthology/N18-1202/) in our system which have been shown to improve our model performance.

## About the implementation
### Data cleaning
First, we clean the data and do basic editing of the Tweets by removing the stop words. Then we define the functions and methods which we will be using to create the model, train it and get its metrics of accuracy.

### RNNs 
After that, we test different RNN cell types GRU and LSTM with bidirectionality, different hidden layer sizes, and different stacked RNNs.
If a configuration overfits then we introduce some dropout or a more aggressive gradient clipping.
After the testing, we pick the most promising configurations that do not overfit.
After finding the best configuration, these parameters will undergo some tuning with a randomized grid search.

### Self-attention
For the self-attention model, we implement the method proposed by [(Lin et al., 2017)](https://arxiv.org/pdf/1703.03130.pdf).
The parameters of w1 and w2 weights from the Dense layers are 10 and 5 according to the paper.
The use of the self-attention model gives us a context about the location to which the model should pay more attention in the sentence. With this method, the model can activate its neurons more to the areas of the sentence that statistically have shown more contextual importance.
Then try different configurations on the self-attention model

### ELMo
Embeddings from a language model are superior to the word2vec embeddings because they take into consideration a lot of contextual meaning rather than the nearby words. 
The pre-trained model that will be used is described in [(Peters et el., 2018)](https://www.aclweb.org/anthology/N18-1202/). For the pre-trained model, the [allennlp library](https://allennlp.org/elmo) will be used.
We train the LSTM model using the full 5.5B pre-trained model and also with the small pre-trained model with the full dataset of 1.2 million Tweets. 


# Code and report
The code and the report can be found at the notebook which is insisted to be opened on google colab from here ->
