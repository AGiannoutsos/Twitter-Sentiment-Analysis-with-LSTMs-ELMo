# Twitter-Sentiment-Analysis-with-LSTMs-ELMo
Twitter Sentiment analysis using RNS like LSTMs, GRUs and enhancing the performance with ELMo embeddings and a self-attention model

# Code and report
The code and the report can be found at the notebook which is insisted to be opened on Google colab from here ->
Also, you are highly insisted to use the Google colab's Table of Contents section since there are many paragraphs and sections in the notebook.

## About the project
In this project, there is a system for classifying Twitter posts using RNNs. We use different types of RNNs like LSTMs and GRUs and we make an extensive grid search to find the best parameters. Then to enchase the performance, we implement a self attention-model, a variant of the attention mechanism, proposed by  [(Lin et al., 2017)](https://arxiv.org/pdf/1703.03130.pdf).
Finally, we use ELMo embeddings [(Peters at el., 2018)](https://www.aclweb.org/anthology/N18-1202/) in our system which have been shown to improve our model performance.

## About the implementation
### Data cleaning
First, we clean the data and do basic editing of the Tweets by removing the stop words. Then we define the functions and methods which we will be using to create the model, train it and get its metrics of accuracy.

### RNNs, LSTMs, GRUs
After that, we test different RNN cell types GRU and LSTM with bidirectionality, different hidden layer sizes, and different stacked RNNs.
If a configuration overfits then we introduce some dropout or a more aggressive gradient clipping.
After the testing, we pick the most promising configurations that do not overfit.
After finding the best configuration, these parameters will undergo some tuning with a randomized grid search.
For the grid-search we will be using Weights and Biases which is a tool for logging the metrics of the model.

### Self-attention
For the self-attention model, we implement the method proposed by [(Lin et al., 2017)](https://arxiv.org/pdf/1703.03130.pdf).
The parameters of w1 and w2 weights from the Dense layers are 10 and 5 according to the paper.
The use of the self-attention model gives us a context about the location to which the model should pay more attention in the sentence. With this method, the model can activate its neurons more to the areas of the sentence that statistically have shown more contextual importance.
Then try different configurations on the self-attention model

### ELMo
Embeddings from a language model are superior to the word2vec embeddings because they take into consideration a lot of contextual meaning rather than the nearby words. 
The pre-trained model that will be used is described in [(Peters et el., 2018)](https://www.aclweb.org/anthology/N18-1202/). For the pre-trained model, the [allennlp library](https://allennlp.org/elmo) will be used.
We train the LSTM model using the full 5.5B pre-trained model and also with the small pre-trained model with the full dataset of 1.2 million Tweets. 

## About the results

### RNNs, LSTMs, GRUs

From the different parameters and configurations of RNN models, we cannot surpass the 0.78 validation accuracy.
GRU and LSTMS even though LSTM blocks have more gates and are more complex in this dataset don't have many differences. We can even say that relatively to GRU's computational complexity it is superior.
About the number of hidden layers, small sizes of 16, 32, and 64 have great generalization capabilities. Larger sizes only tend to overfit without increasing the score significantly.
Bidirectionality also doesn't have a great impact but we can see a slight improvement.
Dropout and gradient clipping won't also change drastically the results. Also, dropout needs to be very small otherwise the model won't converge.
Finally, for the best configuration that we will apply a random grid search, we will choose the bidirectional LSTM with 8 to 64 layers.
These are the best model configuration because the results show that these configurations have room for improvement.

####  Randomized grid search

We can't find a strong correlation between the parameters that play the most crucial role.
One trend that we can detect is that with hidden layers of less than 16 f1 models perform worse. The best-hidden size should be between 32 and 64.
Another trend is that having a smaller learning rate of 0.0001, models score less in contrast with a greater learning rate of 0.001. The in-between values of the learning rate achieve the best results.


### Self-attention

From the tested models and parameters, we couldn't find a model that performed better than the non-attention model having a simple sigmoid function at the last RNN block.

However, the self-attention model could learn faster and, it converged at faster rates than the simple sentiment classification model.
Although it was prone to more overfitting because the model could learn faster by ignoring parts of the sentence that didn't have important contextual meaning.
As for the accuracy, it did not add that much up. Maybe the pre-trained embeddings from a language model may be better at increasing the accuracy of the model.

### ELMo
512 hidden layer size gave us the best results from all the previous experiments. The model even reached a validation accuracy of 0.85 but from that point and on the model was overfitting and it reduced its validation accuracy.
For that reason, we may try to early stop it at 5 epochs only.
A further examination of dropout ar l2 regularization is practically forbidden caus the model took 7 hours to train which is just enough for a Google Colab GPU Runtime.

From the testing we can clearly notice that the Embeddings from a Language Model can impove the scores by an remarkable rate. 
A bigger model with 512 hidden layers and more could benefit from the elmo embeddings. 
Also the elmo was using only one layer beacause of memory limitations. Using one more might have even grater results.



