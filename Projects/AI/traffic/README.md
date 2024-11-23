My notes here:

1/ use os.path.join instead of '\' or '/' so the program can run in multiple hedieuhanhs
2/ use os.listdir to list all the files/folders in a directory so that we can iterate over it
3/ cv2.imread (read a image) and resizing cv2.resize()




Convolutional Neural Network

Convolution:
First we have a 2D (or 3D since there are RGB values?) images as input
But the size of image is too large, as well as there is so much information about that so it would take an enormous computational power to train a NN
--> We have to filter it! using a method called "Convolution" which is having several filters, each filters mean to extract a particular trait of the image and
also each filter is a "kernel" has a particular size that will slide window through the image and extract information (as numbers)
--> then, after filtering, now we have several filtered images, each one has particular info, each of them then goes through max-pooling (just to compress the size
of the image - this one does not really affect the quality of training because at one area of the image, we care about other features somewhere else in the
image, so we don't really need to care things around that point) --> then we MAY do a "convolution" again to extract high level features (for example: if the first 
convolution is for detecting curves, edges,... the the second would extract objects) --> then max-pool again? (depends on u, and there are several methods rather
than max-pooling too!). After that, we can FLATTEN these 2D images and feed them into the NN and begin to learn.

DNN: (Deep Neural Network)
the inputs, numerical
then the second layer, each unit (or each neuron) learn a particular trait, or has a particular job, or learn sth particularly
each layer has an activation method: reLU, sigmoid, softmax,...
and again again inputs, or lets say information flow through several layers, at each layer learn something, and then output to the LAST LAYER - which contains
the answer! 
At the last layer, if we are using Supervised Learning (or fine tuning, or whichever already has labels - i read this from Kaggle 5 days intensive course),
 which means we have some labels, then we evaluate the loss function, and then tell the fucking NN to backpropagate (i learned from 3Blue1Brown here, a lot of
 calculus here lol: gradien descent, derivative,...) to learn backwards, fix the weights...

 each time we do an epoch, the loss function gets smaller and smaller -> imagine like its descending (u may visualize a mountain and a man trying to go down the hill) 
 -> so do several epoches, each epoch let gradien descent minimizes the loss.


Questions i asked myself:
- how if after all convolution steps, we get an array of lets say, 100 values, but our first input layer has 128 units?
--> Nah, one thing you get wrong there is that the first input layer doesnt look like what you define. In fact, the first input layer is THAT 100 VALUES you feed
into the Neural Network, then these values are then passed to the next layer, also called the first layer of hidden layers, which has 128 units.

 Also key notes from 5 days Kaggle intensive course GenAI:
 - if we train (or lets say let the model learn) without labels, it becomes more general! meaning it knows more!. But if we want it to perform particularly good 
 at a field, we fine tune it! Which is to feed a set of labeled data to make the model more "in" that data :) and become good at that field.    





ChatGPT helps clarifying these:
- In each epoch, the model trains on the entire dataset once.

However, there’s an important detail: during an epoch, the dataset is often split into smaller parts called batches, and the model processes one batch at a time. Let’s break it down simply:

Dataset and Epochs
If your dataset has 10,000 images, one epoch means the model processes all 10,000 images exactly once.
But the model doesn’t process all 10,000 images in one go (this would require too much memory). Instead, it splits the dataset into batches.

Batches
A batch is a smaller subset of the dataset. For example, you might divide the 10,000 images into batches of 100 images each.
This means:
The model trains on 100 images at a time (1 batch).
After 100 images are processed, it moves to the next batch of 100, and so on.
To complete one epoch (processing all 10,000 images), the model processes 100 batches (if batch size = 100).


In each epoch, the model trains on the entire dataset once.

However, there’s an important detail: during an epoch, the dataset is often split into smaller parts called batches, and the model processes one batch at a time. Let’s break it down simply:

Dataset and Epochs
If your dataset has 10,000 images, one epoch means the model processes all 10,000 images exactly once.
But the model doesn’t process all 10,000 images in one go (this would require too much memory). Instead, it splits the dataset into batches.
Batches
A batch is a smaller subset of the dataset. For example, you might divide the 10,000 images into batches of 100 images each.
This means:
The model trains on 100 images at a time (1 batch).
After 100 images are processed, it moves to the next batch of 100, and so on.
To complete one epoch (processing all 10,000 images), the model processes 100 batches (if batch size = 100).
Steps in an Epoch
Divide the dataset into batches.
Train the model on each batch sequentially.
After all batches are processed, the epoch is complete.


Why Use Batches?
Efficiency: Training on smaller batches uses less memory than processing the entire dataset at once.
Faster Updates: After each batch, the model adjusts its weights, allowing it to learn incrementally.
Scalability: For large datasets, splitting into batches ensures the model can fit into memory.


# Hmm, in this code, the author doesn't use batches (segmenting the data into small parts and train them one by one)
# Instead, he uses the whole data to train the model at once, which is 60% of the data for training and 40% for testing
# I should try to use batches to see if it is better or not


tips:
- tell chatgpt to explain to a 5-yrs old kid if things get too hard to understand:)


usage:
- detect objects, classify people,...


from CS50, types of NN:
- Feed-forward: pretty simple, pass data one way (input -> network -> output). Helpful for classification task, but it has limitations: the input must be in
a fixed shape of inputs/ outputs
-> solution: RNN - recurrent neural network, its outputs are fed back to its network  --> useful when dealing with senquences of data (for example: the bot that can
read an image and describe the image --> fixed input but flexible output). Imagine input -> network -> output then feed the output into network again --> output2
--> again and again...
other use cases: youtube uses RNN to detect copyright violations, feed a frame first, then pass to the network, but the network doesn't give the output but continues
to feed another frame, and feed into the network again. At this time, the network gets information about the first network that was fed by the first frame to "learn" more information
. again and again,... --> provide an ouput: violate or not //// or classify if a review is positive or negative /// or spoken language, we can segment waveforms
into chunks and feed into the input of RNN to classify whose voice /// (these are one-to-many relationships, wat about many-to-many? -> here:) google translator
: input fed into network, then next input fed into network and network uses info from previous input,... then repeat -> then generate the first translated word
-> then feed into the network, then get another output then feed into the network again.
LSTM - Long short term memory

evolution: Neural network -> feed-forward neural network -> convolutional neural network -> recurrent neural network -> (adversarial network) -> transformer
but transformer is more into Natural language

later we will come to TRANSFORMERRRRR - a fucking cool model that i think can do whatever RNN can do, but in a different way (that is running parallelly?)


CHAD CHIPITI compares RNN and Transformer:

Recurrent Neural Networks (RNNs) and Transformers are both architectures used in deep learning, particularly for sequence-based tasks like natural language processing (NLP). However, they are fundamentally different in how they handle data, process sequences, and represent dependencies. Below is a deep differentiation:

1. Architecture Design
RNN:

Processes input sequentially, one step at a time.
Maintains a hidden state that captures information from previous steps.
Often uses feedback loops to update its state.
Variants like LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units) improve the ability to handle long-term dependencies by mitigating vanishing gradients.
Transformer:

Processes input in parallel rather than sequentially.
Relies on a self-attention mechanism to capture relationships between all tokens in a sequence simultaneously.
Employs a feedforward network architecture without recurrent connections.
Contains layers of multi-head attention and position-wise feedforward networks.


2. Sequential Processing
RNN:

Sequential by nature, so it cannot parallelize computations over time steps.
Slower for long sequences since each time step depends on the output of the previous step.
Struggles with long-range dependencies due to gradient vanishing/exploding issues (mitigated but not eliminated by LSTMs and GRUs).
Transformer:

Non-sequential; computes relationships for all tokens simultaneously using self-attention.
Highly parallelizable, making it much faster for training on long sequences.
Handles long-range dependencies more effectively since attention weights are computed directly for all pairs of tokens.



3. Handling Context
RNN:

Captures temporal context incrementally through hidden states.
Limited ability to retain distant context due to memory decay in hidden states.
May require larger hidden states to capture long-term dependencies effectively.
Transformer:

Uses self-attention to capture global context from the entire sequence.
Each token attends to every other token, enabling robust understanding of dependencies regardless of distance.
Positional encodings are added to tokens to retain the order information lost in parallel processing.

4. Computational Efficiency
RNN:

Computationally inefficient for long sequences due to sequential processing.
Backpropagation through time (BPTT) is used, which is memory-intensive as it requires storing intermediate states.
Transformer:

Efficient for long sequences due to parallel processing.
Scales quadratically with sequence length in the self-attention mechanism, which can be computationally expensive for very long inputs but manageable with optimizations (e.g., sparse attention).





5. Data Representation
RNN:

Learns temporal representations from sequences incrementally.
Depends heavily on the hidden state to pass information from earlier time steps to later ones.
Transformer:

Learns representations through attention across the entire sequence.
Utilizes embeddings and attention mechanisms to encode relationships explicitly.

6. Application and Suitability
RNN:

Suited for tasks requiring strict temporal modeling, such as time-series forecasting or continuous speech recognition.
Performs well on tasks with shorter sequences or where dependencies are mostly local.
Transformer:

Dominates in NLP tasks like machine translation, text summarization, and question answering due to its ability to model global dependencies.
Has extended into non-NLP domains like computer vision (Vision Transformers) and time-series analysis.

7. Popular Models
RNN:

Vanilla RNN, LSTM, GRU.
Applications: Speech-to-text systems (e.g., DeepSpeech), time-series analysis, early NLP models.
Transformer:

Transformer architecture (Vaswani et al., 2017).
Pretrained models like BERT, GPT, and T5 are Transformer-based.
Applications: ChatGPT, Google Translate, DALL·E, Vision Transformers.

8. Limitations
RNN:

Struggles with long-range dependencies and gradient issues.
Slow training due to sequential nature.
Harder to parallelize, limiting scalability.
Transformer:

Requires significant computational resources for large-scale pretraining and fine-tuning.
Quadratic complexity of self-attention makes it challenging for very long sequences.

Key Takeaway
RNNs focus on temporal, step-by-step processing, making them suitable for smaller-scale tasks or applications requiring strict sequential processing.
Transformers, through self-attention and parallelism, revolutionized deep learning by enabling the efficient handling of large datasets and capturing global context, making them the state-of-the-art choice in NLP and beyond.



regarding coding part:


1/ tf.keras.layers.Flatten(),
          # Now add first hidden layer
          tf.keras.layers.Dense(128, activation="softmax"),
          tf.keraas.layers.Dense(64, activation="relu"),
          # remember: each unit "learns" something! (and how it learns? it is kinda like introducing a new function and that function has it own job. TensorFlow will do that)


model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

use "adam" optimizer. More about it, ask ChadGPT (mathematially heavy). Adam will utilize backpropagation? or Senquential already uses it? to minimize the loss?
use categorical_crossentropy to evaluate loss function. More about it, read the above sentence

capture of wat chadgpt said:
The training process will use backpropagation to compute gradients and update weights. Backpropagation is a method for calculating the gradients of the loss function with respect to the model parameters by applying the chain rule through the network. It allows the optimizer (in this case, Adam) to adjust the weights to minimize the loss. The backpropagation process involves:

Propagating the error backward from the output layer to the input layer.
Updating the weights and biases of the layers based on these computed gradients.


Some Observations:
1/ just realized that linear regression or logistic regression are just special cases of Neural network




This is how ChatGPT explain my questions:
1/ Why adding layers make the NN "smarter"? Why these layers have these meanings??? Why we initially don't define each layer's meaning but when the NN trained with data, these "meanings" for each layer gradually appears?
2/ Why do the early layers tend to represent more general meaning than the later layers?

My answers after researching:
1 - Yes, you are right. Initially, it has no meaning, its weights are set randomly (utter trash). However, when we feed it with dataset, it is like "LEARNING"
from the dataset to get smarter to be meaningful. For example in Gomoku game, if we feed it with a board of a-about-to-lose state, if it gives the move (x, y) for that state but the optimal move is (a, b) then it will learn backward, using backpropagation, minimizing the loss function, and tell the network to adjust the weights, implying that "this move" is BAD --> pass backwards info --> gets smarter = changes weights ~~~ gradually the layers have meanings (for ex: early layers after several training say that: "this region seems promising, could lead to a win - because the weight is pretty high there"...)
2 - Because later layers mean more kernel applied --> they have deeper and deeper meaning.

Chad GPT's answer:
Ah, now I understand your question more clearly! You're wondering how the neural network "knows" that detecting simple configurations like "two of my pieces in a row" or "three opponent pieces in a row" is important for learning higher-level strategies in a game like Gomoku.

The Core Idea: Learning Features by Gradients and Backpropagation
The key concept to understand here is that the neural network doesn't explicitly "know" that certain patterns, like "two pieces in a row," are important for winning or losing. Rather, it learns which features matter based on the training process and gradients computed during backpropagation. Here's a deeper breakdown:

1. Neural Network Initialization:
When you first initialize a neural network, the network’s weights are set randomly (or in some cases, according to some initialization strategy).
Initially, these weights don’t encode any useful knowledge or understanding. They are just random numbers, so the network doesn’t "know" what is a "dangerous move" or a "winning move" at all.
2. Training Process:
The network learns by minimizing an error (loss). This is the heart of how it learns to recognize useful features like "two pieces in a row" or "three opponent pieces in a row."

Forward Pass:
When the network receives an input (a game board), it makes a prediction (like whether the next move is a "winning move" or not) based on its current weights. At this stage, it’s just applying the current weights and biases to calculate the output, which often leads to an incorrect prediction at first because the network hasn't learned anything yet.

Loss Calculation:
Next, the network compares its prediction with the true label (like whether the move is actually a winning move or not). This comparison results in an error or loss. For instance:

If the network predicted that the move is not a winning move, but the actual outcome is a winning move, the error is large.
If the prediction is closer to the truth (e.g., predicting a winning move correctly), the error is smaller.
Backpropagation and Gradient Descent:
Once the error is computed, the network then uses backpropagation to calculate how much each weight in the network contributed to the error. The basic steps are:

Gradient Calculation: The network calculates the gradient (the partial derivative of the loss function) with respect to each weight in the network. This tells the network how much each weight should be adjusted to minimize the error.

Weight Update: Using an optimization algorithm like gradient descent, the weights are adjusted slightly to reduce the error. This update process is repeated for every training example in the dataset.

The key thing to understand here is that the network doesn't know beforehand that "two pieces in a row" is a critical feature. It figures this out during training by adjusting its weights in a way that reduces error over time.

3. How Simple Features Like "Two Pieces in a Row" Emerge:
As the network trains, it begins to learn features that correlate with the output label (such as "winning move" or "danger"). Here's why the "two pieces in a row" or "three opponent pieces in a row" patterns might emerge:

Winning Move Patterns: If the task is to predict a winning move, then the network may start to learn that when two pieces are in a row with an empty spot on one side, it can lead to a winning move.

Danger Patterns: If the task is to predict danger (e.g., preventing an opponent from winning), the network will learn that three opponent pieces in a row with an open spot on one side represents a dangerous situation (i.e., the opponent can win if the spot is filled).

But the network doesn't start out knowing these concepts. Instead, through gradient descent, it learns to give high importance to certain features (like pairs, triplets, or gaps in the board) because they help reduce the loss function over time.

4. Why Early Layers "Recognize" Simple Patterns Like Pairs or Triplets:
In convolutional neural networks (CNNs), which are often used for tasks like image recognition (and could be applied to a game board), early layers are designed to detect simple spatial features:

Local Patterns: Early convolutional layers typically detect local patterns (such as edges, corners, or small shapes) because they use small filters (kernels) to look at small sections of the input.
Learning Simple Structures: In the case of a Gomoku game board, these filters could start to identify basic structures like pairs or triplets of pieces because these are the basic building blocks for more complex strategies (like winning or blocking).
The reason they emerge in early layers is because the filters in those layers are small (e.g., 3x3 or 5x5 grids) and local. As a result, they are able to learn the simplest, most common patterns in the input data.

For instance:

A filter might learn to detect two pieces in a row (a pair) in a specific direction.
Another filter might detect three pieces in a row or detect gaps between pieces.
5. Feature Hierarchy and Deep Layers:
As the network deepens, it begins to combine these simple features learned in the earlier layers to build more abstract representations of the board. For example, deeper layers may combine pairs and triplets of pieces into more strategic concepts like:

A line of 4 pieces with a gap on one side could represent a winning move.
A line of 3 opponent pieces with a gap on one side could represent danger (the opponent could win with the next move).
The network learns these abstract concepts by combining simple features from the earlier layers. The deeper layers gradually build a hierarchy of features, which leads to complex understanding like "winning move" or "danger."

Summary:
The neural network doesn't "know" that certain simple configurations, like "two pieces in a row," are important at the outset. However, through training and the process of backpropagation, the network gradually learns to assign importance to certain features in the input (like pairs or triplets) that minimize the error in predicting the correct output (like "winning move" or "danger"). These features emerge because they help improve the network's performance in predicting the right outcomes.

As the network deepens, it combines these simpler features into more complex representations, allowing it to recognize higher-level strategies like winning or defensive moves.

Let me know if you need more clarification!