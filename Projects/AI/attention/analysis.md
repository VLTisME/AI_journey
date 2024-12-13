# Analysis

## Layer TODO, Head TODO

TODO

Example Sentences:
- TODO
- TODO

## Layer TODO, Head TODO

TODO

Example Sentences:
- TODO
- TODO


## My own explanations:
- [Regarding attention] During training process, we need both inputs and outputs and it will train to know the relationship between tokens. The training process will normally be like, sequence X to sequence Y, then the model will learn, for each token, pay attention to what token in the output? Then after all the training, we have something like the 'attention board', or let's say each token will have a logits, representing the probability of the next word. Then in inference step, when we feed in a sentence, we will use self-attention uhmm? to analyze the relationship between each token in the INPUT sentence only, then it utilizes the attention board which mentioned earlier before, to predict a new word, gradually to a new sentence. In BERT, the output is a single logits, at each index of the logits is the tensorflow tensor which represents the probalities of all the words in the BERT's dict. Then in BERT, at the MASK token index, we will pick the highest one to predict the word for MASK. One more thing to pay attention is that the logits will have the shape (bacth_size, sequence_length, vocab_size). Batch_size represents the number of sentences... let's say? ok so if only input one sentence, it is logits[0]. Then logits[0, MASK_TOKEN_INDEX] will be the tensors representing the probabilities of all words in vocab_size. Imagine in Transformer? you also saw the logits, at each token index will be sth like the vector stands for the probabilities of all words that could emerge after that token... Cool, need to rewatch 3B1B for better understanding now :). In transformer, in the last ? ((self)attention layer)? after self-attention, pick the index of the last word, then pick the highest probability of the logits[last_word_index], we will have the next word, then feed it into the NN again,...

questions:
- nma sao trong video cua 3b1b, attention block lai la nhung tu trong cung 1 cau anh huong nhau nhi?

steps in a transformer:
- take an input, tokenize it, embed it to vectors
- feed it into the network, it will go through attention block, where each token affects the meanings of others
- now go to the MLP step, each vector does not talk to each other.
- at that step, it is asked a lot of questions based on the vector representing the embeded token
remember both of these blocks contain lots of matrix multiplication :) and they run parallelly
- between them have normalization steps
- repeat again and again
- last vector will represent the probabilities of the next word, pick the highest, then re-feed into the network

make a chatbot?
- first have a system prompt, to tell the model that it is interacting with the user
- next is the user's input

## wut?
- paremeters for embedding must also be learned even in the first step (first it will be randomly generated). Then in attention (tokens affect tokens, questions are asked to examine the relationship between tokens) or dense layer (being asked questions), it learns and each token will contain richer meaning of the sentence. Then using the actual outputs to backpropagation to modify the parameters. remember that the We (matrix for embedding) has all words in the model's dict, but when feed into the model, we feed the sentences of data as inputs because we are limited by the context size, so u cant feed the whole dict of the model


Key Point:
The model looks up embeddings from We only for the words in the input sentence.
For instance, if the word index "2" maps to "is," the embedding for "is" (We[2]) is fetched from the embedding matrix.
The model only interacts with part of the vocabulary during each forward pass (based on the current input). Not all words in the vocabulary are used for every training step.
Training Updates (Backpropagation):

During backpropagation, only the embeddings corresponding to the words in the input sentence will be updated.
Other embeddings in We remain unchanged until their corresponding words appear in future training samples.

ur explanation: u have a big dict, but during training process, due to the limit of the context size, u take only several sentences from the dataset as inputs and outputs (for ex, inputs: "handsom", outputs: "handsome"), then u take all the embeded vectors of words appeared in the sentences u pick, then feed into the Network, train it using backpropagation,... then u have the trained embeded vectors for words appeared in sentences you choose, then these vectors will be updated to the big dict. Words that don't appear remain unchanged.




ok ok when we have the rich embeded meaning vector for each token, then for ALL tokens there will be a matrix called Wu. If we take Wu multiply with the embeded vector, we will get the logits [logits is decoding step, logits = embeded vector of a token * Wu], which are the probabilities of the next words that could appear after that token. :) then we can apply the softmax... preparing for the next step or simply just let it there :D. Take the last embeded vector and multiply with Wu, we'll get the logits, then normalize it 

oh shit this part is important to pay attention:
1/ For inference, yeah u basically just need to multiply the Wu with the last embedded vector to predict the next word!
2/ But for training process, u have to multiply the Wu with all the embedded vectors to predict the word comes next for EACH TOKEN. Simply because you are modifying the parameters of Wu here, and you have the inputs as well as outputs, so multiply Wu with each vector to calculate the loss in the training process (for example because we are training so we know what word should come next after a particular word), then sum all the loss of each logit of each token, then backpropagation to tuning the parameters of Wu.
3/ But I wonder, when going from this layer to the next layer, not from the last layer to the output of the model, do we need to multiply with the Wu? If yes, for what?
---> OH YES the answer is indeed NO as I understand :v. Because from this step to the next (lets say Attention block to MLP), we don't conclude any official prediction of the next word bla bla,... it is still in riching the meaning of each word bla bla (or lets say it is understanding more about the context of the sentence (using self-attention (questions are asked here, they are the matrixes keys and queries), feedforward netword (this one operates independently on each token's embedding, and u know that each token's embedding already contain the meanings of other tokens, so passing through this MLP block will further enrich, learn more the meaning of that embedding vectors, AND THIS STEP also has questions to ask too, imagine like there are 4 * dimensions_of_embedvectors questions (each represents a neuron) are asked!!!. More specifically, lets consider a vector embedding that is fed into the MLP, now in your mind, imagine an only one single DNN to use for all embedding vectors, then it has a lot of edges right? these are parameters to be tuned - and The MLP weights are shared across all tokens - not as u thought one MLP for one token but all tokens pass only one MLP :v)) BEFORE CONCLUDING - which is the decoding step). So yeah at the very last step, we will use projection matrix (Wu) to multipply with the super-rich meaning embedded vectors to predict the next word, we only multiply with Wu at this last step (decoding step) to then compare with the actual output to calculate the sum loss to tune the Wu again. 

- Ngoai ra con co cai Residual Connections nua, thi simply thay vi cai output cua DNN or Attention block la output sau khi qua cac operations cua DNN or Attention block thi bh no ket hop them cai input embeddings vao luon nua :v. Nhung ma [pay attention] la cai input chac chac la dua vao tu thang Attention Block ma co the la mot thang nao do truoc do rat xa. For ex:

Input Embeddings → Self-Attention Block → Add Residual (Input + Attention Output) → LayerNorm  

   ^                                                                |  
   |                                                                |  
   └---------------------------- Residual Connection <--------------┘

or:

Attention Block Output → MLP (Position-wise FFN) → Add Residual (Attention Block Output + MLP Output) → LayerNorm

- Layer Output = LayerNorm(Layer Input + Layer Transformation Output)
- The layer output is the sum of: the layer's input (unchanged) + the output of the current layer's operations (obv changed during DNN layers).
- nch don gian chi la thay vi return cai output thi bh return cai output + input cua thang nao do :v
vd nhu trong self-attention:
- gia su x la input cua embedding vector cua token nao do di, roi output cua token do la F(x). Thay vi tra ve F(x) thi bh tra ve F(x) + x
vd nhu trong MLP, xet mot token nao do di , thay vi no tra ve embedding vector cua token do khong thoi thi bh no + them output cua thang attention [RIGHT BEFORE THAT MLP] vao nua 
- VD nhu output cua MLP la mot vector embed thang 'basketball' va input la 1 vector embed 'Michale Philipse" thi output real se + hai cai vector do lai thanh mot vector mang y nghia cua all 'Michale Philipse Basketball' roi feed vao Attention block tiep va se trai qua 96 head khac nhau tiep



- Each attention block has many heads inside called multi-head attention (GPT-2, it's 96 attention heads in each block) and each head has it's own own query, key, and value matrices (Wq, Wk, Wv). Each of these (Wq, Wk, Wv) serves different context meanings.
let's call the embedding vectors are **E**: E1, E2,... E_contextsize
the Wq matrix contain weights so that when taking Wq * Ei we have questions vector
multiply every Ei for Wq we got: Ei * Wq = Qi (or E * Wq = Q (big matrix, formed from all Qi vectors))
Similarly, we have Wk, and Wv
and Ei * Wk = Ki (or E * Wk = K)
and Ei * Wv = Vi (or E * Wv = V)
simply speaking: we have embedding vector Ei, we have matrix Wq, if we take Ei * Wq, we have the questions, lets call it Qi. For example: if we take an embedding vector Ei, which is for example a word of "creature", then after multiply Ei with Wq, then we got the Qi and the Qi[0] (first question) is "is there any adjective before my position". The answer of that question is stored in Ki :), which can be taken from taking Ei * Wk -> Ki. Ok let's say Q5[0] = "Is there any adjective before 5-th position", then there is a K3[2] that says "Yes I am an adjective" :). How to know if Kj[y] is the answer of Qi[x]? Simply just take the DOT PRODUCT! Because each of these Qi[x] is a vector in a high-dimensional space so when the two vectors stay near, they relate to each others!!! Then take the softmax of the dot product to normalize the numbers. After that, after having all these dot products, then softmax, masking, what we want is to update the initial embeding vectors to make it enrich more meanings... we will update it using Wv matrix. Now for example, take one single initial embedding vector call Ei, multiply it with matrix Wv, you will get a matrix Vi. ok now the question is how much of Vi should be added to other Ej's to enrich the meaning of the i-th word to the j-th word? Simply by taking the dot product calculated before, multiply it with Vi!!! For example: how much of the i-th word affect the j-th word = How much should we add Vi in the Ej to make it contains information about i-th word? It is Vi * dot_product(Ki, Qj)  !!!!!!!!!!!!!!!!!!! (Ki contains answers for Qj's questions) (Vi + Ej will be the original formula but you know that the each word affects a certain amount to another word so we must take Vi * how_much_it_relates_to_j_word + Ej). Then take Ej and add all these Vi * dot_product(Ki, Qj) (called delta_Vj) for every i into Ej, we will get new E'j, which is enriched with context meaning! However, there are so many heads inside one attention block so you add all delta_Vj of every head (sigma delta_Vj_headk for every k from 1 to 96) and then take that sum of delta_Vj, add to Ej..

Bonus: that Ki * Qj = value, then softmax the value, that value means how much is the i-th word affect or relate to the j-th word!
However, as you can see, the Value matrix (Wv) is too big, and not efficient especially running paralelly so many heads --> reduce its size from (12,288; 12,288) to (12,288; 128) * (128; 12,288).
(128; 12,288) is value down matrix and the other is value up. The value down matrix will map the large embedding vector down to a smaller space (image 3D to 2D). That is called Low Rank Transformation
Plus: GPT-2 is used to predict the next word so it has self-attention head but when it comes to translating system or audio to text, the attention head has cross-attention, which we don't want to eliminate any of the dot products because the latter can affect the former ones, and no masking.

- in GPT-2, we will utilize the last layer's all embedding vector to train a lot of times all at once right? so yeah it saves time and makes the model more powerful, and so yeah we don't want the next words to influence the previous words (otherwise the prediction will be biased so not true (its like the latter words give answers to earlier words so obviously the prediction would be biased), we are simultaneously train one sentence but parsed into smaller prefix sentences ya ya to calculate the loss) so we have to eliminate half of the attention board to prevent it from affecting previous words.

- the query and key matrices obviously will create an attention board of size context_size * context_size right...? so yeah lmao big LLMs face this problem. Here raises the question: How to scale it? Is it scalable?
- usually, biases are put in another vector, each matrix, or MLP, has bias vector.
- parameters for kernels in CNN also learned during training process.
- logits roi moi toi normalization
- by gradually passing information in attention block in each time it takes the input data, the embedding vectors enriched more meanings and yea it can last the meaning in the model for so long :) its like in a paragraph, first take several sentences and attention, then even when later consider the last sentences, it still has the information about these first sentences. Simply talking: the more you train, the richer in meaning the embedding vectors contain.

[regarding embedding vectors]: Yes you can train the model to have the embedding matrix BUT you can also pretrain the embedding matrix using Word2vec or FastText :)

