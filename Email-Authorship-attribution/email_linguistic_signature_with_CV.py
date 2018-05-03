
# coding: utf-8

# In[12]:


import numpy as np
import dynet_config
dynet_config.set(
    mem=4096,
    autobatch=True,      # utilize autobatching
    random_seed=1978     # simply for reproducibility here
)
##### To run on the GPU
### NOTE: Comment the line below if GPU is not available
dynet_config.set_gpu()
#####
import dynet as dy
#write your own!import utilsCapizzi as u
from argparse import ArgumentParser
import time
import sys
from os import listdir, path
from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize
#nltk.download('punkt')


# In[13]:


#Here top 10 frequent emails are chosen, it can be parametrized and changed
def filter_freq_email(path_to_files):
    email_files_list = listdir(path_to_files)
    dictionary_ = {}
    for file in email_files_list:
        fname = file.split("_")
        if len(fname) > 2:
            continue
        if int(dictionary_.get( fname[0], 0)) < int(fname[1]):
            dictionary_[fname[0]] = fname[1]
    top_files= sorted([(k,v) for (k,v) in dictionary_.items()],key= lambda item: (int(item[1]),item[0]), reverse=True)[:10]
    top_files_filtered = [l[0] for l  in top_files]
    return top_files_filtered


# In[14]:


def parse_email_data(path_to_files):
    tokens = []
    labels = []
    filter_email_files_list = filter_freq_email(path_to_files)
    email_files_list = listdir(path_to_files)
    for file in email_files_list:
        if(file.split("_")[0] in filter_email_files_list):
            with open(path_to_files + file, 'r') as f:
                email_body = f.read()
                sent_tokenize_list = sent_tokenize(email_body)
                words_of_sentence_list = []
                for s in sent_tokenize_list:
                    words_of_sentence = word_tokenize(s)
                    words_of_sentence_list.append(words_of_sentence)
                tokens.append(words_of_sentence_list)
                labels.append(file.split("_")[0])
            f.close()
        #print(tokens, " label =", labels)
    return tokens, labels
        
def import_emails(path_to_root):
    print("loading training data")
    train_sentences, train_labels = parse_email_data(path.join(path_to_root, "taylor-m/"))
    #ToDo - cross validation
    #return train_tokens, train_labels, dev_tokens, dev_labels, test_tokens, test_labels
    #print("train_sentences =", len(train_sentences), " train_labels = ",len(train_labels))
    #print("train_sentences =",train_sentences)
    return train_sentences, train_labels


def load_pretrained_embeddings(path_to_file, take=None):
    embedding_size = None
    embedding_matrix = None
    lookup = {"<unk>": 0}
    c = 0
    with open(path_to_file, "r") as f:
        for line in f:
            if c == 0:
                # check for header line
                if len(line.split()) == 2:
                    c = 1
                    pass
            else:
                # check for delimiter
                if "\t" in line:
                    delimiter = "\t"
                else:
                    delimiter = " "
                if (take and c <= take) or not take:
                    # split line
                    line_split = line.rstrip().split(delimiter)
                    # extract word and vector
                    word = line_split[0]
                    vector = np.array([float(i) for i in line_split[1:]])
                    # get dimension of vector
                    embedding_size = vector.shape[0]
                    # add to lookup
                    lookup[word] = c
                    # add to embedding matrix
                    if np.any(embedding_matrix):
                        embedding_matrix = np.vstack((embedding_matrix, vector))
                    else:
                        embedding_matrix = np.zeros((2, embedding_size))  #unknown word embedding is all 0s
                        embedding_matrix[1] = vector
                    c += 1
    return embedding_matrix, lookup

def build_w2i_lookup(training_corpus):
    lookup = {"<unk>": 0}
    c = 1
    for doc in training_corpus:
        for word in doc:
            word = word.lower()
            if word not in lookup:
                c += 1
                lookup[word] = c
    return lookup

def labels_to_index_map(all_training_labels):
    dict_ = {}
    c = 0
    for label in all_training_labels:
        if label not in dict_:
            dict_[label] = c
            c+=1
    return dict_

#print(labels_to_index_map(train_labels))


# In[15]:


def words2indexes(seq_of_words, w2i_lookup):
    """
    This function converts our sentence into a sequence of indexes that correspond to the rows in our embedding matrix
    :param seq_of_words: the document as a <list> of words
    :param w2i_lookup: the lookup table of {word:index} that we built earlier
    """
    seq_of_idxs = []
    for w in seq_of_words:
        w = w.lower()            # lowercase
        i = w2i_lookup.get(w, 0) # we use the .get() method to allow for default return value if the word is not found
                                 # we've reserved the 0th row of embedding matrix for out-of-vocabulary words
        seq_of_idxs.append(i)
    return seq_of_idxs

#def check_score(pred, true_y):
#    return 1 if pred == true_y else 0

def check_sentence_score(pred,true_y):
    return 1 if pred == true_y else 0

def get_accuracy(flat_list_of_scores):
    return float(sum(flat_list_of_scores) / len(flat_list_of_scores) * 100)

def evaluate(nested_preds, nested_true):
    email_scores = []
    for i in range(len(nested_true)):
        #print("nested_preds[i] =", nested_preds[i] , " nested_true[i] =", nested_true[i] )
        email_scores.append(1 if nested_preds[i] == nested_true[i] else 0)
    email_accuracy = get_accuracy(email_scores)
    return email_accuracy

def predict(output):
    """
    This function will convert the outputs from forward_pass() to a <list> of label indexes
    """
    # take the softmax of each timestep
    # note: this step isn't actually necessary as the argmax of the raw outputs will come out the same
    # but the softmax is more "interpretable" if needed for debugging
    pred_prob = dy.softmax(output)
    # convert each timestep's output to a numpy array
    pred_prob_np = pred_prob.npvalue()
    # take the argmax for each step
    pred_prob_idx = np.argmax(pred_prob_np) 
    return pred_prob_idx

def forward_pass(x, embedding_parameters, pW, pb, RNN_unit):
    """
    This function will wrap all the steps needed to feed one sentence through the RNN
    :param x: a <list> of indices
    """
    # convert sequence of ints to sequence of embeddings
    input_seq = [embedding_parameters[i] for i in x]   # embedding_parameters can be used like <dict>
    # convert Parameters to Expressions
    W = dy.parameter(pW)
    b = dy.parameter(pb)
    # initialize the RNN unit
    rnn_seq = RNN_unit.initial_state()
    # run each timestep through the RNN
    rnn_hidden_outs = rnn_seq.transduce(input_seq)
    #print("rnn_hidden_outs = ", rnn_hidden_outs)
    #print("rnn_hidden_outs  len = ", len(rnn_hidden_outs))
    return rnn_hidden_outs[-1]

def trainAlgo(train_tokens, train_labels, num_epochs, num_batches_training, batch_size, w2i, embedding_parameters, pW, pb, modelPath, RNN_unit, trainer, RNN_model):
    # i = epoch index
    # j = batch index
    # k = sentence index (inside batch j)
    # l = token index (inside sentence k)

    epoch_losses = []
    overall_accuracies = []
    sentence_accuracies = []

    start_train_time = time.clock()
    for i in range(num_epochs):
        epoch_loss = []
        print("Starting epoch: " + str(i+1))
        start_epoch_time = time.clock()
        for j in range(num_batches_training):
            # begin a clean computational graph
            dy.renew_cg()
            # build the batch
            batch_tokens = train_tokens[j * batch_size:(j + 1) * batch_size]
            batch_labels = train_labels[j * batch_size:(j + 1) * batch_size]
            # iterate through the batch
            for each_email_index in range(len(batch_tokens)):
                hidden_representation_list =[]
                for k in batch_tokens[each_email_index]:
                    # prepare input: words to indexes
                    seq_of_idxs = words2indexes(k, w2i)
                    # make a forward pass
                    hidden_representation = forward_pass(seq_of_idxs, embedding_parameters, pW, pb, RNN_unit)
                    hidden_representation_list.append(hidden_representation)
                    #print (batch_labels[k])
                #np_hidden_representation_list = np.array(hidden_representation_list)
                #np.mean(np_hidden_representation_list,axis=0, dtype=np.float64
                #Take average of all the hidden representation of the sentences of that email
                hr = dy.average(hidden_representation_list)
                # convert Parameters to Expressions
                W = dy.parameter(pW)
                b = dy.parameter(pb)
                # project each timestep's hidden output to size of labels
                rnn_output = dy.transpose(W) * hr + b
                #print("rnn_output = ", rnn_output)
                #print("batch_labels[each_email_index]", batch_labels[each_email_index])
                # calculate loss for each email 
                sent_loss = dy.pickneglogsoftmax(rnn_output, batch_labels[each_email_index]) 
                # sum the loss for each token
                #sent_loss = dy.esum(loss)
                # backpropogate the loss for the email
                sent_loss.backward()
                trainer.update()
                epoch_loss.append(sent_loss.npvalue())
                # print("epoch: " + str(i+1) + " batch: " + str(j+1) + " loss: " + str(np.average(epoch_loss)) + "\r")
        # record epoch loss
        epoch_losses.append(np.sum(epoch_loss))
        print("Train loss after epoch: " + str(i + 1) + " loss: " + str(np.average(epoch_loss)))
        print ("Epoch " + str(i+1) + " Time Taken: " + str(time.clock()-start_epoch_time))
        # get accuracy on test set
        # # print("Train loss after epoch {}".format(i + 1))
        # epoch_predictions = test(train_tokens, train_labels, num_batches_training, w2i, embedding_parameters, pW, pb)
        # epoch_overall_accuracy, epoch_sentence_accuracy = evaluate(epoch_predictions, train_labels)
        # overall_accuracies.append(epoch_overall_accuracy)
        # sentence_accuracies.append(epoch_sentence_accuracy)

    print("Training Completed. Time taken: " + str(time.clock() - start_train_time))
    print("Saving model in " + str(modelPath))
    RNN_model.save(modelPath)
    print("Done!")

def testAlgo(test_tokens, test_labels, num_batches_testing, batch_size, w2i, embedding_parameters, pW, pb, modelPath, RNN_unit, RNN_model):

    print ("Evaluating the saved model: \"" + str(modelPath) + "\" on the test data")
    print ("Loading the model ..")
    RNN_model.populate(modelPath)
    print ("Done!")

    # j = batch index
    # k = sentence index (inside batch j)
    # l = token index (inside sentence k)
    all_predictions = []

    for j in range(num_batches_testing):
        # begin a clean computational graph
        dy.renew_cg()
        # build the batch
        batch_tokens = test_tokens[j*batch_size:(j+1)*batch_size]
        batch_labels = test_labels[j*batch_size:(j+1)*batch_size]
        # iterate through the batch
        for each_email_index in range(len(batch_tokens)):
            hidden_representation_list =[]
            for k in batch_tokens[each_email_index]:
                # prepare input: words to indexes
                seq_of_idxs = words2indexes(k, w2i)
                # make a forward pass
                hidden_representation = forward_pass(seq_of_idxs, embedding_parameters, pW, pb, RNN_unit)
                hidden_representation_list.append(hidden_representation)
            hr = dy.average(hidden_representation_list)
            # convert Parameters to Expressions
            W = dy.parameter(pW)
            b = dy.parameter(pb)
            # project each timestep's hidden output to size of labels
            rnn_output = dy.transpose(W) * hr + b
            label_pred = predict(rnn_output)
            all_predictions.append(label_pred)
    return all_predictions



# In[16]:



# fold_size = int (np.ceil(len(all_labels)/10))
# print(fold_size)
# i=0
# test_t =[]
# test_l=[]
# train_l = []
# train_t = []
# while i < len(all_labels):
#     test_t = all_tokens[i:i+fold_size]
#     test_l = all_labels[i:i+fold_size]
#     train_t = all_tokens[0:i]
#     train_t.extend(all_tokens[i+fold_size:])
#     train_l = all_labels[0:i]
#     train_l.extend(all_labels[i+fold_size:])
#     print("len(test_l)= ", len(test_l), " len(train_l)= ", len(train_l), " fold_sz= ", fold_size)
#     i+=fold_size


# In[19]:


def main(datapath, train=None, test=None, num_epochs=2, batch_size=256, embedding_approach='random', embedding_size=300):

    if train is None and test is None:
        print("Either train or test!")
        sys.exit()
    ################################################################
    RNN_model = dy.ParameterCollection()

    ################################################################
    # HYPERPARAMETERS
    ################################################################
    # size of word embedding (if using "random", otherwise, dependent on the loaded embeddings)
    # embedding_size = 300
    # size of hidden layer of `RNN`
    hidden_size = 200
    # number of layers in `RNN`
    num_layers = 1
    # type of trainer
    trainer = dy.SimpleSGDTrainer(
        m=RNN_model,
        learning_rate=0.01
    )
    ################################################################
    ## Load the training and test data
    all_tokens, all_labels = import_emails(datapath)  #Add test_tokens, test_labels
    ################################################################
    #print("train_tokens = ", train_tokens[:10], "train_labels = ", train_labels[:10])
    if embedding_approach == "pretrained":
        emb_matrix_pretrained, w2i_pretrained = load_pretrained_embeddings(
            path.join(datapath + "../../hw/", "pretrained_embeddings.txt"),
            take=10000
        )
        embedding_parameters = RNN_model.lookup_parameters_from_numpy(emb_matrix_pretrained)
        embedding_size = emb_matrix_pretrained.shape[1] ## Rewriting `embedding_size`
        w2i = w2i_pretrained  # ensure we use the correct lookup table
        print("embedding matrix shape: {}".format(emb_matrix_pretrained.shape))

    elif embedding_approach == "random":
        #### randomly initialized embeddings
        w2i_random = build_w2i_lookup(train_tokens)
        embedding_parameters = RNN_model.add_lookup_parameters((len(w2i_random) + 1, embedding_size))
        w2i = w2i_random  # ensure we use the correct lookup table
    else:
        raise Exception("Choose a proper embedding approach")

    ###### CHOOSE HERE which approach you want to use. ######
    # RNN_unit = dy.LSTMBuilder(num_layers, embedding_size, hidden_size, RNN_model)
    RNN_unit = dy.GRUBuilder(num_layers, embedding_size, hidden_size, RNN_model)
    ################################################################
    #10 fold Cross validation
    fold_size = int (np.ceil(len(all_labels)/10))
    print(fold_size, " fold size of a 10-Fold Cross validation")
    i=0
    test_tokens =[]
    test_labels=[]
    train_labels = []
    train_tokens = []
    email_accuracy_folds = []
    while i < len(all_labels):
        test_tokens = all_tokens[i:i+fold_size]
        test_labels = all_labels[i:i+fold_size]
        train_tokens = all_tokens[0:i]
        train_tokens.extend(all_tokens[i+fold_size:])
        train_labels = all_labels[0:i]
        train_labels.extend(all_labels[i+fold_size:])
        print("len(test_l)= ", len(test_labels), " len(train_labels)= ", len(train_labels), " fold_sz= ", fold_size)
        i+=fold_size
    
        ## convert the labels to ids
        l2i = labels_to_index_map(train_labels)
        #print("len(l2i) = ",l2i)
        train_labels = [l2i[l] for l  in train_labels]
        #print("After mapping train_labels = ", train_labels[:10] )
        test_labels = [l2i[l] for l in test_labels]


        # training hyperparams
        # batch_size = 256
        num_batches_training = int(np.ceil(len(train_tokens) / batch_size))
        num_batches_testing = int(np.ceil(len(test_tokens) / batch_size))
        # num_epochs = 1

        ## Projection layer
        # W (hidden x num_labels)
        pW = RNN_model.add_parameters(
            (hidden_size, len(list(l2i.keys())))
        )

        # b (1 x num_labels)
        pb = RNN_model.add_parameters(
            (len(list(l2i.keys())))
        )
        print("train_tokens len = ", len(train_tokens), "train_labels len =", len(train_labels))
        if train is not None:
            modelPath = train
            trainAlgo(train_tokens, train_labels, num_epochs, num_batches_training,
                                                    batch_size, w2i, embedding_parameters, pW, pb, modelPath, RNN_unit, trainer, RNN_model)

        if test is not None:
            modelPath = test
            final_predictions = testAlgo(test_tokens, test_labels, num_batches_testing, batch_size, w2i, embedding_parameters, pW, pb, modelPath, RNN_unit, RNN_model)
            email_accuracy = evaluate(final_predictions, test_labels)
            print("Email overall accuracy : {}".format(email_accuracy))
            email_accuracy_folds.append(email_accuracy)
    print("Average over 10 folds = ", float(sum(email_accuracy_folds)/ len(email_accuracy_folds)))


if __name__ == "__main__":

    parser = ArgumentParser("Training a Neural Network for POS tagging "
                            "and evaluating a trained model")
    parser.add_argument("--datapath", type=str,
                        default='/home/pooja/dynet/hw', help="Path which contains the dir PTBSmall")
    parser.add_argument("--train", type=str,
                        default=None, help="Command to train the model. Model saved in the param given")
    parser.add_argument("--test", type=str,
                        default=None, help="Test the saved model on testing data")
    parser.add_argument("--num_epochs", type=int,
                        default=2, help="Number of epochs on the training data")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="Size of each minibatch")
    parser.add_argument("--embedding_approach", type=str,
                        default='random', help="[pretrained|random pretrained] pre-trained embeddings or initialize the embeddings by random. "
                                               "For pre-trained embeddings case the embedding file is presumed to be "
                                               "in the `datapath` and is named `pretrained_embeddings.txt`")
    parser.add_argument("--embedding_size", type=int,
                        default=300, help="Size of the emebddings .. Do not use this when loading pre-trained embeddingsß")

    args = parser.parse_args()
    print('-----------------------------------------')
    print('ARGUMENTS TO THE PROGRAM')
    print("-----------------------------------------")
    print(args)
    print("-----------------------------------------")

    main(**vars(args))



# In[ ]:




