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
import utils as u
from argparse import ArgumentParser
import time
import sys
from os import path

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

def predict(list_of_outputs):
    """
    This function will convert the outputs from forward_pass() to a <list> of label indexes
    """
    # take the softmax of each timestep
    # note: this step isn't actually necessary as the argmax of the raw outputs will come out the same
    # but the softmax is more "interpretable" if needed for debugging
    pred_probs = [dy.softmax(o) for o in list_of_outputs]
    # convert each timestep's output to a numpy array
    pred_probs_np = [o.npvalue() for o in pred_probs]
    # take the argmax for each step
    pred_probs_idx = [np.argmax(o) for o in pred_probs_np]
    return pred_probs_idx

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
    # project each timestep's hidden output to size of labels
    rnn_outputs = [dy.transpose(W) * h + b for h in rnn_hidden_outs]
    return rnn_outputs

def check_score(pred, true_y):
    return 1 if pred == true_y else 0

def check_sentence_score(sentence_scores):
    return 0 if 0 in sentence_scores else 1

def get_accuracy(flat_list_of_scores):
    return float(sum(flat_list_of_scores) / len(flat_list_of_scores) * 100)

def evaluate(nested_preds, nested_true):
    flat_scores = []
    sentence_scores = []
    for i in range(len(nested_true)):
        scores = []
        pred = nested_preds[i]
        true = nested_true[i]
        for p,t in zip(pred,true):
            score = check_score(p,t)
            scores.append(score)
        sentence_scores.append(check_sentence_score(scores))
        flat_scores.extend(scores)
    overall_accuracy = get_accuracy(flat_scores)
    sentence_accuracy = get_accuracy(sentence_scores)
    return overall_accuracy, sentence_accuracy

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
        for k in range(len(batch_tokens)):
            # prepare input: words to indexes
            seq_of_idxs = words2indexes(batch_tokens[k], w2i)
            # make a forward pass
            preds = forward_pass(seq_of_idxs, embedding_parameters, pW, pb, RNN_unit)
            label_preds = predict(preds)
            all_predictions.append(label_preds)
    return all_predictions


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
            for k in range(len(batch_tokens)):
                # prepare input: words to indexes
                seq_of_idxs = words2indexes(batch_tokens[k], w2i)
                # make a forward pass
                preds = forward_pass(seq_of_idxs, embedding_parameters, pW, pb, RNN_unit)
                # calculate loss for each token in each example
                loss = [dy.pickneglogsoftmax(preds[l], batch_labels[k][l]) for l in range(len(preds))]
                # sum the loss for each token
                sent_loss = dy.esum(loss)
                # backpropogate the loss for the sentence
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

def main(datapath, train=None, test=None, num_epochs=2, batch_size=256, embedding_approach='random', embedding_size=300):

    if train is None and test is None:
        print("Either train or test!")
        sys.exit()

    ### initialize empty model
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
    train_tokens, train_labels, _, _, test_tokens, test_labels = u.import_ptb(datapath)
    ################################################################

    if embedding_approach == "pretrained":
        emb_matrix_pretrained, w2i_pretrained = u.load_pretrained_embeddings(
            path.join(datapath, "pretrained_embeddings.txt"),
            take=10000
        )
        embedding_parameters = RNN_model.lookup_parameters_from_numpy(emb_matrix_pretrained)
        embedding_size = emb_matrix_pretrained.shape[1] ## Rewriting `embedding_size`
        w2i = w2i_pretrained  # ensure we use the correct lookup table
        print("embedding matrix shape: {}".format(emb_matrix_pretrained.shape))

    elif embedding_approach == "random":
        #### randomly initialized embeddings
        w2i_random = u.build_w2i_lookup(train_tokens)
        embedding_parameters = RNN_model.add_lookup_parameters((len(w2i_random) + 1, embedding_size))
        w2i = w2i_random  # ensure we use the correct lookup table
    else:
        raise Exception("Choose a proper embedding approach")

    ###### CHOOSE HERE which approach you want to use. ######
    # RNN_unit = dy.LSTMBuilder(num_layers, embedding_size, hidden_size, RNN_model)
    RNN_unit = dy.GRUBuilder(num_layers, embedding_size, hidden_size, RNN_model)
    ################################################################

    ## convert the labels to ids
    l2i = u.labels_to_index_map(train_labels)
    train_labels = [[l2i[l] for l in sent] for sent in train_labels]
    test_labels = [[l2i[l] for l in sent] for sent in test_labels]

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

    if train is not None:
        modelPath = train
        trainAlgo(train_tokens, train_labels, num_epochs, num_batches_training,
                                                batch_size, w2i, embedding_parameters, pW, pb, modelPath, RNN_unit, trainer, RNN_model)

    if test is not None:
        modelPath = test
        final_predictions = testAlgo(test_tokens, test_labels, num_batches_testing, batch_size, w2i, embedding_parameters, pW, pb, modelPath, RNN_unit, RNN_model)
        overall_accuracy, sentence_accuracy = evaluate(final_predictions, test_labels)
        print("overall accuracy: {}".format(overall_accuracy))
        # print("sentence accuracy (all tags in sentence correct): {}".format(sentence_accuracy))


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

###################
    ###### HOW TO RUN #########
    # python nlp_hw_pos_rnn.py --datapath `pwd` --train temp.model --num_epochs 1 --embedding_approach pretrained
    # usage: Training a Neural Network for POS tagging and evaluating a trained model
    # [-h][--datapath DATAPATH] [--train TRAIN][--test TEST]
    # [--num_epochs NUM_EPOCHS][--batch_size BATCH_SIZE]
    # [--embedding_approach EMBEDDING_APPROACH]
    # [--embedding_size EMBEDDING_SIZE]

    # NOTE: Remembed to provide the --embedding_approach=pretrained during test when trained on a pre-trained word embedding
    # NOTE: SEE TOP to RUN on GPU/GPU

    ## Uses and repurposes much of Michael Capizzi's code of the tutorial on Dynet
    ## Model attached : pos.nn.5.model
###################
