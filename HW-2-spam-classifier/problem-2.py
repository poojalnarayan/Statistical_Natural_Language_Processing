import numpy as np
import ast
import matplotlib.pyplot as plt
import sys


def sigmoid(z):
    return 1/(1+ np.exp(-z))

def logistic_regression(X, Y, number_of_epocs, size_of_minibatch, learning_rate):
    theta = np.zeros(len(X[0]))
    correct_answers=0
    for epoc in range(number_of_epocs):
        #todo Random shuffle of data
        i=0
        while i < len(X):
            delta =0
            for j in range(size_of_minibatch):
                if i+j >= len(X):
                    break
                prediction= sigmoid(np.dot(theta.T,X[i+j] ))
                #print("prediction = ", prediction, "Y= ", Y[i+j])
                if( (prediction > 0.5 and  Y[i+j] == 1) or (prediction <= 0.5 and  Y[i+j] == 0)):
                    correct_answers+=1

                delta += (learning_rate) *(np.dot(Y[i+j] - prediction, X[i+j]))

            delta /= size_of_minibatch
            theta=theta+delta
            i = i+size_of_minibatch

    #print(theta.shape[0],theta)
    print("% = ", (correct_answers/(len(Y) * number_of_epocs))* 100)
    #save model

    f = open('save_model.txt', 'a')
    #value = number_of_epocs, size_of_minibatch, learning_rate
    f.write(str(number_of_epocs)+'\n')
    f.write(str(size_of_minibatch)+'\n')
    f.write(str(learning_rate)+'\n')
    #f.write(str(value).__add__(',\n'))
    #np.savetxt(f, theta, fmt='%f')  #, delimiter=',', fmt='%float64'
    i=0
    while i < theta.shape[0]:  #unpacking a tuple
        f.write(str(theta[i])+'\n')
        i+=1
    f.close()

def evaluation():
    F = open("save_model.txt", 'r')
    dic = F.readline().strip('\n')
    dic = ast.literal_eval(dic)
    #print(dic)
    number_of_epocs = F.readline().strip('\n')
    size_of_minibatch = F.readline().strip('\n')
    learning_rate = F.readline().strip('\n')
    #print(number_of_epocs,size_of_minibatch,learning_rate)
    theta_arr =[]
    for line in F:
        theta_arr.append(line.strip('\n'))

    #print(theta_arr)
    theta = np.array(theta_arr, dtype=np.float64)  #mention type else it will be in string format
    #print(theta.shape[0], "   ", theta)
    F.close()
    F1 = open("SMSSpamCollection.test", 'r')
    X = []
    Y = []
    length = theta.shape[0]
    for message in F1:
        words = message.split()
        # print(words)
        Y.append(1 if words.pop(0).__eq__('spam') else 0)
        # print("After pop...",words)
        cur_vector = np.zeros(length)
        # print("empty cur_vector ", cur_vector)
        #unigrams
        for word in words:
            w = word.strip('.')
            if not w:
                continue  # print("not word") like space etc

            if w not in dic:
                continue

            # print("w = ", w)
            # print("before update cur_vector ", cur_vector)
            cur_vector[dic[w]] = cur_vector[dic[w]] + 1
        #bigrams
        i = 0
        while i < len(words) - 1:
            w1 = words[i].strip('.')
            w2 = words[i + 1].strip('.')

            if w1+"_"+w2 not in dic:
                i+=1
                continue

            # print("w = ", w)
            # print("before update cur_vector ", cur_vector)
            cur_vector[dic[w1+"_"+w2]] = cur_vector[dic[w1+"_"+w2]] + 1
            # print("after update cur_vector ", cur_vector)
            i+=1
        X.append(cur_vector)

    # print(x)
    #print("y= ", Y)
    F1.close()
    F.close()
    i=0
    correct_answers=0
    while i < len(X):
        #print(i," X shape= ",str(X[i].shape), "theta shape= ", theta.shape)
        prediction = sigmoid(np.dot(theta.T, X[i]))
        if ((prediction > 0.5 and Y[i] == 1) or (prediction <= 0.5 and Y[i] == 0)):
            correct_answers += 1
        i += 1
    print("% = ", (correct_answers / (len(Y) )) * 100)

def run_LR(file_name):
    F = open(file_name, 'r') #SMSSpamCollection
    my_set = set()
    for message in F:
        # print(line)
        words = message.split()
        #print(words)
        words.pop(0)
        #print("After pop...",words)
        #unigrams
        for word in words:
            w = word.strip('.')
            if not w:
                continue #print("not word") or empty

            #print("Replaced ", word, " with ", w)
            my_set.add(w)

        #bigrams
        i=0
        while i< len(words)-1:
            w1 = words[i].strip('.')
            w2 = words[i + 1].strip('.')
            ''' 
            while not w1:
                if i< len(words)-1:
                    w1 = words[i].strip('.')
                    i+=1
                else:
                    break

            if i < len(words):
                w2 = words[i+1].strip('.')
                while not w1:
                    if i< len(words)-1:
                        w2 = words[i].strip('.')
                        i+=1
                    else:
                        break
            '''

            #print("Replaced ", word, " with ", w)
            #todo- capital U small u same?
            my_set.add(w1+"_"+w2)
            i+=1
    #print(my_set,len(my_set))


    #Now construct the nparray with vector size as of the dic
    F.seek(0, 0)
    dic = {k: v for v, k in enumerate(my_set)}
    #print(dic)
    F1 = open("save_model.txt", 'w')
    #F1.write('{}'.format(dic.items()))
    F1.write(str(dic).__add__('\n'))
    F1.close()

    X = []
    Y = []
    length = len(my_set)
    for message in F:
        words = message.split()
        #print(words)
        Y.append(1 if words.pop(0).__eq__('spam') else 0)
        # print("After pop...",words)
        cur_vector = np.zeros(length)
        #print("empty cur_vector ", cur_vector)
        #unigrams
        for word in words:
            w = word.strip('.')
            if not w:
                continue #print("not word") or empty

            #print("Replaced ", word, " with ", w)
            cur_vector[dic[w]] = cur_vector[dic[w]] + 1
        #bigrams
        i = 0
        while i < len(words) - 1:
            w1 = words[i].strip('.')
            w2 = words[i + 1].strip('.')


            #print("w = ", w)
            #print("before update cur_vector ", cur_vector)
            cur_vector[dic[w1+"_"+w2]] = cur_vector[dic[w1+"_"+w2]] +1
            #print("after update cur_vector ", str(cur_vector))
            i += 1
        X.append(cur_vector)

    #print(x)
    print("y= " , Y)
    F.close()
    logistic_regression(X,Y,7,3,0.1)


def run_LR_filter(file_name):
    F = open(file_name, 'r') #SMSSpamCollection
    my_dic = {}
    my_features ={}
    for message in F:
        # print(line)
        words = message.split()
        #print(words)
        spam_ham_label = words.pop(0)
        #print("After pop...",words)
        #unigrams
        for word in words:
            w = word.strip('.')
            if not w:
                continue #print("not word") or empty

            #print("Replaced ", word, " with ", w)
            if w not in my_dic:
                my_dic[w] = 1
                my_features[w] = 1 if spam_ham_label.__eq__('spam') else 0
            else:
                my_dic[w] += 1
                my_features[w] = 1 if spam_ham_label.__eq__('spam') else 0

        #bigrams
        i=0
        while i< len(words)-1:
            w1 = words[i].strip('.')
            w2 = words[i + 1].strip('.')

            #print("Replaced ", word, " with ", w)
            #todo- capital U small u same?
            w = w1+"_"+w2
            if w not in my_dic:
                my_dic[w] = 1
                my_features[w] = 1 if spam_ham_label.__eq__('spam') else 0
            else:
                my_dic[w] += 1
                my_features[w] = 1 if spam_ham_label.__eq__('spam') else 0

            i+=1
    #print(len(my_dic), my_dic)

    my_set = set()
    for key in my_dic.keys():
        if not (my_dic[key] == 1 or my_dic[key] == 2 ):#or  my_dic[key] == 3 or  my_dic[key] == 4 or my_dic[key] == 5) :
            my_set.add(key)

    #calculate the top features
    top_features = sorted([(k,v) for (k,v) in my_dic.items()],key= lambda item: (item[1],item[0]), reverse=True)[:180]
    #print(top_features)
    one_zero_arr = [(feat, my_features[feat]) for feat,count in top_features]

    #print(len(my_features), my_features)
    #print(one_zero_arr)
    #print("================================================================================================")
    #print(len(my_set),my_set)


    #Now construct the nparray with vector size as of the dic
    F.seek(0, 0)
    dic = {k: v for v, k in enumerate(my_set)}
    #print(dic)
    F1 = open("save_model.txt", 'w')
    #F1.write('{}'.format(dic.items()))
    F1.write(str(dic).__add__('\n'))
    F1.close()

    X = []
    Y = []
    length = len(my_set)
    for message in F:
        words = message.split()
        #print(words)
        Y.append(1 if words.pop(0).__eq__('spam') else 0)
        # print("After pop...",words)
        cur_vector = np.zeros(length)
        #print("empty cur_vector ", cur_vector)
        #unigrams
        for word in words:
            w = word.strip('.')
            if not w:
                continue #print("not word") or empty

            if w not in dic:
                continue

            #print("Replaced ", word, " with ", w)
            cur_vector[dic[w]] = cur_vector[dic[w]] + 1
        #bigrams
        i = 0
        while i < len(words) - 1:
            w1 = words[i].strip('.')
            w2 = words[i + 1].strip('.')

            if w1+"_"+w2 not in dic:
                i+=1
                continue

            #print("w = ", w)
            #print("before update cur_vector ", cur_vector)
            cur_vector[dic[w1+"_"+w2]] = cur_vector[dic[w1+"_"+w2]] +1
            #print("after update cur_vector ", str(cur_vector))
            i += 1
        X.append(cur_vector)

    #print(x)
    #print("y= " , Y)
    F.close()
    logistic_regression(X,Y,7,3,0.1)


def plot_fun():
    x = np.arange(1, 6, 1)
    arr= [96.588848833035, 96.67863554757629, 96.31956912028726, 96.40933572710951, 96.49910233393177]
    y = np.array(arr)

    plt.figure()
    plt.xlabel("x filter features with frequency")
    plt.ylabel("percentage accuracy on test")
    plt.title("Plots the classification accuracy for the different feature threshold values")
    plt.plot(x, y)
    plt.show()


if sys.argv[1] == "train":
    run_LR("SMSSpamCollection.train")

if sys.argv[1] == "tune":
    run_LR("SMSSpamCollection.devel")

if sys.argv[1] == "test":
    evaluation()

'''

#run_LR("SMSSpamCollection.devel")
#evaluation()
run_LR_filter("SMSSpamCollection.train")
#plot_fun()
'''