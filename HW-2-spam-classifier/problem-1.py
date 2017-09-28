import numpy as np
import ast
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
        for word in words:
            w = word.strip('.')
            if not w:
                continue  # print("not word") like space etc

            if w not in dic:
                continue

            # print("w = ", w)
            # print("before update cur_vector ", cur_vector)
            cur_vector[dic[w]] = cur_vector[dic[w]] + 1
            # print("after update cur_vector ", cur_vector)
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
        for word in words:
            w = word.strip('.')
            if not w:
                continue #print("not word") or empty

            #print("Replaced ", word, " with ", w)
            #todo- capital U small u same?
            my_set.add(w)
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
        for word in words:
            w = word.strip('.')
            if not w:
                continue #print("not word")


            #print("w = ", w)
            #print("before update cur_vector ", cur_vector)
            cur_vector[dic[w]] = cur_vector[dic[w]] +1
            #print("after update cur_vector ", cur_vector)
        X.append(cur_vector)

    #print(x)
    #print("y= " , Y)
    F.close()
    logistic_regression(X,Y,7,3,0.1)


if sys.argv[1] == "train":
    run_LR("SMSSpamCollection.train")

if sys.argv[1] == "tune":
    run_LR("SMSSpamCollection.devel")

if sys.argv[1] == "test":
    evaluation()


#run_LR("SMSSpamCollection.train")
#evaluation()
