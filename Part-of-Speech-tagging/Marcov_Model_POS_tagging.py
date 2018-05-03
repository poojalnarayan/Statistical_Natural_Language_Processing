from collections import defaultdict
import ast
import sys

def train_save():
    with open("train.tagged", 'r') as f: # test
        sents = []
        s = []
        for line in f:
            if line == "\n":
                s.append(("END", "END"))
                sents.append(s)
                s = []
            else:
                if s.__len__() == 0:
                    s.append(("START", "START"))
                s.append(line.strip('\n').split('\t'))
                #print(s)

    #accounting last sentence
    s.append(("END", "END"))
    sents.append(s)
    #print(sents)

    tag_tag_counts = {}
    tag_tag_counts = defaultdict(lambda: 0, tag_tag_counts)
    word_tag_counts = {}
    word_tag_counts = defaultdict(lambda: 0, word_tag_counts)
    tag_counts = {}
    tag_counts = defaultdict(lambda: 0, tag_counts)

    for sent in sents:
        #print(sent)
        word_tag_counts[("START", "START")] += 1
        tag_counts[sent[0][1] ] +=1

        for i in range(1, len(sent)):
            #print(sent[i][0], " tag= ", sent[i][1])
            #tag_tag_counts.setdefault((sent[i-1][1],sent[i][1]),0)
            tag_tag_counts[(sent[i-1][1],sent[i][1])] += 1

            #word_tag_counts.setdefault((sent[i][0],sent[i][1]),0)
            word_tag_counts[(sent[i][0],sent[i][1])] += 1

            tag_counts[sent[i][1]] += 1

    #print(tag_tag_counts)
    #print(word_tag_counts)
    #print(tag_counts)

    f = open('save_model.txt', 'w')
    f.write(str(tag_tag_counts)+'\n')
    f.write(str(word_tag_counts)+'\n')
    f.write(str(tag_counts) )

    f.close()

def evaluation():
    F = open("save_model.txt", 'r')
    dics = F.read().split('\n')
    tag_tag_counts = ast.literal_eval(dics[0].split('>, ')[1].strip(')'))
    word_tag_counts = ast.literal_eval(dics[1].split('>, ')[1].strip(')'))
    tag_counts = ast.literal_eval(dics[2].split('>, ')[1].strip(')'))

    #print(tag_tag_counts)
    #print(word_tag_counts)
    #print(tag_counts)
    F.close()

    with open("test.tagged", 'r') as f: #test.tagged
        sents = []
        s = []
        for line in f:
            if line == "\n":
                s.append(("END", "END"))
                sents.append(s)
                s = []
            else:
                if s.__len__() == 0:
                    s.append(("START", "START"))
                s.append(line.strip('\n').split('\t'))
        f.close()
        # accounting last sentence
        s.append(("END", "END"))
        sents.append(s)

    correct_pred = 0
    total_words =0
    for sent in sents:
        #print(sent)
        total_words += len(sent) - 1  # minus 1 to remove first word START frm accounting
        predicted_tags = []
        predicted_tags.append("START")
        for i in range(1, len(sent)):
            max_prob = 0
            best_tag = "NIL"
            for tag in tag_counts:
                #print("predicted_tags[i-1],tag = ", predicted_tags[i-1]," ",tag)
                #print("tag_tag_counts.get((predicted_tags[i-1],tag), 0) = ", tag_tag_counts.get((predicted_tags[i-1],tag), 0), "tag_counts[predicted_tags[i-1]] = ", tag_counts[predicted_tags[i-1]])

                # print("sent[i][0], tag =", sent[i][0], "tag = ", tag)
                # print("word_tag_counts.get((sent[i][0], tag), 0) = ", word_tag_counts.get((sent[i][0], tag), 0), "tag_counts[tag] =", tag_counts[tag])
                trans_prob =  tag_tag_counts.get((predicted_tags[i-1],tag), 0) / tag_counts[predicted_tags[i-1]]
                emiss_prob = word_tag_counts.get((sent[i][0], tag), 0) / tag_counts[tag]
                #print("trans_prob = ", trans_prob, " emiss_prob = ", emiss_prob)
                if trans_prob * emiss_prob > max_prob:
                    max_prob = trans_prob * emiss_prob
                    best_tag = tag

            if best_tag == "NIL":
                # if word_tag_counts.get((sent[i][0], tag), 0) == 0:
                #     print("Unknown word ", sent[i][0])

                most_frequent_tag = sorted([(k,v) for (k,v) in tag_counts.items()],key= lambda item: (item[1],item[0]), reverse=True)[:1]
                predicted_tags.append(most_frequent_tag[0][0])
                #print("most_frequent_tag = ", most_frequent_tag[0][0])
                #break
            else:
                predicted_tags.append(best_tag)

            if best_tag == sent[i][1]:
                correct_pred+=1

            # else:
            #     break

    #     print("sent = ", sent, "predicted_tags = ", predicted_tags)
    #     if predicted_tags[predicted_tags.__len__() - 1] == "END":
    #         correct_pred += 1
    #
    # print("Sentence level accuracy ", correct_pred/ len(sents)) #unknown?
    # print("len(sents) = ", len(sents))

    print(correct_pred/total_words)

def evaluation_smoothing():
    F = open("save_model.txt", 'r')
    dics = F.read().split('\n')
    tag_tag_counts = ast.literal_eval(dics[0].split('>, ')[1].strip(')'))
    word_tag_counts = ast.literal_eval(dics[1].split('>, ')[1].strip(')'))
    tag_counts = ast.literal_eval(dics[2].split('>, ')[1].strip(')'))

    #print(tag_tag_counts)
    #print(word_tag_counts)
    #print(tag_counts)
    F.close()

    total_words_in_dictionary = set()
    with open("test.tagged", 'r') as f: #test.tagged
        sents = []
        s = []
        for line in f:
            if line == "\n":
                s.append(("END", "END"))
                sents.append(s)
                s = []
            else:
                if s.__len__() == 0:
                    s.append(("START", "START"))
                s.append(line.strip('\n').split('\t'))
                total_words_in_dictionary.add( line.strip('\n').split('\t')[0])
        f.close()
        # accounting last sentence
        s.append(("END", "END"))
        sents.append(s)

    correct_pred = 0
    total_words =0
    K_1 = len(tag_counts)   #total num of tags
    K_2 = len(total_words_in_dictionary)    #total num of words in the dictionary
    for sent in sents:
        #print(sent)
        total_words += len(sent) - 1  # minus 1 to remove first word START frm accounting
        predicted_tags = []
        predicted_tags.append("START")
        for i in range(1, len(sent)):
            max_prob = 0
            best_tag = "NIL"
            for tag in tag_counts:
                #print("predicted_tags[i-1],tag = ", predicted_tags[i-1]," ",tag)
                #print("tag_tag_counts.get((predicted_tags[i-1],tag), 0) = ", tag_tag_counts.get((predicted_tags[i-1],tag), 0), "tag_counts[predicted_tags[i-1]] = ", tag_counts[predicted_tags[i-1]])

                # print("sent[i][0], tag =", sent[i][0], "tag = ", tag)
                # print("word_tag_counts.get((sent[i][0], tag), 0) = ", word_tag_counts.get((sent[i][0], tag), 0), "tag_counts[tag] =", tag_counts[tag])
                trans_prob =  (tag_tag_counts.get((predicted_tags[i-1],tag), 0)) / (tag_counts[predicted_tags[i-1]] )
                emiss_prob = (word_tag_counts.get((sent[i][0], tag), 0) +1) / (tag_counts[tag] +K_2)
                #print("trans_prob = ", trans_prob, " emiss_prob = ", emiss_prob)
                if trans_prob * emiss_prob > max_prob:
                    max_prob = trans_prob * emiss_prob
                    best_tag = tag

            if best_tag == "NIL":
                # if word_tag_counts.get((sent[i][0], tag), 0) == 0:
                #     print("Unknown word ", sent[i][0])
                most_frequent_tag = sorted([(k,v) for (k,v) in tag_counts.items()],key= lambda item: (item[1],item[0]), reverse=True)[:1]
                predicted_tags.append(most_frequent_tag[0][0])
                #print("most_frequent_tag = ", most_frequent_tag[0][0])
                #break
            else:
                predicted_tags.append(best_tag)

            if best_tag == sent[i][1]:
                correct_pred+=1

    print(correct_pred/total_words)


#train_save()
#evaluation()
#evaluation_smoothing()


if sys.argv[1] == "train":
    train_save()

if sys.argv[1] == "test":
    evaluation()

if sys.argv[1] == "smooth":
    evaluation_smoothing()