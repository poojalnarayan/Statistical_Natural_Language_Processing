#print("hello world")
import sys
import collections
F = open("brown_sample.txt", 'r')
wordsDic = dict()
POSDic = dict()
pairDic = dict()
for line in F:
    #print(line)
    list1 = line.split(' ')
    for temp in list1:
        if not any(c.isalpha() for c in temp):
            continue

        if pairDic.get(temp) != None:
            pairDic[temp] +=1
        else:
            pairDic[temp] = 1

        list2 = temp.split('/')
        if not any(c.isalpha() for c in list2[0]):
            continue

        if wordsDic.get(list2[0]) != None:
            wordsDic[list2[0]] +=1
        else:
            wordsDic[list2[0]] = 1

        if POSDic.get(list2[1]) != None:
            POSDic[list2[1]] += 1
        else:
            POSDic[list2[1]] = 1
        #print(list2[0])
        #print(list2[1])

F.close()

if sys.argv[1] == "words":
    topWords= sorted([(k,v) for (k,v) in wordsDic.items()],key= lambda item: (item[1],item[0]), reverse=True)[:10]
    print(collections.OrderedDict(topWords).keys())

if sys.argv[1] == "POS":
    topPOSTags = sorted([(k,v) for (k,v) in POSDic.items()],key= lambda item: (item[1],item[0]), reverse=True)[:10]
    print(collections.OrderedDict(topPOSTags).keys())

if sys.argv[1] == "word-POS":
    topPairs = sorted([(k,v) for (k,v) in pairDic.items()],key= lambda item: (item[1],item[0]), reverse=True)[:10]
    print(collections.OrderedDict(topPairs).keys())
#print(wordsDic)
#print(POSDic)
#print(pairDic)
