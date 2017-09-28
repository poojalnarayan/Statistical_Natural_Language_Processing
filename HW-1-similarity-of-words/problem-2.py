import sys
import collections

F = open("vectors_top3000.txt", 'r')
homeVector = []
wordsDic = dict()

for line in F:
    #print(line)
    if line.find("home") != -1:
        list1 = line.split(' ')

        for  i in range(1,202):
            homeVector.append(list1[i])
        break
#get the F pointer back to beginning!!
#remove \n from line!!
F.seek(0,0)
#print(homeVector)
#print(len(homeVector))

for line in F:
    if line.find("home") != -1:
        continue

    list1 = line.split(' ')
    wordsDic[list1[0]]= 0
    dotProduct =0

    for i in range(1,201):
        #print(dotProduct , " i " , i, "list1[i] ", list1[i]," homeVector[i-1] ", homeVector[i-1] )
        dotProduct += (float(list1[i]) * float(homeVector[i-1]))


    wordsDic[list1[0]]= dotProduct
    #print(len(wordsDic))
    #print(wordsDic)
    '''
    list1 = line.split(' ')
    for temp in list1:'''

F.close()
#print(homeVector)

if sys.argv[1] == "similar":
    topWords= sorted([(k,v) for (k,v) in wordsDic.items()],key= lambda item: (item[1],item[0]), reverse=True)[:10]
    print(collections.OrderedDict(topWords).keys())

if sys.argv[1] == "dissimilar":
    topWords= sorted([(k,v) for (k,v) in wordsDic.items()],key= lambda item: (item[1],item[0]))[:10]
    print(collections.OrderedDict(topWords).keys())
