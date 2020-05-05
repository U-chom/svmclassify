import csv
import sys
from pyknp import Juman
import os
import glob
from sklearn.model_selection import KFold

print('sys.argv[0]      : ', sys.argv[0])
def first():
    with open('./長崎スタジアムシティに関するアンケート（回答）.csv') as csvf:
        read_csv = csv.reader(csvf)
        for row in read_csv:
            for i in range(1,7):
                out1 = row[i]
                out1 = out1.replace("\n","")
                out1 = out1.replace("\\","")
                out1 = out1.replace(" ","")
                jumanpp = Juman()
                if len(out1) == 0:
                    R1 = [ ]
                elif out1.count("。") >= 2:
                    out1x = out1.split("。")
                    l1 = []
                    for j in out1x:
                        result = jumanpp.analysis(out1)
                        R1 =  [mrph.midasi for mrph in result.mrph_list()]
                        l1 = l1 + R1
                    result = l1
                else:
                    result = jumanpp.analysis(out1) 
                    R1 =  [mrph.midasi for mrph in result.mrph_list()]
            #    except:
             #       print(out1)
                R2 = " ".join(R1)
                title = "A" + str(i) 
                with open("./"+str(title)+".txt","a",encoding="utf-8") as o1: 
                    print(R2,file = o1)
def one_hot(words,termFreq):
    wordArray = words.split(" ")
        # 各行の全単語を処理するまで繰り返し
    for word in wordArray:
        # 既出の単語なら1加算、初めてなら1をセット
        if word in termFreq:
            termFreq[word] += 1
        else:
            termFreq[word] = 1
    
def second():
    with open("./label.txt") as label:
        lab = label.readlines()
    with open("./A5.txt") as p01:
        p01_1 = p01.readlines()
    i01 = 0
    i02 = 0
    with open("./freq1.txt","w",encoding="utf-8") as f:
        pass
    with open("./freq2.txt","w",encoding="utf-8") as f:
        pass
    for m,n in zip(lab, p01_1):
        n = n.replace("\n","")
        if "1" in m:
            termFreq = {}
            #one-hotへ
            one_hot(n,termFreq)
            with open("./freq1.txt","a",encoding="utf-8") as f:
                print("@" + str(i01),file = f)
                print("1 z",file = f)
                print("1 i:" + str(i01),file = f)
                print("1 c:1",file = f)
                for term, count in termFreq.items():
                    print("{} {}".format(count,term),end="\n",file = f)
            i01 += 1

        elif "2" in m:
            termFreq = {}
            #one-hotへ
            one_hot(n,termFreq)
            with open("./freq2.txt","a",encoding="utf-8") as f:
                print("@" + str(i02),file = f)
                print("1 z",file = f)
                print("1 i:" + str(i02),file = f)
                print("1 c:2",file = f)
                for term, count in termFreq.items():
                    print("{} {}".format(count,term),end="\n",file = f)
            i02 += 1

def thead():
    
    os.system("svm_perf_learn -c 20 -l 2 --b 0 output/train.dat output/model")
    os.system("svm_perf_classify example1/test.dat example1/model example1/predictions")




#first()
#second()
therd()
