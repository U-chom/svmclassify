import sklearn
# import pandas as pd
import numpy as np
import subprocess
import sys
import glob
import elasticsearch
from elasticsearch import Elasticsearch, helpers
import time
#大規模なデータではdifferent_wordsとBoWはelasticsearchに載せるべき

es = Elasticsearch()

mapping = {
    "mappings": {
           "my_type": {
            "properties": {
                "name": {
                    "type": "keyword"
                },
                "num": {
                    "type": "integer"
                },
                "svm_score": {
                    "type": "keyword"
                }
            }
        }
    }
}


es.indices.delete('svm')
ex = es.indices.exists(index="svm")
if False == ex:
    es.indices.create(index='svm', body = mapping)
    print("indexを新規作成しました。")

def Data_load():
    with open("./A5.txt") as A:
        docs = A.readlines()
        docs.pop(0)
    return docs
#docには分かち書きした１記事を入れること

def put_data(name,num,sw):
    if sw == 1:
        print(f"{name}をindexに追加します。onehotベクトルは{num}です。")
        body = {"name": name,"num" :num} 
        es.index(index="svm",id=num,body=body)
        time.sleep(1)
        if searcher("name",name,1) == 0:
            print("追加失敗")
        elif searcher("name",name,1) == 1:
            print("追加成功")
        else:
            print(f"{name}は重複しています。")
    elif sw == 2:
        result = es.get(index="svm",id=name)
        name_org = result['_source']['name']
        es.delete(index="svm",id=name)#項目削除
        body = {"name": name_org,"num": name,"svm_score": num}
        es.index(index="svm",id=name,body=body)#項目再生成
        time.sleep(1)
        if searcher("name",name_org,4) == num:
            print("追加成功")
        else:
            print("追加失敗") 

def searcher(keyname,key,step):
    if step == 1:
        result = es.search(index='svm',body={"query": {"term": {keyname: key}}})
        hit_count = result['hits']['total']
        if hit_count > 1:
            print("警告:登録項目に重複が見られます。")
        return hit_count
    elif step == 2:#引数は"name"とその値
        result = es.search(index='svm',body={"query": {"term": {keyname: key}}})
        hits = result['hits']
        # print(hits)
        out = hits['hits'][0]['_source']['num']
        return out
    elif step == 3:#引数は"name"と"その値
        result = es.search(index='svm',body={"query": {"term": {keyname: key}}})
        try:
            Ans = result['hits']['hits'][0]['_source']['svm_score']
            return Ans
        except:
            return 0
    elif step == 4:#引数は"num"とその値
        result = es.search(index='svm',body={"query": {"term": {keyname: key}}})
        try:
            # print(result)
            Ans = result['hits']['hits'][0]['_source']['svm_score']
            return Ans
        except:
            print("エラー:svmscoreが登録されていません。")
    elif step == 5:#引数は"svm_score"とその値
        result = es.search(index='svm',body={'query': {'term': {'svm_score': key}}})
        Ans = result['hits']['hits']
        Ans_name = []
        for m in Ans:
            Ans_name.append(m['_source']['name'])
        return Ans_name

def different_words_checker(num,doc,sw):#辞書生成 dic={"word":{"No":0,"Freq":1},...},この中でonehotベクトル化する。
    #異なり後の抽出
    if sw == 1:#頻度なし
        for x in doc:
            # print(f"ターゲット => {x}")
            Ans = searcher("name",x,1)#Ans = x in different_words.keys()
            if Ans == 0:
                put_data(x,num,1)# different_words[x] = num
                num = num + 1
    elif sw == 2:#頻度あり
        freqdic = {}#頻度辞書
        for x in doc:# 記事の頻度を調べる
            Ans = x in freqdic.keys()
            if not True == Ans:
                freqdic[x] = 1
                num = num + 1
            elif True == Ans:
                freq = freqdic[x]
                freqdic[x] = freq + 1
        freqword = []#頻度付き単語
        for x in freqdic:#単語に頻度を反映させる
            freq = freqdic[x]
            new_word = f"{freq}:{x}"
            freqword.append(new_word)
        for x in freqword:#大辞書を参照する
            Ans = searcher("name",x,1)#Ans = x in different_words.keys()
            if Ans == 0:
                put_data(x,num,1)#different_words[x] = num
                num = num + 1
    BoW = []
    if sw == 1:
        for x in doc:
            number = searcher("name",x,2)#number = different_words[x]
            BoW.append(int(number))
    elif sw == 2:
        for x in freqword:
            number = searcher("name",x,2)#number = different_words[x]
            BoW.append(int(number))
    
    with open("./BoW","a",encoding="utf-8") as bow:
        BoW = list(set(BoW))
        BoW.sort()
        for i,m in enumerate(BoW):
            m2 = f"{m}:1"
            BoW[i] = m2
        BoW_s = " ".join(BoW)
        print(BoW_s,end="\n",file = bow)
    return num

def SVM_perf(sw,j):
    if sw == 0:#全素性学習
        with open("out/learn_out.txt","w",encoding='utf-8') as lo:
            print("-------------------------------全学習--------------------------------")
            # os.system('svm_perf_learn -c 1.0 learn.dat model.dat')#学習
            command = ['svm_perf_learn','-c','20.0','out/learn/learn.dat','out/model/model.dat']#学習
            Learn = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(Learn.stdout.decode("utf8"),file=lo)
            print(Learn.stderr.decode("utf8"),file=lo)
            # res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # sys.stdout.buffer.write(res.stdout)
        
        with open("out/test_out.txt","w",encoding='utf-8') as to:
            print("--------------------------------分類--------------------------------")
            classify = ['svm_perf_classify','out/test/test.dat','out/model/model.dat','out/predictions/predictions']#分類
            Classify = subprocess.run(classify, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(Classify.stdout.decode("utf8"),file=to)
            print(Classify.stderr.decode("utf8"),file=to)
    if sw == 1:#cv学習
        with open("out/learn_out_cv"+str(j)+".txt","w",encoding='utf-8') as lo:
            print("-------------------------------全学習--------------------------------")
            # os.system('svm_perf_learn -c 1.0 learn.dat model.dat')#学習
            command = ['svm_perf_learn','-c','20.0','out/learn/learn_cv'+str(j)+'.dat','out/model/model_cv'+str(j)+'.dat']#学習
            Learn = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(Learn.stdout.decode("utf8"),file=lo)
            print(Learn.stderr.decode("utf8"),file=lo)
            # res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # sys.stdout.buffer.write(res.stdout)
        
        with open("out/test_out_cv"+str(j)+".txt","w",encoding='utf-8') as to:
            print("--------------------------------分類--------------------------------")
            classify = ['svm_perf_classify','out/test/test_cv'+str(j)+'.dat','out/model/model_cv'+str(j)+'.dat','out/predictions/predictions_cv'+str(j)+'']#分類
            Classify = subprocess.run(classify, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(Classify.stdout.decode("utf8"),file=to)
            print(Classify.stderr.decode("utf8"),file=to) 
    

def validation_split(BoW,label):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=2, shuffle=True, random_state=71)
    j = 1
    for train, test in kf.split(BoW,label):
        BoW_x = []
        label_x = []
        BoW_y = []
        label_y = []
        for i in train:
            BoW_x.append(BoW[i])
            label_x.append(label[i])
        for i in test:
            BoW_y.append(BoW[i])
            label_y.append(label[i])

        with open("./out/learn/learn_cv"+str(j)+".dat","w",encoding="utf-8") as cv:
            for m,n in zip(label_x,BoW_x):
                print(f"{m} {n}",end="",file = cv)
        with open("./out/test/test_cv"+str(j)+".dat","w",encoding="utf-8") as cv:
            for m,n in zip(label_y,BoW_y):
                print(f"{m} {n}",end="",file = cv)
        SVM_perf(1,j)
        j = j + 1

def model_decide():
    test_out = glob.glob("./out/test_out_cv*")
    acc_dic = {}
    acc_list = []
    for testfile in test_out:
        with open(testfile) as acc_file:
            acc = acc_file.readlines()[11]
        acc_s = acc.split(": ")
        acc_dic[testfile] = (float(acc_s[1]))
        acc_list.append(float(acc_s[1]))
    acc_list.sort(reverse=True)
    acc = "".join([k for k, v in acc_dic.items() if v == acc_list[0]])
    print(acc)#これがクロスバリデーションで得られる最適解
    return acc

def feature_selection():
    pass

def make_dir():
    try:
        subprocess.run("mkdir","out")
        subprocess.run("mkdir","out/learn")
        subprocess.run("mkdir","out/model")
        subprocess.run("mkdir","out/test")
        subprocess.run("mkdir","out/predictions")
    except:
        pass

def first():
    sw = 1#1：頻度なし、2：頻度あり
    docs = Data_load()
    # different_words = {} =>elasticsearchに変更済
    make_dir()
    num = 1
    with open("./BoW","w",encoding="utf-8") as bow:
        pass
    for doc_org in docs:#全体でone_hotベクトルを作成する。
        doc_org = doc_org.replace("\n","")
        doc = doc_org.split(" ")
        num = different_words_checker(num,doc,sw)#辞書生成
      
def second():
    with open("./BoW") as bow:
        bows = bow.readlines()
    with open("./label.txt") as labels:
        label = labels.read().split("\n")
        label.pop(0)
        label.pop()
    bi_label = []
    bi_bow = []
    for m,n in zip(label,bows):
        if m == "0":
            continue
        bi_label.append(m)
        bi_bow.append(n)
    del label,bows
    validation_split(bi_bow,bi_label)

def therd():
    acc = model_decide()
    cv_num = acc.replace("./out/test_out_cv","")
    cv_num = cv_num.replace(".txt","")
    with open("./out/model/model_cv"+str(cv_num)+".dat") as md:
        models = str(md.readlines()[11]).split(" ")
        models.pop(0)
        models.pop()
    model_dic = {}
    model_list = []
    for m in models:
        mm = m.split(":")
        # print(type(mm[1]))
        if searcher("num",mm[0],3) == mm[1]:
            print("すでにsvm scoreが登録されています。")
        elif searcher("num",mm[0],3) == 0:
            print(f"ID:{mm[0]}にsvm score => {mm[1]}を追加します。")
            put_data(mm[0],mm[1],2)#model_dic[int(mm[0])] = float(mm[1])#これをfirstの辞書にくっつける
        else:
            print("不明なsvm scoreが登録されています。\n再登録します。")
            put_data(mm[0],mm[1],2)
        model_list.append(float(mm[1]))
    print(len(model_list))
    model_list = list(set(model_list))
    model_list.sort(reverse=True)
    with open("./out/svm_score","w",encoding="utf-8") as f:
        i = 0
        for m in model_list:
            Ans = searcher("svm_score",str(m),5)
            for n in Ans:
                print(f"{n} : {m}",end="\n",file=f)
                i = i + 1
        print(i)
first()
second()
therd()


# #全件数検索
# result = es.search(index='svm',body={"query": {"match_all": {}}})
# hits = result['hits']
# print('indexヒット数 : %s' % hits['total'])
# print(hits)