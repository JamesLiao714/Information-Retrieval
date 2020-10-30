import numpy as np
import collections 
from sklearn.metrics.pairwise import cosine_similarity
from math import log2
import time
# create qry_list & doc_list
start = time.clock()


raw_file = open('doc_list.txt', "r")
docs = raw_file.read().splitlines()

raw_file = open('query_list.txt', "r")
queries = raw_file.read().splitlines()

doc_list=[]
for doc in docs:
    f = open('docs/'+ doc + '.txt')
    content = f.read().split()
    doc_list.append(content)

qry_list=[]
for qry in queries:
    f=open('queries/'+ qry + '.txt')
    content = f.read().split()    
    qry_list.append(content)


# # Lexicon

def creat_lexicon(doc_list):
    flattened = [val for sublist in doc_list for val in sublist]
    all_words=list(set(flattened))
    lexicon=dict(zip(all_words,list(range(len(all_words)))))
    return lexicon


# Term Frequency

def get_tf(lexicon, file_list):
    
    tf = np.zeros((len(lexicon),len(file_list)))
    for j in range(len(file_list)): 
        content = file_list[j]
        count = dict(collections.Counter(content)) 
        for word in count:
            if word in lexicon:
                i = lexicon[word]
                tf[i][j] = count[word]
    return tf
 
# Term Frequency Weight

def get_Fij(tf, doc_len, k1 = 1.5, b = 0.7):
    avg_len = doc_len.mean()
    Fij = np.zeros(tf.shape)
    print(str(tf.shape[1]) + '  ' + str(tf.shape[0]))
    print(str(len(tf[0])) + '  ' + str(len(tf)))
    for i in range(len(tf)):
        Fij[i] = np.add(Fij[i], (k1 + 1) * tf[i]) / (k1 * ((1 - b) + b * (doc_len / avg_len)) + tf[i])
    #print(Fij)
    return Fij


def get_Fiq(tf, k3 = 1.2):
    Fiq = np.zeros(tf.shape)
    for i in range(len(tf)):
        Fiq[i] = np.add(Fiq[i], ((k3 + 1) * tf[i]) / (k3 + tf[i]))
    return Fiq

# IDF

def get_idf(lexicon, file_list):
    
    df = np.zeros(len(lexicon))
    for j in range(len(file_list)): 
        appear = np.zeros(len(lexicon))
        content = file_list[j]
        count = dict(collections.Counter(content)) 
        for word in count:
            if word in lexicon:
                i = lexicon[word]
                appear[i] = 1
        df = np.add(df,appear)
    
    idf = np.log((len(file_list) - df + 0.5) / (df + 0.5))
    #print(idf)
    return idf


# # Similarity

def BM25_sim(doc_list, qry_list, lexicon, Fij, Fiq, idf):
    sim = np.zeros((len(qry_list),len(doc_list)))
    for q in range(len(qry_list)):
        for j in range(len(doc_list)):
            intersection = list(set(qry_list[q]) & set(doc_list[j]))
            for word in intersection:
                i = lexicon[word]
                sim[q][j] += Fij[i][j] * Fiq[i][q] * idf[i]
    return sim


# # Ranking & Output result
doc_len = np.array([len(i) for i in doc_list])

(k1, k3, b) = (2, 1, 0.85)
lexicon = creat_lexicon(doc_list)
print('tfij')
tfij = get_tf(lexicon, doc_list)
print('tfiq')
tfiq = get_tf(lexicon, qry_list)
print('idf')
idf = get_idf(lexicon, doc_list)
print('fij')
Fij = get_Fij(tfij, doc_len, k1, b)
print('fiq')
Fiq = get_Fiq(tfiq, k3)
print('sim')
sim = BM25_sim(doc_list, qry_list, lexicon, Fij, Fiq, idf)

fname = "./result2.txt"
f = open(fname, 'w')
f.write("Query,RetrievedDocuments\n")  

for q in range(len(qry_list)):
    f.write(queries[q] + ",")        
    rank = np.argsort(-sim[q])
    for j in rank:
        f.write(docs[j]+" ")
    f.write("\n")
f.close()

end = time.clock()
print(end - start)