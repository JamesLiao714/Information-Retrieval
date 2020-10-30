import numpy as np
import collections 
from sklearn.metrics.pairwise import cosine_similarity
from math import log2
# create qry_list & doc_list

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


# labelize word
def creat_lexicon(doc_list):
    flattened = [val for sublist in doc_list for val in sublist]
    all_words=list(set(flattened))
    lexicon=dict(zip(all_words,list(range(len(all_words)))))
    return lexicon


# TF implementation

def get_tf(lexicon, file_list, weight = "N2"):
    
    tf=np.zeros((len(lexicon), len(file_list)))

    for j in range(len(file_list)): 
        content = file_list[j]
        count=dict(collections.Counter(content)) 

        for word in count:
            if word in lexicon:
                i = lexicon[word]
                if weight == "N1": 
                    tf[i][j] = 1 + log2(count[word])
                else:
                    tf[i][j] = count[word] #term[i] in doc[j]'
    return tf
    
# IDF implementatio

def get_idf(lexicon, file_list, weight ='PIF'):
    
    df = np.zeros(len(lexicon))
    for j in range(len(file_list)): 
        appear = np.zeros(len(lexicon))
        content = file_list[j]
        count = dict(collections.Counter(content)) 
        for word in count:
            if word in lexicon:
                i = lexicon[word]
                appear[i] = 1
        df = np.add(df, appear)
    
    if weight =='IF':   #70.740
        idf = np.log(len(file_list)/df)
    
    elif weight == 'IFS':
        idf = np.log(1 + len(file_list)/df)
    
    elif weight == 'IFM':
        idf = np.log(1 + max(df)/df)
    
    elif weight == 'PIF': #70.750
        idf = np.log((len(file_list)-df)/df)
    idf = idf.reshape(-1, 1)
    return idf


# Term Weight implementation

def get_term_weight(lexicon, doc_list, qry_list):
   
    doc_tf = get_tf(lexicon, doc_list)
    qry_tf = get_tf(lexicon, qry_list)
 
    idf = get_idf(lexicon, doc_list)
    
    doc_weight=np.multiply(doc_tf,idf)
    qry_weight=np.multiply(qry_tf,idf)
   
    qry_weight=np.transpose(qry_weight)
    doc_weight=np.transpose(doc_weight)
    
    return qry_weight,doc_weight

def main():
    lexicon=creat_lexicon(doc_list)
    qtw,dtw=get_term_weight(lexicon, doc_list, qry_list)

    fname = "./result.txt"
    f = open(fname, 'w')
    f.write("Query,RetrievedDocuments\n")  

    sim = cosine_similarity(qtw, dtw)
    sim = np.array(sim)
    for q in range(len(qry_list)):
        f.write(queries[q] + ",")  
        t = sim[q]
        rank = np.argsort(-t)
        print(rank)
        for j in rank:
            f.write(docs[j] + " ")
        f.write("\n")
    f.close()

if __name__ == "__main__":
    main()
    