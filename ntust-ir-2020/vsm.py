import numpy as np
import collections 


# Readfile

text_file = open('./doc_list.txt', "r")
docs = text_file.read().splitlines()

text_file = open('./query_list.txt', "r")
queries = text_file.read().splitlines()

doc_list=[]
for doc in docs:
    f=open('./docs/'+ doc + '.txt')
    content = f.read().split()[5:]
    content = [x for x in content if x != '-1']
    doc_list.append(content)

qry_list=[]
for qry in queries:
    f=open('./queries/'+ qry + '.txt')
    content = f.read().split()    
    content = [x for x in content if x != '-1']
    qry_list.append(content)


# Lexicon

def creat_lexicon(doc_list):
    flattened = [val for sublist in doc_list for val in sublist]
    all_words=list(set(flattened))
    lexicon=dict(zip(all_words,list(range(len(all_words)))))
    return lexicon


# Term Frequency

def get_tf(lexicon, file_list, weight="Raw Frequency", sigma = 0.5):
    
    tf=np.zeros((len(lexicon),len(file_list)))
    for j in range(len(file_list)): 
        content=file_list[j]
        count=dict(collections.Counter(content)) 
        for word in count:
            if word in lexicon:
                i=lexicon[word]
                tf[i][j]=count[word]
    
    if weight=="Raw Frequency":    
        pass
    
    elif weight=="Log Normalization":
        tf = 1+np.log2(tf)
    
    elif weight=="Double Normalization":
        tf_max=np.amax(tf,axis=1).reshape(-1,1)
        tf = sigma+(1-sigma)*(tf/tf_max)
       
    return tf
    


# Inverse Document Frequency

def get_idf(lexicon, file_list, weight ='Inverse Frequency'):
    
    df=np.zeros(len(lexicon))
    for j in range(len(file_list)): 
        appear=np.zeros(len(lexicon))
        content=file_list[j]
        count=dict(collections.Counter(content)) 
        for word in count:
            if word in lexicon:
                i=lexicon[word]
                appear[i]=1
        df=np.add(df,appear)
    
    if weight =='Inverse Frequency':
        idf = np.log(len(file_list)/df)
    
    elif weight =='Inverse Frequency Smooth':
        idf = np.log(1 + len(file_list)/df)
    
    elif weight =='Inverse Frequency Max':
        idf = np.log(1 + max(df)/df)
    
    elif weight =='Probabilistic Inverse Frequency':
        idf = np.log((len(file_list)-df)/df)
    idf = idf.reshape(-1, 1)
    return idf


# Term Weight

def get_term_weight(lexicon, doc_list, qry_list, scheme = 1):
    
    if scheme==1:
        
        doc_tf=get_tf(lexicon, doc_list)
        qry_tf=get_tf(lexicon, qry_list, weight="Double Normalization")
        idf=get_idf(lexicon, doc_list)
        
        doc_weight=np.multiply(doc_tf,idf)
        qry_weight=np.multiply(qry_tf,idf)
    
    elif scheme==2:
        
        doc_tf=get_tf(lexicon, doc_list)
        qry_tf=get_tf(lexicon, qry_list)
        idf=get_idf(lexicon, doc_list, weight="Inverse Frequency Smooth")
        
        doc_weight=np.add(doc_tf,1)
        qry_weight=np.multiply(qry_tf,idf)
        
    elif scheme==3:
        
        doc_tf=get_tf(lexicon, doc_list)
        qry_tf=get_tf(lexicon, qry_list)
        idf=get_idf(lexicon, doc_list)
        
        doc_weight=np.multiply(doc_tf,idf)
        qry_weight=np.multiply(qry_tf,idf)
        
    qry_weight=np.transpose(qry_weight)
    doc_weight=np.transpose(doc_weight)
    return qry_weight,doc_weight


# Cosine Similarity

def cos_sim(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))


# Ranking & Output result

    # * scheme=3, score=0.55128
    # * scheme=2, score=0.13736
    # * scheme=1, score=0.01975
    
lexicon=creat_lexicon(doc_list)
qtw,dtw=get_term_weight(lexicon, doc_list, qry_list, scheme = 3)

fname = "./result.txt"
f = open(fname, 'w')
f.write("Query,RetrievedDocuments\n")  

for q in range(len(qry_list)):
    f.write(queries[q] + ",")   
    
    for j in range(len(doc_list)):         
        if j==0:
            sim=cos_sim(qtw[q],dtw[j])
        else:
            sim=np.append(sim,cos_sim(qtw[q],dtw[j]))
        
    rank = np.argsort(-sim)
    for j in rank:
        f.write(docs[j]+" ")
    f.write("\n")
f.close()