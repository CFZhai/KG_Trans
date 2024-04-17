
from tqdm import tqdm
import numpy as np 
import pandas as pd

tqdm.pandas()

import os 
os.chdir("/Users/chaofanzhai/UMN/Project_memory/data/find_dynamic_relationship")
df = pd.read_csv('tmp_word_net_dataset.csv') 
 # 考研单词书， sep 21 - nov 1
 # first time study -word 
 # # uid :52779, # spell: 2023

df['first_response'] = df['first_response'].map(lambda x: {1:1,2:0,3:0,4:1}[x]) # 1:认识,2:模糊,3:忘记,4:熟知
df_gp = df.groupby(['pts', 'user_id'])[['spelling', 'first_response']].agg(lambda x: list(x))
df_gp = df_gp.reset_index(drop=False)
df_gp.columns = ['pts', 'user_id', 'word_list_bydate', 'first_response_list']
df_gp.to_csv('tmp_word_net_dataset-gp.tsv',sep='\t',index=None)


## filter 
temp = df_gp.groupby(["user_id"]).count()["pts"]
temp = temp.reset_index(drop=False)
temp.groupby(['pts']).count()
temp.describe()



wordset = tuple(set(df.spelling.values))
print("num of words", len(wordset))

def get_word_vector(x):
    word_list_bydate=x[0]
    first_response_list=x[1]
    dic = dict(zip(word_list_bydate,first_response_list))
    vector = []
    for word in wordset:
        if word in dic.keys():
            vector.append(dic[word])
        else:
            vector.append(np.nan)
    return vector

print('generate word vector')

#df_gp['vector']=  df_gp[['word_list_bydate','first_response_list']].progress_apply(get_word_vector, axis= 1)
df_gp['vector']=  df_gp[['word_list_bydate','first_response_list']].apply(get_word_vector, axis= 1)


# using a dataframe to store the bag of words 
print("generate word matrix")
X=[]
for i in tqdm(df_gp['vector'].values) :
    X.append(i)


matrix = np.asarray(X)
matrix_pd = pd.DataFrame(matrix) # each row is user-date level observation. each column is a word. 

m = matrix_pd.head(10000)
# check the shared google doc sktech 
def get_pair_frequency(matrix_pd: pd.DataFrame):
    Z = matrix_pd==1
    Z = Z.astype(int) # 记住=1，没记、没学=0
    L = matrix_pd.notna().astype(int) # 有学=1, 没学=0   
    ## map reduce ,   
    
    return L.T.dot(Z) , L.T.dot(L)   # L.T.dot(Z):  row:有没有学 col:有没有记住


#remember_pair_frequency_matrix, study_pair_frequency_matrix = get_pair_frequency(matrix_pd)
remember_pair_frequency_matrix, study_pair_frequency_matrix = get_pair_frequency(m)

## -- 出现-》记住

def find_frequent_item_pair(supp_threshold,cond_threshold,lift_threshold, wordset) : 
    
    # update the significance , 
    
    length = matrix_pd.shape[0]
    pair_frequency_matrix = study_pair_frequency_matrix/length  #同时出现的次数/所有记录的长度
    
    cond_matrix = remember_pair_frequency_matrix/study_pair_frequency_matrix  #同时记住的次数/同时出现的次数
    
   
    print(cond_matrix)
    diag = np.diag(cond_matrix) # diag: X 记住/X 出现
    length = diag.shape[0]
    lift_matrix = cond_matrix/diag.reshape(1,length) #（ 同时记住的次数/同时出现的次数）/（X记住的次数/x出现的次数）
    print(lift_matrix)
    
    x = pair_frequency_matrix>supp_threshold
    y = cond_matrix>cond_threshold
    z = lift_matrix>lift_threshold
    S = x*y*z # S satisfy the both conditions 
    #M= np.tril(S,-1) # get the Lower triangle of an array. 
    loc = np.where(S==True)
    
    pairs ={}
    for i, pair_x in enumerate(tqdm(loc[0])):
        pair_y = loc[1][i]
        pairs[(wordset[pair_x],wordset[pair_y])]= (pair_frequency_matrix.loc[pair_x,pair_y], cond_matrix.loc[pair_x,pair_y], lift_matrix.loc[pair_x,pair_y])
            
    return pairs
    

def find_frequent_item_pair_2(matrix_pd, supp_thres1,cond_thres1,lift_thres1,supp_thres2,cond_thres2,lift_thres2, wordset) : 
    
    # update the significance , 

    supp_matrix1 = remember_pair_frequency_matrix/study_pair_frequency_matrix  #同时记住的次数/同时出现的次数
    supp_matrix2= remember_pair_frequency_matrix/ matrix_pd.shape[0] #同时记住的次数/所有记录的长度
    
    diag = np.diag(remember_pair_frequency_matrix)
    length = diag.shape[0]
    diag = diag.reshape(1,length)   # row #=1, column # = length
    diag_T = diag.reshape(length,1)
    
    diag_study = np.diag(study_pair_frequency_matrix).reshape(1,length)
    diag_study_T = diag_study.reshape(length,1)
    
    cond_matrix1 = remember_pair_frequency_matrix/ diag  #同时记住的次数/ X记住的次数
    cond_matrix2 = remember_pair_frequency_matrix/ diag_study  #同时记住的次数/ X出现的次数
    
    lift_matrix1 = (remember_pair_frequency_matrix/diag_study)/ (diag_T/diag_study_T)  #（同时记住的次数/X出现的次数 ）/（Y记住的次数/Y出现的次数）
    lift_matrix2 = (remember_pair_frequency_matrix/study_pair_frequency_matrix)/ (diag_T/diag_study_T)  #（同时记住的次数/同时出现的次数 ）/（Y记住的次数/Y出现的次数）
   # lift_matrix3 = (remember_pair_frequency_matrix/study_pair_frequency_matrix)/ (diag_T/diag_study_T)  #（Y记住的次数/同时出现的次数 ）/（Y记住的次数/Y出现的次数）
              
    
    x1 = supp_matrix1>supp_thres1
    y1 = cond_matrix1>cond_thres1
    z1 = lift_matrix1>lift_thres1
    
    x2 = supp_matrix2>supp_thres2
    y2 = cond_matrix2>cond_thres2
    z2= lift_matrix2>lift_thres2   
    
    S = x1*y1*z1*x2*y2*z2 # S satisfy the both conditions 
    #M= np.tril(S,-1) # get the Lower triangle of an array. 
    loc = np.where(S==True)
    
    pairs ={}
    for i, pair_x in enumerate(tqdm(loc[0])):
        pair_y = loc[1][i]
        pairs[(wordset[pair_x],wordset[pair_y])]= (supp_matrix1.loc[pair_x,pair_y],supp_matrix2.loc[pair_x,pair_y], cond_matrix1.loc[pair_x,pair_y], cond_matrix2.loc[pair_x,pair_y], lift_matrix1.loc[pair_x,pair_y],lift_matrix2.loc[pair_x,pair_y])
            
    return pairs
        




supp_thres1,cond_thres1,lift_thres1,supp_thres2,cond_thres2,lift_thres2=0,0,0,0,0,0

frequent_pairs =find_frequent_item_pair_2(matrix_pd, supp_thres1,cond_thres1,lift_thres1,supp_thres2,cond_thres2,lift_thres2, wordset) 


d = {}
for i, key in enumerate(tqdm(frequent_pairs)):
    d[i] = {
        'a': key[0], 
        'b': key[1], 
        'supp1': round(frequent_pairs[key][0],6), 
        'cond1': round(frequent_pairs[key][1],6), 
        'lift1': round(frequent_pairs[key][2],6),
        'supp2': round(frequent_pairs[key][3],6), 
        'cond2': round(frequent_pairs[key][4],6), 
        'lift2': round(frequent_pairs[key][5],6),        
    }
df = pd.DataFrame.from_dict(d, "index")
df.sort_values(by=["b", "lift2"], inplace=True)
df.to_csv("./word_pair_2.csv", index=None)



