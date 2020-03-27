#The code is important since the code is used to load the best model to predict. All results from paper  is from the code.
from __future__ import print_function
import numpy as np
import tensorflow as tf
tf.set_random_seed(10)
np.random.seed(10)
from sklearn.utils import shuffle
from sklearn import metrics
import datetime
import time
import sys
import os
from sklearn.metrics import auc as auc_s
from sklearn.metrics import precision_recall_curve
from scipy import sparse
import pickle
from tools import cp_ids_to_data_ids
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

c="PC3"
file2 = "./eight_cell_line/ppi_expression.npy"
file="./eight_cell_line/"+c+"test_cp_id_cp_information.pickle"


def norm_laplacian(adj):
    
    D_inv = np.divide(1., np.sqrt(D), out=np.zeros_like(D), where=D!=0.)

    
    D_inv_diag=np.zeros_like(adj)
    np.fill_diagonal(D_inv_diag,D_inv)

    adj = D_inv_diag.dot(adj).dot(D_inv_diag)
    return adj       
partition = np.load(file2)#adj
A=partition
D= np.array(A).sum(axis=1)
D1=np.zeros_like(A)
np.fill_diagonal(D1,D)
L=D1-A
L=np.array(norm_laplacian(L),dtype=np.float32)
eigenvalue,featurevector=np.linalg.eig(L)
U=featurevector
U_T=U.T


id_=[]

## hyper-parameters and settings
L2_weight_decay = 0.00001

drop = 0
learning_rate = 0.001
training_epochs = 70
batch_size =16
display_step = 1
n_hidden_1 = np.shape(partition)[0]
n_hidden_2 =2048
n_hidden_3 =4
n_classes = 2
n_features = 978
n_embbding=100

loss_rec = np.zeros([training_epochs, 1])
training_eval = np.zeros([training_epochs, 2])


def multilayer_perceptron(x, weights, biases, droprate):
    W=tf.diag(weights['h1'])
    Z=tf.matmul(tf.matmul(U,W),U_T)
    layer_1=tf.matmul(x,Z,transpose_b=True)
    layer_1 = tf.nn.relu(layer_1) 
    layer_1 = tf.nn.dropout(layer_1, rate=droprate
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
   
    layer_2 = tf.nn.dropout(layer_2,  rate=droprate)

    

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

x1 = tf.placeholder(tf.float32, [None, n_features])
x2=tf.placeholder(tf.float32,[None,n_features])
other_placeholder=tf.placeholder(tf.float32,[None,4])
y = tf.placeholder(tf.int32, [None, n_classes])
drop_placeholder= tf.placeholder(tf.float32)

weights = {
    'h1': tf.Variable(tf.truncated_normal(shape=[n_features], stddev=0.1)),#500*500
    'h2': tf.Variable(tf.truncated_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.1)),#500*64
  
    'out': tf.Variable(tf.truncated_normal(shape=[n_hidden_2, n_embbding], stddev=0.1))#16*2

}


biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
   
    'out': tf.Variable(tf.zeros([n_embbding]))
}

weights_out = {
    'h1': tf.Variable(tf.truncated_normal(shape=[5,n_hidden_3], stddev=0.1)),#500*64
    'h2':tf.Variable(tf.truncated_normal(shape=[n_hidden_3,n_classes],stddev=0.1))
  
}

biases_out = {
    'b1': tf.Variable(tf.zeros([n_hidden_3])),
   'b2':tf.Variable(tf.zeros([n_classes])),
    
}



pred1_ = multilayer_perceptron(x1, weights, biases,drop_placeholder )
pred2_=multilayer_perceptron(x2,weights,biases,drop_placeholder)
pred1_mean=tf.reshape(tf.reduce_mean(pred1_,axis=1),(-1,1))
pred2_mean=tf.reshape(tf.reduce_mean(pred2_,axis=1),(-1,1))
pred1=tf.subtract(pred1_,pred1_mean)
pred2=tf.subtract(pred2_,pred2_mean)
pred1_norm = tf.sqrt(tf.reduce_sum(tf.square(pred1), axis=1))
pred2_norm = tf.sqrt(tf.reduce_sum(tf.square(pred2), axis=1))
pred1_pred2=tf.reduce_sum(tf.multiply(pred1, pred2),axis=1)
r2=tf.square(pred1_pred2/(pred1_norm*pred2_norm))
r2=tf.expand_dims(cosin,axis=1)

out_feature=tf.concat([r2,other_placeholder],axis=1)

layer_1=tf.add(tf.matmul(out_feature,weights_out['h1']),biases_out['b1'])
layer_1=tf.nn.relu(layer_1)
pred=tf.add(tf.matmul(layer_1,weights_out['h2']),biases_out['b2'])





## Evaluation
y_score = tf.nn.softmax(pred)
y_p=tf.argmax(y_score, 1)
y_true=tf.argmax(y, 1)
correct_prediction = tf.equal(y_p, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



all_cpid_cpinformation=pickle.load(open(file,"rb")) #load_external_test_dataset
#load knockdown_gene_expression
dict_gene_name_gene_expression={}
KD_knockdwon_time_infomation=[]
with open("./eight_knockdown/"+c+"knockdown_time.txt") as f:
    for line in f:
        hang=line.rstrip("\n").split("\t")
        gene_name=hang[0]
        gene_time=float(gene_name.split("_")[1])
        gene_expression=hang[1:]
        gene_expression=np.array([float(i) for i in gene_expression])
        dict_gene_name_gene_expression[gene_name]=gene_expression
        KD_knockdwon_time_infomation.append([gene_time])
KD_knockdwon_time_infomation=np.array( KD_knockdwon_time_infomation)
        
        
saver_dir="drop_0.3_learning_rate_0.001_hidden_2048_2019-10-15-09-35-51/" 
d1 = datetime.datetime.now()
date=d1.strftime('%Y-%m-%d-%H')+"/"
result_path="./test_3_26/"+c+"_result_"+date+"/"
if not os.path.exists(result_path):
    os.makedirs(result_path)    
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,'./evaluate_2019_10_15/'+saver_dir+"169.ckpt")      
    for id_ in all_cpid_cpinformation.keys():
        cp_information=all_cpid_cpinformation[id_]
        cp_perturbation=cp_information[0:978]
        other=cp_information[978:]
        all_prediction_gene_name=list(dict_gene_name_gene_expression.keys())
        num_prediction_gene_name=len(all_prediction_gene_name)
        all_prediction_gene_expression=[]
        for gene_name in all_prediction_gene_name:
            gene_expression=dict_gene_name_gene_expression[gene_name]
            all_prediction_gene_expression.append(gene_expression)
        all_prediction_gene_expression=np.vstack(all_prediction_gene_expression)
        cp_information=np.vstack([cp_perturbation]*num_prediction_gene_name)
        other_information=np.vstack([other]*num_prediction_gene_name)
        other_information=np.concatenate([other_information,KD_knockdwon_time_infomation],axis=1)
        y_s= sess.run( y_score, feed_dict={x1: cp_information,x2:all_prediction_gene_expression,  drop_placeholder:0,other_placeholder:other_information})
        y_s=y_s[:,1] 
        y_s= y_s.tolist()
        gene_name_y_s=dict(zip(all_prediction_gene_name,y_s))
        
        with open(result_path+str(id_)+".txt","w") as cp_score:
            for g in gene_name_y_s.keys():
                cp_score.write(str(g)+"\t"+str(gene_name_y_s[g])+"\n")
            
            
            
