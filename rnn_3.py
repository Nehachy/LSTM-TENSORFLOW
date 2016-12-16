import tensorflow as tf
import numpy as np
import traceback
import csv
from tensorflow.python.ops import rnn,rnn_cell
import sys


#reading a csv file and creating list of np.array
#data_list: contains the covariates for each time step
#obv_list: containts the observation data for each time step
def readCSV(inp_file):
   try:
      f = open(inp_file)
      reader = csv.reader(f,delimiter=',')

      count = 0
      columns = []
      data_list = []
      obvs_list = []
      for lines in reader:
         if count == 0:
            columns = lines
         else:
            item_list = []
            obv_list = []
            for i in range(len(lines)):
               if i > 0:
                  item_list.append(float(lines[i]))
               elif i == 0:
                  obv_list.append(float(lines[i]))
            x = np.array([item_list])
            y = np.array([obv_list])

            data_list.append(x)
            obvs_list.append(y)

         count += 1

      return data_list,obvs_list
   except:
      traceback.print_exc()

#covariated & Observations
data_list,obvs_list = readCSV('sin.csv')
datalen = len(data_list)
#state dimension calculations
N = len(data_list[0][0])

#Extracting Training and Testing data
percentage = 0.8
index = int(datalen*percentage)

data_list_train = data_list[:index]
data_list_test  = data_list[index:]

obvs_list_train = obvs_list[:index]
obvs_list_test  = obvs_list[index:]

#for i in range(datalen-index):
#   data_list_train.append(np.array([[0 for _ii in range(N)]]))

index_diff = datalen - index

for i in range(index_diff,index):
   data_list_test.append(np.array([[0 for _ii in range(N)]]))

#for i in range(datalen-index):
#   obvs_list_train.append(np.array([[0 for _ii in range(1)]]))

for i in range(index_diff,index):
   obvs_list_test.append(np.array([[0 for _ii in range(1)]]))

datalen = index

aa = np.array(data_list_train)
aa = np.transpose(aa,(1,0,2))

bb = np.array(obvs_list_train)
bb = np.transpose(bb,[1,0,2])

cc = np.array(data_list_test)
cc = np.transpose(cc,[1,0,2])

dd = np.array(obvs_list_test)
dd = np.transpose(dd,[1,0,2])

covariates = tf.placeholder(tf.float32,[None,datalen,N],name="covariates")
dependent = tf.placeholder(tf.float32,[None,datalen,1],name="dependent")
covariates_ = [covariates]

rnn_size = 3

weights = tf.Variable(tf.random_normal([datalen,rnn_size,1]))
biases = tf.Variable(tf.random_normal([1]))

def model():

   x = tf.transpose(covariates,[1,0,2])
   x = tf.reshape(covariates,[-1,N])
   x = tf.split(0,datalen,x)

   lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
   #DCell = rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=0.8)
   multi_cell = rnn_cell.MultiRNNCell([lstm_cell]*2,state_is_tuple=True)
   #init_state = multi_cell.zero_state(1,tf.float32)

   #outputs,states = rnn.rnn(multi_cell,x,dtype=tf.float32)
   outputs,_ = rnn.dynamic_rnn(multi_cell,x,dtype=tf.float32)

   tsize = int(outputs.get_shape()[0])

   #last = tf.gather(outputs,int(outputs.get_shape()[0])-1)

   #output = tf.matmul(outputs[-1],weights) + biases
   #output = tf.matmul(last,weights) + biases
   #output  = [tf.matmul(tf.gather(outputs,i),weights)+biases for i in range(tsize)]
   output = tf.batch_matmul(outputs,weights)+biases
   output = tf.transpose(output,[1,0,2])

   return output

prediction = model()
cost = tf.reduce_mean((dependent-prediction)**2,1)
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
   sess.run(tf.initialize_all_variables())
  
   for iter_ in range(5000):
      feed_dict = {covariates:aa,dependent:bb}
      _,cost_,pred_ = sess.run([optimizer,cost,prediction],feed_dict=feed_dict)
      print "*********************************"
      print iter_
      print cost_
      for i in range(datalen):
         print pred_[0][i],obvs_list_train[i]
      print "*********************************\n"

   feed_dict = {covariates:cc} 
   pred1_ = sess.run([prediction],feed_dict=feed_dict)

   for i in range(len(obvs_list_test)):
      print pred1_[0][0][i],obvs_list_test[i]
