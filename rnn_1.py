import tensorflow as tf
import numpy as np
import traceback
import csv
from tensorflow.python.ops import rnn,rnn_cell

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
               if i > 1:
                  item_list.append(float(lines[i]))
               elif i == 1:
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
data_list,obvs_list = readCSV('input.csv')
datalen = len(data_list)

#state dimension calculations
N = len(data_list[0][0])

covariates = tf.placeholder(tf.float32,[1,N],name="covariates")
dependent = tf.placeholder(tf.float32,[None,1],name="dependent")
covariates_ = [covariates]

rnn_size = 100

def model():
   weights = tf.Variable(tf.random_normal([rnn_size,1]))
   biases = tf.Variable(tf.random_normal([1]))

   lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
   outputs,states = rnn.rnn(lstm_cell,covariates_,dtype=tf.float32)
 
   output = tf.matmul(outputs[-1],weights) + biases

   return output

prediction = model()
#cost = tf.nn.softmax_cross_entropy_with_logits(prediction,dependent)
cost = tf.reduce_mean((dependent-prediction)**2)
#optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
   sess.run(tf.initialize_all_variables())
  
   for iter_ in range(10000):
     for i in range(datalen):
        feed_dict = {covariates:data_list[i],dependent:obvs_list[i]}
        _,cost_,pred_ = sess.run([optimizer,cost,prediction],feed_dict=feed_dict)
        print pred_,obvs_list[i],cost_
