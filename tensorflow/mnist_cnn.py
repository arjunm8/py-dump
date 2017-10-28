# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:30:38 2017

@author: 1337
"""

import numpy as np
import tensorflow as tf
import time,threading, cv2

tf.logging.set_verbosity(tf.logging.INFO)
#tf.logging.set_verbosity(tf.logging.ERROR)

TRAIN_MODE = False
EVAL_MODE = True
BENCHMARK = False

#DEFINE THE CNN MNIST CLASSIFIER
def cnn_model_fn(features,labels,mode):

    # reshape features input map to [batch_size, image_width, image_height, channels]
    #-1(dynamically computed,tunable),28x28,1(1 coz monochrome,3 for rgb etc)
    input_layer = tf.reshape(features["x"],[-1,28,28,1])

    #CONVOLUTIONAL LAYER,apply 32 5x5 //now 3x3 filters to input layer,
    #padding same = keep input/output dimensions same, relu activation
    #output shape = [batch_size, 28, 28, 32] (now has 32 channels)
    conv1 = tf.layers.conv2d(
                inputs = input_layer,
                filters = 32,
                kernel_size = [3,3],
                padding = "same",
                activation = tf.nn.relu
            )
    conv2 = tf.layers.conv2d(
                inputs = conv1,
                filters = 32,
                kernel_size = [3,3],
                padding = "same",
                activation = tf.nn.relu
            )
    #POOLING LAYER, uses max pooling(max value of each submatrix),2x2filter with stride of 2
    #output shape = [batch_size, 14, 14, 32] read doc on stride
    pool1 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

    dropout1 = tf.layers.dropout(
                inputs = pool1,
                rate = 0.25,
                training = mode==tf.estimator.ModeKeys.TRAIN
            )
    conv3 = tf.layers.conv2d(
                inputs = dropout1,
                filters = 64,
                kernel_size = [3,3],
                padding = "same",
                activation = tf.nn.relu
            )
    conv4 = tf.layers.conv2d(
                inputs = conv3,
                filters = 64,
                kernel_size = [3,3],
                padding = "same",
                activation = tf.nn.relu
            )

    pool2 = tf.layers.max_pooling2d(inputs=conv4,pool_size=[2,2],strides=2)

    dropout2 = tf.layers.dropout(
                inputs = pool2,
                rate = 0.25,
                training = mode==tf.estimator.ModeKeys.TRAIN
            )
    
    pool_flat = tf.reshape(dropout2,[-1,7*7*64])
    
    '''
    #connect second conv layer
    #output shape = [batch_size, 14, 14, 64] (now has 64 channels)
    conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
    #connect second pooling layer
    #output shape = [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    '''

    #FLATTEN FINAL FEATURE MAP(pool2) to 2D [batch_size,features] for building dense layer
    #-1 to calc batch size dynamically based on input data example count
    #each example has 7(pool2width)*7(pool2height)*64(pool2channels) features
    #output = [batch_size,3136]
    
    #pool_flat = tf.reshape(pool1,[-1,14*14*32])

    #DENSE LAYER1 for performing classification on extracted features
    #(with 1024neurons(arbitrarily decided, may need adjustment) and relu activation)
    dense = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)

    #improve results with dropout regularisation, refer docs
    #rhe rate argument specifies the dropout rate; here, 40% of the elements will be randomly dropped out during training.
    #training argument makes sure to only dropout  elements during training(when our model func cnn_model_fn is in TRAIN mode)
    dropout = tf.layers.dropout(
                inputs = dense,
                rate = 0.5,
                training = mode==tf.estimator.ModeKeys.TRAIN
            )

    #LOGITS LAYER / final dense layer with 10 neurons (1 for each class)
    #output shape = [batch_size,10]
    logits = tf.layers.dense(inputs = dropout, units = 10)

    #convert raw logit output to two different formats ie. predicted class and probability
    predictions = {
                #predicted class = element correspoding to row with highest value
                #sort all classes in order
                #used for PREDICT and EVAL
                "classes":tf.argmax(input=logits,axis=1),
                #derive probability from logits layer using softmax activation
                #naming the operation for later reference during logging
                #add 'softmax_tensor' to the graph, (used for PREDICT and logging hook)
                "probabilities": tf.nn.softmax(logits,name="softmax_tensor")
            }
    #if program is in prediction mode then return the compiled predictions dict as an EstimatorSpec object
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)



    #DEFINE LOSS FUNCTION (for both TRAIN and EVAL mode)
    #the labels(arg) tensor contains a list of predictions for our examples
    #for calculating cross entropy, convert the labels to the corresponding one-hot encoding*
    #one-hot encoding formats categorical features to truthtable type representation
    #(https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science)
    #indices = 'on' locations, depth = numpber of target classes
    onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)
    #compute cross entropy of onehotlabels and softmax predictions from our logits layer
    #performs softmax activation on logits layer and then calculates cross emtropy with onehotlabels
    #returns loss as a scalar tensor
    loss = tf.losses.softmax_cross_entropy(
                onehot_labels = onehot_labels,
                logits=logits
            )


    #CONFIGURE TRAINING OPERATION(important af)(for TRAIN mode)
    #configure model to optimize calculated loss value during training
    #learning rate= 0.001 and stochastic gradient descent
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step()
                )
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)


    #add evaluation metrics for EVAL mode
    eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                            labels=labels,
                            predictions=predictions['classes']
                        )
            }
    return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops = eval_metric_ops
            )




#TRAINING AND EVALUATING THE CNN MNIST CLASSIFIER
def main(unused_argv):
    #load mnist training and test data from le internet
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images #returns np array
    #asarray converts any input for(list,tuples etc) to array
    train_labels = np.asarray(mnist.train.labels,dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels,dtype=np.int32)

    #create an Estimator
    #a TensorFlow class for performing high-level model training, evaluation, and inference
    #model_dir argument specifies the directory where model data (checkpoints) will be saved
    mnist_classifier = tf.estimator.Estimator(
                model_fn = cnn_model_fn,
                model_dir = "/tmp/mnist_convnet_model"
            )
    
    if TRAIN_MODE:
        #setup a logging hook
        #store dict of the tensors to log
        #Each key is a label of our choice that will be printed in the log output, and the corresponding label is the name of a Tensor in the TensorFlow graph
        #refer to the softmax_tensor label in the cnn_model_fn()
        tensors_to_log = {"probabilities":"softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
                    tensors = tensors_to_log,
                    #record after every 50 training steps
                    every_n_iter = 50
                )
    
        #training the model
        # format input using numpy_input_fn(
        #x = training feature data(x as dict), and y = label data
        #batchsize=model will train on mini batches of 100 at each step
        #'num_epochs = None' means that the model will train till the specific step count is reached
        #shuffle training data
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={'x':train_data},
                    y = train_labels,
                    batch_size = 100,
                    num_epochs=None,
                    shuffle=True
                )
        
        #model will train 20000 steps total, attach logging hook
        mnist_classifier.train(
                   input_fn = train_input_fn,
                   steps = 1000,
                   hooks = [logging_hook]
            )
    

    if EVAL_MODE:
        #evaluate the model
        #shuffle off to iterate sequentially, evaluate model over 1 epoch and return result
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x":eval_data},
                    y=eval_labels,
                    num_epochs=1,
                    shuffle=False
                )
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
    
    if BENCHMARK:
        #predict single image IMPORTANT: use index [0:1] instead of [0] because shape of [0] in [10x100] array is read as [100,] instead of [1,100] (in [0:1])
        #has label 7
        a = eval_data[0:1]
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x":a},
                num_epochs=1,
                shuffle=False
            )
        start_time = time.clock()
        count_per_sec = 0
        while((time.clock()-start_time)<=1):
            print(list(mnist_classifier.predict(input_fn=predict_input_fn))[0]["classes"])
            count_per_sec+=1
            
        print("fps: ",count_per_sec)
    

    #read_image(mnist_classifier)      
    
    
    '''
    start_time = time.clock()
    count_per_sec = 0
    seq = 0
    while((time.clock()-start_time)<=10):
        time.sleep(0.2)
        seq+=1
        t= threading.Thread(target=predict,args=(mnist_classifier,predict_input_fn,seq))
        t.start()
        count_per_sec+=1
        #print(count_per_sec)
    '''

def trds():
    return threading.enumerate()

def predict(mnist_classifier,predict_input_fn,seq):
    p = list(mnist_classifier.predict(input_fn=predict_input_fn))[0]["classes"]
    print("#",seq,". ",p)

def k():
    cap = cv2.VideoCapture(0)
    cap.release()
    cv2.destroyAllWindows()

def read_image(mnist_classifier):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        #1 defines channel count
        mono = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY,1)
        ret2, thresh = cv2.threshold(mono,100,255,cv2.THRESH_BINARY)
        x = cv2.resize(thresh,(28,28))
        
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":np.array(x, dtype = np.float32)},
            num_epochs=1,
            shuffle=False
        )
        
        p = list(mnist_classifier.predict(input_fn=predict_input_fn))[0]
        print(p["classes"]," : ",p["probabilities"][np.argmax(p["probabilities"])])
        
        cv2.imshow('feed',x)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



'''
#runtime constant, true=programming running under shell instead of imported as a module(if imported, __name__=module name)
#https://stackoverflow.com/questions/419163/what-does-if-name-main-do
if __name__ == "__main__":
    #parses the argument flags and then runs the script's `main` function while passing the flags.
    tf.app.run()

#placement at the top triggers- AttributeError: module '__main__' has no attribute 'main'

'''
#prevent force exit error by using this instead(temporary)
main(1)

    









'''

The methods in the layers module for creating convolutional and pooling layers for two-dimensional image data expect input tensors to have a shape of [batch_size, image_width, image_height, channels], defined as follows:

    batch_size. Size of the subset of examples to use when performing gradient descent during training.
    image_width. Width of the example images.
    image_height. Height of the example images.
    channels. Number of color channels in the example images. For color images, the number of channels is 3 (red, green, blue). For monochrome images, there is just 1 channel (black).

Note that we've indicated -1 for batch size, which specifies that this dimension should be dynamically computed based on the number of input values in features["x"], holding the size of all other dimensions constant. This allows us to treat batch_size as a hyperparameter that we can tune. For example, if we feed examples into our model in batches of 5, features["x"] will contain 3,920 values (one value for each pixel in each image), and input_layer will have a shape of [5, 28, 28, 1]. Similarly, if we feed examples in batches of 100, features["x"] will contain 78,400 values, and input_layer will have a shape of [100, 28, 28, 1].


#convolution1
The filters argument specifies the number of filters to apply (here, 32), and kernel_size specifies the dimensions of the filters as [width, height] (here, [5, 5]).

If filter width and height have the same value, you can instead specify a single integer for kernel_size—e.g., kernel_size=5.

The padding argument specifies one of two enumerated values (case-insensitive): valid (default value) or same. To specify that the output tensor should have the same width and height values as the input tensor, we set padding=same here, which instructs TensorFlow to add 0 values to the edges of the output tensor to preserve width and height of 28. (Without padding, a 5x5 convolution over a 28x28 tensor will produce a 24x24 tensor, as there are 24x24 locations to extract a 5x5 tile from a 28x28 grid.)

The activation argument specifies the activation function to apply to the output of the convolution. Here, we specify ReLU activation with tf.nn.relu.

Our output tensor produced by conv2d() has a shape of [batch_size, 28, 28, 32]: the same width and height dimensions as the input, but now with 32 channels holding the output from each of the filters.


#Pooling1
The pool_size argument specifies the size of the max pooling filter as [width, height] (here, [2, 2]). If both dimensions have the same value, you can instead specify a single integer (e.g., pool_size=2).

The strides argument specifies the size of the stride. Here, we set a stride of 2, which indicates that the subregions extracted by the filter should be separated by 2 pixels in both the width and height dimensions (for a 2x2 filter, this means that none of the regions extracted will overlap). If you want to set different stride values for width and height, you can instead specify a tuple or list (e.g., stride=[3, 6]).

Our output tensor produced by max_pooling2d() (pool1) has a shape of [batch_size, 14, 14, 32]: the 2x2 filter reduces width and height by 50% each.


#Dense Layer

Next, we want to add a dense layer (with 1,024 neurons and ReLU activation) to our CNN to perform classification on the features extracted by the convolution/pooling layers. Before we connect the layer, however, we'll flatten our feature map (pool2) to shape [batch_size, features], so that our tensor has only two dimensions:

pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

In the reshape() operation above, the -1 signifies that the batch_size dimension will be dynamically calculated based on the number of examples in our input data. Each example has 7 (pool2 width) * 7 (pool2 height) * 64 (pool2 channels) features, so we want the features dimension to have a value of 7 * 7 * 64 (3136 in total). The output tensor, pool2_flat, has shape [batch_size, 3136].

Now, we can use the dense() method in layers to connect our dense layer as follows:

dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

The inputs argument specifies the input tensor: our flattened feature map, pool2_flat. The units argument specifies the number of neurons in the dense layer (1,024). The activation argument takes the activation function; again, we'll use tf.nn.relu to add ReLU activation.

To help improve the results of our model, we also apply dropout regularization to our dense layer, using the dropout method in layers:

dropout = tf.layers.dropout(
    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

Again, inputs specifies the input tensor, which is the output tensor from our dense layer (dense).

The rate argument specifies the dropout rate; here, we use 0.4, which means 40% of the elements will be randomly dropped out during training.

The training argument takes a boolean specifying whether or not the model is currently being run in training mode; dropout will only be performed if training is True. Here, we check if the mode passed to our model function cnn_model_fn is TRAIN mode.

Our output tensor dropout has shape [batch_size, 1024].



#One hot encoding transforms categorical features to a format that works better with classification and regression algorithms.
Let’s take the following example. I have seven sample inputs of categorical data belonging to four categories. Now, I could encode these to nominal values as I have done here, but that wouldn’t make sense from a machine learning perspective. We can’t say that the category of “Penguin” is greater or smaller than “Human”. Then they would be ordinal values, not nominal.
What we do instead is generate one boolean column for each category. Only one of these columns could take on the value 1 for each sample. Hence, the term one hot encoding.
This works very well with most machine learning algorithms. Some algorithms, like random forests, handle categorical values natively. Then, one hot encoding is not necessary. The process of one hot encoding may seem tedious, but fortunately, most modern machine learning libraries can take care of it.
Our labels tensor contains a list of predictions for our examples, e.g. [1, 9, ...]. In order to calculate cross-entropy, first we need to convert labels to the corresponding one-hot encoding:

[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 ...]

We use the tf.one_hot function to perform this conversion. tf.one_hot() has two required arguments:

    indices. The locations in the one-hot tensor that will have "on values"—i.e., the locations of 1 values in the tensor shown above.
    depth. The depth of the one-hot tensor—i.e., the number of target classes. Here, the depth is 10.

'''
