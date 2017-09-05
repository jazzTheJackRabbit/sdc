
# coding: utf-8

# In[1]:


#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')


# # LeNet Lab
# ![LeNet Architecture](lenet.png)
# Source: Yan LeCun

# ## Load Data
# 
# Load the MNIST data, which comes pre-loaded with TensorFlow.
# 
# You do not need to modify this section.

# In[2]:


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


# The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.
# 
# However, the LeNet architecture only accepts 32x32xC images, where C is the number of color channels.
# 
# In order to reformat the MNIST data into a shape that LeNet will accept, we pad the data with two rows of zeros on the top and bottom, and two columns of zeros on the left and right (28+2+2 = 32).
# 
# You do not need to modify this section.

# In[3]:


import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))


# ## Visualize Data
# 
# View a sample from the dataset.
# 
# You do not need to modify this section.

# In[4]:


import random
import numpy as np
# import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

# plt.figure(figsize=(1,1))
# plt.imshow(image, cmap="gray")
print(y_train[index])


# ## Preprocess Data
# 
# Shuffle the training data.
# 
# You do not need to modify this section.

# In[5]:


from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)


# ## Setup TensorFlow
# The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.
# 
# You do not need to modify this section.

# In[7]:


import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128


# ## TODO: Implement LeNet-5
# Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.
# 
# This is the only cell you need to edit.
# ### Input
# The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.
# 
# ### Architecture
# **Layer 1: Convolutional.** The output shape should be 28x28x6.
# 
# **Activation.** Your choice of activation function.
# 
# **Pooling.** The output shape should be 14x14x6.
# 
# **Layer 2: Convolutional.** The output shape should be 10x10x16.
# 
# **Activation.** Your choice of activation function.
# 
# **Pooling.** The output shape should be 5x5x16.
# 
# **Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.
# 
# **Layer 3: Fully Connected.** This should have 120 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 4: Fully Connected.** This should have 84 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 5: Fully Connected (Logits).** This should have 10 outputs.
# 
# ### Output
# Return the result of the 2nd fully connected layer.

# In[8]:


import tensorflow as tf


# In[12]:


# get_ipython().magic(u'pinfo tf.nn.max_pool')


# # In[13]:


# get_ipython().magic(u'pinfo flatten')


# # In[17]:


# get_ipython().magic(u'pinfo tf.nn.conv2d')


# In[20]:


from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    layer1_filter = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    layer1_bias = tf.Variable(tf.zeros(6))
    layer1_strides = [1,1,1,1]
    layer1_padding = 'VALID'
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    import pdb; pdb.set_trace()  # breakpoint aa8d0e5f //
    layer1 = tf.nn.conv2d(x, layer1_filter, layer1_strides, layer1_padding) + layer1_bias
    
    # TODO: Activation.
    layer1 = tf.nn.relu(layer1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    layer1_filter = [1,1,1,6]
    layer1_strides = [1,2,2,1]
    layer1 = tf.nn.max_pool(layer1, layer1_filter, layer1_strides, layer1_padding)

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    layer2_filter = [5,5,1,6]
    layer2_strides = [1,1,1,1]
    layer2_padding = 'VALID'
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    layer2 = tf.nn.conv2d(x, layer2_filter, layer2_strides, layer2_padding)
    
    # TODO: Activation.
    layer2 = tf.nn.relu(layer2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    layer2_filter = [2,2,1,6]
    layer2_strides = [1,2,2,1]
    layer2 = tf.nn.max_pool(layer2, layer2_filter, layer2_strides, layer2_padding)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    layer2 = flatten(layer2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    layer3_weights = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    layer3_bias = tf.Variable(tf.zeros(120))
    layer3 = tf.add(tf.matmul(layer2, layer3_weights), layer3_bias)
    
    # TODO: Activation.
    layer3 = tf.nn.relu(layer3)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    layer4_weights = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    layer4_bias = tf.Variable(tf.zeros(84))
    layer4 = tf.add(tf.matmul(layer3, layer4_weights), layer4_bias)
    
    # TODO: Activation.
    layer4 = tf.nn.relu(layer4)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    layer5_weights = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
    layer5_bias = tf.Variable(tf.zeros(10))
    logits = tf.add(tf.matmul(layer4, layer5_weights), layer5_bias)
    
    return logits


# ## Features and Labels
# Train LeNet to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data.
# 
# `x` is a placeholder for a batch of input images.
# `y` is a placeholder for a batch of output labels.
# 
# You do not need to modify this section.

# In[15]:


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)


# ## Training Pipeline
# Create a training pipeline that uses the model to classify MNIST data.
# 
# You do not need to modify this section.

# In[19]:


rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# ## Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.
# 
# You do not need to modify this section.

# In[ ]:


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ## Train the Model
# Run the training data through the training pipeline to train the model.
# 
# Before each epoch, shuffle the training set.
# 
# After each epoch, measure the loss and accuracy of the validation set.
# 
# Save the model after training.
# 
# You do not need to modify this section.

# In[ ]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


# ## Evaluate the Model
# Once you are completely satisfied with your model, evaluate the performance of the model on the test set.
# 
# Be sure to only do this once!
# 
# If you were to measure the performance of your trained model on the test set, then improve your model, and then measure the performance of your model on the test set again, that would invalidate your test results. You wouldn't get a true measure of how well your model would perform against real data.
# 
# You do not need to modify this section.

# In[ ]:


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

