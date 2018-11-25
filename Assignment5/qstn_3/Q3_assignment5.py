#!/usr/bin/env python
# coding: utf-8

# In[29]:


import tfrbm


# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from tfrbm import BBRBM, GBRBM
import tensorflow as tf
from tensorflow import keras
import numpy as np
import mnist
import random
random.seed(20)
images = mnist.train_images()
labels = mnist.train_labels()
images = np.reshape(images,(60000,784))

test_images = mnist.test_images()
test_labels = mnist.test_labels()
test_images = np.reshape(test_images,(10000,784))


# In[31]:


#generate 10k random integers between 0,60000
d_indices = random.sample(range(0,60000),10000)


# In[32]:


tr_images = images[d_indices,:]


# In[33]:


tr_labels = labels[d_indices]


# In[34]:


random.seed(40)
val_indices = random.sample(range(0,60000),1000)
val_img = images[val_indices,:]
val_lab = labels[val_indices]


# In[35]:


# from pca_for_large_dimension import pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.decomposition import PCA
pca = PCA(25)
pca.fit(tr_images)
red_data = pca.transform(tr_images)
val_img = pca.transform(val_img)
test_images = pca.transform(test_images)


# In[36]:


val_img.shape


# In[42]:


from tensorflow.keras.layers import Dense, Dropout, Flatten
input_shape=(25,)
num_classes = 10
model = keras.Sequential()
model.add(Dense(256, activation="sigmoid",input_shape=input_shape))
# model.add(Dropout(0.2))
model.add(Dense(256, activation="sigmoid"))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001),
              metrics=['accuracy'])


# In[43]:


model.summary()


# In[50]:


training=model.fit(red_data,tr_labels,batch_size=40,
          epochs=30,
          validation_data=(val_img,val_lab))


# In[48]:


plt.plot(training.history['loss'],label="training loss")
plt.plot(training.history['val_loss'],label="validation loss")
plt.legend()
plt.show()


# In[51]:


loss,acc = model.evaluate(test_images,test_labels)
print("accuracy is",acc)


# ### now using LDA

# In[52]:


red_data = pca.fit(tr_images).transform(tr_images)
lda = LinearDiscriminantAnalysis(n_components=10)
lda = lda.fit(red_data,tr_labels)
red_data2 = lda.transform(red_data)
lda_val_data = lda.transform(val_img)
lda_test_data = lda.transform(test_images)


# In[53]:


print(red_data2.shape)


# In[56]:


input_shape=(9,)
num_classes = 10
model_on_lda = keras.Sequential()
model_on_lda.add(Dense(256, activation="sigmoid",input_shape=input_shape))
model_on_lda.add(Dropout(0.2))
model_on_lda.add(Dense(256, activation="sigmoid",input_shape=input_shape))
model_on_lda.add(Dense(num_classes, activation='softmax'))

model_on_lda.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])


# In[57]:


lda_train=model_on_lda.fit(red_data2,tr_labels,batch_size=20,
          epochs=20,validation_data=(lda_val_data,val_lab))


# In[58]:


plt.plot(lda_train.history['loss'],label="training loss")
plt.plot(lda_train.history['val_loss'],label="validation loss")
plt.legend()
plt.show()


# In[59]:


loss,acc = model_on_lda.evaluate(lda_test_data,test_labels)
print(acc)


# ## 3_part_c...using gaussian RBM
# 

# In[60]:


from tfrbm import BBRBM, GBRBM
print(red_data.shape)


# In[61]:


import numpy as np
import matplotlib.pyplot as plt
from tfrbm import GBRBM

gbrbm = GBRBM(n_visible=25, n_hidden=9, learning_rate=0.01, momentum=0.95, use_tqdm=True)
errs = gbrbm.fit(red_data, n_epoches=300, batch_size=10)
plt.plot(errs)
plt.show()


# In[24]:


# errs = gbrbm.fit(red_data, n_epoches=300, batch_size=20)
# plt.plot(errs)
# plt.show()


# In[62]:


rbm_hidden = gbrbm.transform(red_data)
val_hidden = gbrbm.transform(val_img)
test_hidden = gbrbm.transform(test_images)


# In[63]:


input_shape=(9,)
num_classes = 10
model_rbm = keras.Sequential()
model_rbm.add(Dense(256, activation="sigmoid",input_shape=input_shape))
model_rbm.add(Dropout(0.2))
model_rbm.add(Dense(256, activation="sigmoid",input_shape=input_shape))
model_rbm.add(Dense(num_classes, activation='softmax'))

model_rbm.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
rbm_train=model_rbm.fit(rbm_hidden,tr_labels,batch_size=20,
          epochs=150,validation_data=(val_hidden,val_lab))


# In[64]:


plt.plot(rbm_train.history['loss'],label="training loss")
plt.plot(rbm_train.history['val_loss'],label="validation loss")
plt.legend()
plt.show()


# In[65]:


loss,acc = model_rbm.evaluate(test_hidden,test_labels)
print(acc)


# ### auto encoder 3_part_d

# In[66]:


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# this is the size of our encoded representations
encoding_dim = 9

# input_img = Input(shape=(25,))
h1 = Dense(128, activation='relu',input_shape=(25,))
h2 = Dense(encoding_dim, activation='relu',input_shape=(128,))

h3 = Dense(128,activation="relu",input_shape=(9,))
h4 = Dense(25, activation='sigmoid',input_shape=(128,))
autoencoder = keras.Sequential([h1,h2,h3,h4])
autoencoder.summary()


# In[67]:


##encoder model
encoder = keras.Sequential([h1,h2])
print(encoder.summary())


# In[68]:


#decoder model
decoder = keras.Sequential([h3,h4])
print(decoder.summary())


# In[71]:


autoencoder.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
            )


# In[72]:


ae_train = autoencoder.fit(red_data, red_data,
                epochs=50,
                batch_size=128,
                shuffle=True)


# In[73]:


plt.plot(ae_train.history['loss'])


# In[74]:


decoder.summary()


# In[75]:


encoded_data = encoder.predict(red_data)
val_encoded = encoder.predict(val_img)
test_encoded = encoder.predict(test_images)


# In[76]:


### now training the dnn
input_shape=(9,)
num_classes = 10
model_encoder = keras.Sequential()
model_encoder.add(Dense(256, activation="sigmoid",input_shape=input_shape))
model_encoder.add(Dropout(0.2))
model_encoder.add(Dense(256, activation="sigmoid",input_shape=input_shape))
model_encoder.add(Dense(num_classes, activation='softmax'))

model_encoder.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
encoder_hidden_train=model_encoder.fit(encoded_data,tr_labels,batch_size=20,
          epochs=75,validation_data=(val_encoded,val_lab))


# In[77]:


plt.plot(encoder_hidden_train.history['loss'],label ="train loss")
plt.plot(encoder_hidden_train.history['val_loss'],label =" val loss")
plt.legend()
plt.show()


# In[78]:


loss,acc = model_encoder.evaluate(test_encoded,test_labels)
print(acc)


# In[ ]:




