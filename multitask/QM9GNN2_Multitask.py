#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import mean, square

from spektral.datasets import qm9
from spektral.layers import EdgeConditionedConv, GlobalSumPool, GlobalAttentionPool
from spektral.utils import label_to_one_hot

from sklearn.preprocessing import StandardScaler

print("finished imports")

# In[2]:


learning_rate = 1e-3
epochs = 10
batch_size = 32


# In[3]:


A_all, X_all, E_all, y_all = qm9.load_data(return_type='numpy',
                           nf_keys='atomic_num',
                           ef_keys='type',
                           self_loops=True,
                           amount=None) # chnage this to None to load entire dataset
print("finished reading SDF")

# Preprocessing
X_uniq = np.unique(X_all)
X_uniq = X_uniq[X_uniq != 0]
E_uniq = np.unique(E_all)
E_uniq = E_uniq[E_uniq != 0]

X_all = label_to_one_hot(X_all, X_uniq)
E_all = label_to_one_hot(E_all, E_uniq)


# In[4]:


# Parameters
N = X_all.shape[-2]       # Number of nodes in the graphs
F = X_all[0].shape[-1]    # Dimension of node features
S = E_all[0].shape[-1]    # Dimension of edge features
n_out = y_all.shape[-1]   # Dimension of the target


# In[5]:


# because we don't want to train only on the lightest molecules
# we randomly sample from the dataset
indices = np.random.choice(X_all.shape[0], 10000, replace=False)
X = X_all[indices, :, :]
A = A_all[indices, :, :]
E = E_all[indices, :, :, :]
y = y_all.iloc[indices, :].copy()


# In[6]:


# storing the means and stddevs here allows us 
# to normalize our data 
# TODO: shouldn't we store only the mean/stddev for the training data?
task_to_scaler = dict()
for task in list(y.columns)[1:]:
    scaler = StandardScaler()
    y.loc[:, task] = scaler.fit_transform(y[[task]])
    task_to_scaler[task] = scaler


# In[7]:


clusters = [['A', 'B', 'alpha'], 
            ['C', 'r2', 'u0'],
            ['zpve', 'g298', 'cv'],
            ['lumo', 'u298', 'h298'],
            ['mu', 'homo']]


# In[8]:


A_train, A_test,     X_train, X_test,     E_train, E_test,     y_train, y_test = train_test_split(A, X, E, y, test_size=0.1)


# In[9]:


def build_single_task_model(*, N, F, S):
  X_in = Input(shape=(N, F))
  A_in = Input(shape=(N, N))
  E_in = Input(shape=(N, N, S))

  gc1 = EdgeConditionedConv(64, activation='relu')([X_in, A_in, E_in])
  gc2 = EdgeConditionedConv(128, activation='relu')([gc1, A_in, E_in])
  pool = GlobalAttentionPool(256)(gc2)
  dense = Dense(256, activation='relu')(pool)
  output = Dense(1)(dense)

  # Build model
  model = Model(inputs=[X_in, A_in, E_in], outputs=output)
  optimizer = Adam(lr=learning_rate)
  model.compile(optimizer=optimizer, loss='mse')

  return model


# In[10]:


def build_hard_sharing_model(*, N, F, S, num_tasks):
  X_in = Input(shape=(N, F))
  A_in = Input(shape=(N, N))
  E_in = Input(shape=(N, N, S))

  gc1 = EdgeConditionedConv(64, activation='relu')([X_in, A_in, E_in])
  gc2 = EdgeConditionedConv(128, activation='relu')([gc1, A_in, E_in])
  pool = GlobalAttentionPool(256)(gc2)
  dense_list = [Dense(256, activation='relu')(pool) for i in range(num_tasks)]
  output_list = [Dense(1)(dense_layer) for dense_layer in dense_list]

  # Build model
  model = Model(inputs=[X_in, A_in, E_in], outputs=output_list)
  optimizer = Adam(lr=learning_rate)
  model.compile(optimizer=optimizer, loss='mse')

  return model


# In[11]:


def build_soft_sharing_model(*, N, F, S, num_tasks, share_param):
  X_in = Input(shape=(N, F))
  A_in = Input(shape=(N, N))
  E_in = Input(shape=(N, N, S))

  gc1_list = [EdgeConditionedConv(64, activation='relu')([X_in, A_in, E_in]) for i in range(num_tasks)]
  gc2_list = [EdgeConditionedConv(128, activation='relu')([gc1, A_in, E_in]) for gc1 in gc1_list]
  pool_list = [GlobalAttentionPool(256)(gc2) for gc2 in gc2_list]
  dense_list = [Dense(256, activation='relu')(pool) for pool in pool_list]
  output_list = [Dense(1)(dense) for dense in dense_list]

  def loss(y_actual, y_pred):
    avg_layer_diff = 0
    for i in range(num_tasks):
      for j in range(i):
        for gc in [gc1_list, gc2_list]:
          avg_layer_diff += mean(square(gc[i].trainable_weights - gc[j].trainable_weights))
    avg_layer_diff /= (num_tasks)*(num_tasks-1)/2  
    return mean(square(y_actual - y_pred)) + share_param*avg_layer_diff

  # Build model
  model = Model(inputs=[X_in, A_in, E_in], outputs=output_list)
  optimizer = Adam(lr=learning_rate)
  model.compile(optimizer=optimizer, loss='mse')

  return model


# In[12]:


# FOLDER_PATH = '/content/drive/My Drive/Colab Notebooks/demo_models'
FOLDER_PATH = 'demo_models'

def generate_model_filename(tasks):
  filename = "".join(sorted(tasks))
  return path.join(FOLDER_PATH, filename + '.h5')
  # return filename + '.h5'

def generate_task_scaler_filename(task):
  return path.join(FOLDER_PATH, task + '.txt')
  # return task + '.txt'


# In[13]:


def save_model(model, tasks):
  model.save_weights(generate_model_filename(tasks))
  for task in tasks:
    scaler_filename = generate_task_scaler_filename(task)
    with open(scaler_filename, 'w') as f:
      print(task_to_scaler[task].mean_[0], file=f)
      print(task_to_scaler[task].scale_[0], file=f)

def load_hard_sharing_model(*, N, F, S, tasks):
  model = build_hard_sharing_model(N=N, F=F, S=S, num_tasks=len(tasks))
  model.load_weights(generate_model_filename(tasks))
  task_to_scaler = dict()
  for task in tasks:
    with open(generate_task_scaler_filename(task), 'r') as f:
      lines = f.readlines()
      scaler = StandardScaler()
      scaler.mean_ = float(lines[0].strip())
      scaler.scale_ = float(lines[1].strip())
      task_to_scaler[task] = scaler
  return model, task_to_scaler


# In[21]:


def predict_property(prop, mol_id, clusters, N=N, F=F, S=S):
  for cluster in clusters:
    if prop in cluster:
      model, task_to_scaler = load_hard_sharing_model(N=N, F=F, S=S, tasks=cluster)
      i = mol_id - 1

      # convert shape for batch mode
      def wrap(a):
        return a.reshape([1] + list(a.shape))
      x = list(map(wrap, [X_all[i], A_all[i], E_all[i]]))

      cluster_prediction = model.predict(x)
      prediction = cluster_prediction[cluster.index(prop)]
      prediction = task_to_scaler[prop].inverse_transform(prediction)
      # print(prediction)
      return prediction[0][0]


# In[19]:


if __name__ == '__main__': 
  print('begin training models')
  for cluster in clusters:
    print(f'training {cluster}')
    model = build_hard_sharing_model(N=N, F=F, S=S, num_tasks=len(cluster))
    model.fit(x=[X_train, A_train, E_train], 
              y=y_train[cluster].values,
              batch_size=batch_size,
              validation_split=0.1,
              epochs=25)
    save_model(model, cluster)


# In[20]:


if __name__ == '__main__':
    for cluster in clusters:
      model, _ = load_hard_sharing_model(N=N, F=F, S=S, tasks=clusters[0])
      model_loss = model.evaluate(x=[X_test, A_test, E_test],
                                    y=y_test[cluster].values)
      print(f"Test loss: {model_loss}")


# In[22]:


if __name__ == '__main__':
    print(predict_property('A', 1, clusters, N=N, F=F, S=S))


# In[ ]:




