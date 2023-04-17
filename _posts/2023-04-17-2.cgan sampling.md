---
layout: single
title:  "project_1-1"
categories: jupyter
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>



```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensorflow import keras
from tensorflow.keras import layers, activations
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, LeakyReLU, ReLU, Conv1D, MaxPool1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
```


```python
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
```


```python
timedelta = pd.to_timedelta(data['Time'], unit = 's')
data['Time_day'] = (timedelta.dt.components.days).astype(int)
data['Time_min'] = (timedelta.dt.components.minutes).astype(int)
data['Time_hour'] = (timedelta.dt.components.hours).astype(int)
```


```python
data['Amount'] = np.log(data['Amount'] + 0.001)
```


```python
x = data.drop(['Class', 'Time_day','Time'], 1)
y = data['Class'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, 
                                                    shuffle = True, 
                                                    stratify = y, 
                                                    random_state = 220625)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.5, 
                                                shuffle = True, 
                                                stratify = y_test, 
                                                random_state = 220625)
```


```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
```


```python
class cGAN():
    def __init__(self):
        self.latent_dim = 32
        self.out_shape = 31
        self.num_classes = 2
        self.clip_value = 0.01
        optimizer = Adam(0.0002, 0.5)

        # build discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # build generator
        self.generator = self.build_generator()

        # generating new data samples
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        gen_samples = self.generator([noise, label])

        self.discriminator.trainable = False

        # passing gen samples through disc. 
        valid = self.discriminator([gen_samples, label])

        # combining both models
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer,
                             metrics=['accuracy'])
        self.combined.summary()

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        init = RandomNormal(mean=0.0, stddev=0.02)
        model = Sequential()

        model.add(Dense(128, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha = 0.2))
        #model.add(Dropout(0.4))
        model.add(BatchNormalization(momentum=0.85))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha = 0.2))
        #model.add(Dropout(0.5))
        model.add(BatchNormalization(momentum=0.85))

        #model.add(Dense(512))
        #model.add(Dropout(0.2))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(self.out_shape, activation='tanh'))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        
        model_input = multiply([noise, label_embedding])
        gen_sample = model(model_input)

        return Model([noise, label], gen_sample, name="Generator")

    
    def build_discriminator(self):
        init = RandomNormal(mean=0.0, stddev=0.02)
        model = Sequential()

        model.add(Dense(256, input_dim=self.out_shape, kernel_initializer=init))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.5))
        
        model.add(Dense(128, kernel_initializer=init))
        model.add(LeakyReLU(alpha = 0.2))
        
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        
        gen_sample = Input(shape=(self.out_shape,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.out_shape)(label))

        model_input = multiply([gen_sample, label_embedding])
        validity = model(model_input)

        return Model(inputs=[gen_sample, label], outputs=validity, name="Discriminator")


    def train(self, X_train, y_train, pos_index, neg_index, epochs, batch_size=32, sample_interval=100):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            
            #  Train Discriminator with 8 sample from postivite class and rest with negative class
            idx1 = np.random.choice(pos_index, 8)
            idx0 = np.random.choice(neg_index, batch_size-8)
            idx = np.concatenate((idx1, idx0))
            samples, labels = X_train[idx], y_train[idx]
            samples, labels = shuffle(samples, labels)
            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            gen_samples = self.generator.predict([noise, labels])

            # label smoothing
            if epoch < epochs//1.5:
                valid_smooth = (valid+0.1)-(np.random.random(valid.shape)*0.1)
                fake_smooth = (fake-0.1)+(np.random.random(fake.shape)*0.1)
            else:
                valid_smooth = valid 
                fake_smooth = fake
                
            # Train the discriminator
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch([samples, labels], valid_smooth)
            d_loss_fake = self.discriminator.train_on_batch([gen_samples, labels], fake_smooth)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            # Condition on labels
            self.discriminator.trainable = False
            sampled_labels = np.random.randint(0, 2, batch_size).reshape(-1, 1)
            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            if (epoch+1)%sample_interval==0:
                print (f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")
```


```python
cgan = cGAN()
```

<pre>
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 256)               8192      
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 256)               0         
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
=================================================================
Total params: 41,217
Trainable params: 41,217
Non-trainable params: 0
_________________________________________________________________
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 128)               4224      
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 128)               0         
_________________________________________________________________
batch_normalization (BatchNo (None, 128)               512       
_________________________________________________________________
dense_4 (Dense)              (None, 256)               33024     
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 256)               0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 256)               1024      
_________________________________________________________________
dense_5 (Dense)              (None, 31)                7967      
=================================================================
Total params: 46,751
Trainable params: 45,983
Non-trainable params: 768
_________________________________________________________________
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_5 (InputLayer)            [(None, 32)]         0                                            
__________________________________________________________________________________________________
input_6 (InputLayer)            [(None, 1)]          0                                            
__________________________________________________________________________________________________
Generator (Functional)          (None, 31)           46815       input_5[0][0]                    
                                                                 input_6[0][0]                    
__________________________________________________________________________________________________
Discriminator (Functional)      (None, 1)            41279       Generator[0][0]                  
                                                                 input_6[0][0]                    
==================================================================================================
Total params: 88,094
Trainable params: 46,047
Non-trainable params: 42,047
__________________________________________________________________________________________________
</pre>

```python
y_train = y_train.reshape(-1, 1)
pos_index = np.where(y_train==1)[0]
neg_index = np.where(y_train==0)[0]
cgan.train(x_train, y_train, pos_index, neg_index, epochs = 10000)
```

<pre>
99 [D loss: 0.6165916621685028, acc.: 0.0] [G loss: [0.6488956212997437, 0.71875]]
199 [D loss: 0.5802412182092667, acc.: 0.0] [G loss: [0.697252631187439, 0.40625]]
299 [D loss: 0.5519186556339264, acc.: 0.0] [G loss: [0.7048655152320862, 0.46875]]
399 [D loss: 0.3955982103943825, acc.: 0.0] [G loss: [0.7653761506080627, 0.375]]
499 [D loss: -0.30715931951999664, acc.: 0.0] [G loss: [0.8783853054046631, 0.4375]]
599 [D loss: -0.4087570756673813, acc.: 0.0] [G loss: [1.1386815309524536, 0.5]]
699 [D loss: -2.3177929371595383, acc.: 0.0] [G loss: [1.4416937828063965, 0.28125]]
799 [D loss: -8.262075811624527, acc.: 0.0] [G loss: [0.9650179147720337, 0.65625]]
899 [D loss: -8.822682082653046, acc.: 0.0] [G loss: [1.285879373550415, 0.59375]]
999 [D loss: -14.265609741210938, acc.: 0.0] [G loss: [1.6818277835845947, 0.5]]
1099 [D loss: -18.91232681274414, acc.: 0.0] [G loss: [1.7416465282440186, 0.5625]]
1199 [D loss: -23.17023614048958, acc.: 0.0] [G loss: [2.5925631523132324, 0.40625]]
1299 [D loss: -43.82981374114752, acc.: 0.0] [G loss: [3.295228958129883, 0.5]]
1399 [D loss: -66.38108444213867, acc.: 0.0] [G loss: [2.6467080116271973, 0.6875]]
1499 [D loss: -38.590914726257324, acc.: 0.0] [G loss: [3.4710774421691895, 0.5]]
1599 [D loss: -64.5138324201107, acc.: 0.0] [G loss: [6.4130449295043945, 0.375]]
1699 [D loss: -44.96352034807205, acc.: 0.0] [G loss: [4.89211368560791, 0.625]]
1799 [D loss: -90.56502085924149, acc.: 0.0] [G loss: [4.013472557067871, 0.59375]]
1899 [D loss: -73.89964658021927, acc.: 0.0] [G loss: [8.392986297607422, 0.46875]]
1999 [D loss: -107.36103403568268, acc.: 0.0] [G loss: [11.352012634277344, 0.46875]]
2099 [D loss: -122.9909930229187, acc.: 0.0] [G loss: [7.407649993896484, 0.5625]]
2199 [D loss: -83.73549890518188, acc.: 0.0] [G loss: [14.13847541809082, 0.5]]
2299 [D loss: -25.842885971069336, acc.: 0.0] [G loss: [11.494759559631348, 0.40625]]
2399 [D loss: -197.30968928337097, acc.: 0.0] [G loss: [14.596891403198242, 0.625]]
2499 [D loss: -112.83005845546722, acc.: 0.0] [G loss: [15.785823822021484, 0.65625]]
2599 [D loss: -359.96414399147034, acc.: 0.0] [G loss: [16.383272171020508, 0.4375]]
2699 [D loss: -288.6312322318554, acc.: 0.0] [G loss: [21.54410171508789, 0.5]]
2799 [D loss: -224.0280363559723, acc.: 0.0] [G loss: [28.138172149658203, 0.46875]]
2899 [D loss: -245.16787561774254, acc.: 0.0] [G loss: [15.353250503540039, 0.53125]]
2999 [D loss: -204.79028034210205, acc.: 0.0] [G loss: [17.608570098876953, 0.375]]
3099 [D loss: -351.9512948989868, acc.: 0.0] [G loss: [38.95763397216797, 0.46875]]
3199 [D loss: -167.65500688552856, acc.: 0.0] [G loss: [22.54305076599121, 0.46875]]
3299 [D loss: -590.8432583808899, acc.: 0.0] [G loss: [31.82522201538086, 0.46875]]
3399 [D loss: -790.4359655380249, acc.: 0.0] [G loss: [33.04595184326172, 0.46875]]
3499 [D loss: -633.3161625862122, acc.: 0.0] [G loss: [32.02052688598633, 0.3125]]
3599 [D loss: -712.6011079549789, acc.: 0.0] [G loss: [43.890472412109375, 0.375]]
3699 [D loss: -473.06075048446655, acc.: 0.0] [G loss: [27.979110717773438, 0.5625]]
3799 [D loss: -856.8676528930664, acc.: 0.0] [G loss: [34.95841979980469, 0.5]]
3899 [D loss: -417.2732048034668, acc.: 0.0] [G loss: [55.2061653137207, 0.40625]]
3999 [D loss: -834.5978322029114, acc.: 0.0] [G loss: [73.2412109375, 0.40625]]
4099 [D loss: -957.5527496337891, acc.: 0.0] [G loss: [60.46697235107422, 0.59375]]
4199 [D loss: -590.0055198669434, acc.: 0.0] [G loss: [52.91297912597656, 0.5625]]
4299 [D loss: -936.1962516307831, acc.: 0.0] [G loss: [75.72872924804688, 0.53125]]
4399 [D loss: -537.5403256416321, acc.: 0.0] [G loss: [94.28530883789062, 0.40625]]
4499 [D loss: -358.0260325074196, acc.: 0.0] [G loss: [76.54610443115234, 0.34375]]
4599 [D loss: -1678.7496074438095, acc.: 0.0] [G loss: [78.55979919433594, 0.375]]
4699 [D loss: -650.1054997444153, acc.: 0.0] [G loss: [63.82576370239258, 0.4375]]
4799 [D loss: -1016.1265215873718, acc.: 0.0] [G loss: [64.2233657836914, 0.28125]]
4899 [D loss: -1151.6451091766357, acc.: 0.0] [G loss: [40.983245849609375, 0.65625]]
4999 [D loss: -644.2265319824219, acc.: 0.0] [G loss: [102.74908447265625, 0.25]]
5099 [D loss: -642.8559341430664, acc.: 0.0] [G loss: [87.85138702392578, 0.3125]]
5199 [D loss: -393.34708416461945, acc.: 0.0] [G loss: [178.22760009765625, 0.03125]]
5299 [D loss: -1124.7731909751892, acc.: 0.0] [G loss: [86.76603698730469, 0.25]]
5399 [D loss: -2430.9322395324707, acc.: 0.0] [G loss: [137.175537109375, 0.4375]]
5499 [D loss: -870.93101978302, acc.: 0.0] [G loss: [153.56753540039062, 0.34375]]
5599 [D loss: -1400.8017044067383, acc.: 0.0] [G loss: [114.91169738769531, 0.34375]]
5699 [D loss: -1270.2537879943848, acc.: 0.0] [G loss: [117.27494812011719, 0.34375]]
5799 [D loss: -985.9622650146484, acc.: 0.0] [G loss: [163.13294982910156, 0.21875]]
5899 [D loss: -2415.258605003357, acc.: 0.0] [G loss: [171.30931091308594, 0.46875]]
5999 [D loss: -746.4360580444336, acc.: 0.0] [G loss: [162.09945678710938, 0.34375]]
6099 [D loss: -1522.1935653686523, acc.: 0.0] [G loss: [178.3120880126953, 0.25]]
6199 [D loss: -1704.5457592010498, acc.: 0.0] [G loss: [215.93209838867188, 0.4375]]
6299 [D loss: -3166.5922906398773, acc.: 0.0] [G loss: [157.1644744873047, 0.34375]]
6399 [D loss: -3909.9365158081055, acc.: 0.0] [G loss: [167.05618286132812, 0.46875]]
6499 [D loss: -2284.1313252449036, acc.: 0.0] [G loss: [111.5127182006836, 0.5625]]
6599 [D loss: -5330.25532245636, acc.: 0.0] [G loss: [236.13941955566406, 0.59375]]
6699 [D loss: 131.4794979095459, acc.: 64.0625] [G loss: [194.12557983398438, 0.28125]]
6799 [D loss: 222.55449676513672, acc.: 59.375] [G loss: [141.89007568359375, 0.125]]
6899 [D loss: 234.90774536132812, acc.: 56.25] [G loss: [102.31061553955078, 0.15625]]
6999 [D loss: 148.42638397216797, acc.: 68.75] [G loss: [187.09686279296875, 0.1875]]
7099 [D loss: 91.617431640625, acc.: 75.0] [G loss: [188.85888671875, 0.21875]]
7199 [D loss: 177.92646026611328, acc.: 68.75] [G loss: [177.44473266601562, 0.09375]]
7299 [D loss: 106.66166114807129, acc.: 71.875] [G loss: [134.36099243164062, 0.21875]]
7399 [D loss: 132.69076919555664, acc.: 70.3125] [G loss: [85.54830932617188, 0.25]]
7499 [D loss: 176.7660675048828, acc.: 62.5] [G loss: [183.927001953125, 0.21875]]
7599 [D loss: 95.7921142578125, acc.: 73.4375] [G loss: [249.77142333984375, 0.15625]]
7699 [D loss: 111.27349853515625, acc.: 68.75] [G loss: [149.70272827148438, 0.1875]]
7799 [D loss: 170.58689880371094, acc.: 57.8125] [G loss: [156.9602813720703, 0.28125]]
7899 [D loss: 182.81645965576172, acc.: 62.5] [G loss: [168.09307861328125, 0.28125]]
7999 [D loss: 172.38062286376953, acc.: 59.375] [G loss: [190.38397216796875, 0.25]]
8099 [D loss: 133.86151123046875, acc.: 67.1875] [G loss: [109.23738861083984, 0.21875]]
8199 [D loss: 102.68619918823242, acc.: 73.4375] [G loss: [212.07530212402344, 0.25]]
8299 [D loss: 117.41545104980469, acc.: 68.75] [G loss: [86.07742309570312, 0.09375]]
8399 [D loss: 134.27285766601562, acc.: 62.5] [G loss: [93.69859313964844, 0.1875]]
8499 [D loss: 93.24633407592773, acc.: 76.5625] [G loss: [171.09109497070312, 0.1875]]
8599 [D loss: 118.39128684997559, acc.: 73.4375] [G loss: [120.96672058105469, 0.125]]
8699 [D loss: 223.34209060668945, acc.: 54.6875] [G loss: [160.44338989257812, 0.0625]]
8799 [D loss: 87.87305450439453, acc.: 62.5] [G loss: [79.13565063476562, 0.3125]]
8899 [D loss: 94.09722900390625, acc.: 70.3125] [G loss: [104.64747619628906, 0.21875]]
8999 [D loss: 101.21297836303711, acc.: 68.75] [G loss: [100.7119369506836, 0.21875]]
9099 [D loss: 89.4960708618164, acc.: 73.4375] [G loss: [161.2245330810547, 0.09375]]
9199 [D loss: 71.01003646850586, acc.: 71.875] [G loss: [177.22207641601562, 0.28125]]
9299 [D loss: 82.5373420715332, acc.: 67.1875] [G loss: [131.57339477539062, 0.125]]
9399 [D loss: 125.01629638671875, acc.: 60.9375] [G loss: [89.53704071044922, 0.3125]]
9499 [D loss: 172.48015594482422, acc.: 57.8125] [G loss: [134.3751220703125, 0.15625]]
9599 [D loss: 119.97325134277344, acc.: 67.1875] [G loss: [169.36643981933594, 0.125]]
9699 [D loss: 179.14397430419922, acc.: 59.375] [G loss: [128.00344848632812, 0.28125]]
9799 [D loss: 90.28621292114258, acc.: 67.1875] [G loss: [199.62713623046875, 0.125]]
9899 [D loss: 87.00929641723633, acc.: 68.75] [G loss: [174.411376953125, 0.1875]]
9999 [D loss: 93.66770935058594, acc.: 64.0625] [G loss: [174.06640625, 0.28125]]
</pre>

```python
def generating_data(num):
    noise = np.random.normal(0, 1, (num, 32))
    sampled_labels = np.ones(num).reshape(-1, 1)

    gen_samples = cgan.generator.predict([noise, sampled_labels])
    #gen_samples = scaler.inverse_transform(gen_samples)
    return gen_samples
```


```python
X = np.array(x)
y = np.array(y)

origin_x = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components = 3)
origin_x = pca.fit_transform(origin_x)

# plot
fig = plt.figure(figsize = (15,15))
axs = plt.axes(projection = '3d')
for i in range(origin_x.shape[0]) :
    if y[i] == 1 :
        axs.scatter3D(origin_x[i,0], origin_x[i,1], origin_x[i,2], color = 'orange')

# PCA
pca = PCA(n_components = 3)
new_x = generating_data(1000)
new_x = pca.fit_transform(new_x)

# plot
for i in range(new_x.shape[0]) :
    axs.scatter3D(new_x[i,0], new_x[i,1], new_x[i,2], color = 'red')
plt.title('Scatter plot of real and synthetic fraudulent samples')
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAz0AAANNCAYAAAC9ShC0AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAAsTAAALEwEAmpwYAAEAAElEQVR4nOz9eXwceX3nj78+fd/dakm2LlvyzHhsj+0Zjy1pApnwZQgkLEnIwYSwsBAIBObKQkjCDDlYcnEkkw0BJpxhw7K/BEK+m2W/WZJNsoFkCecMM5Ys2ZJ1X5Ytqbulvo+qz++P7ipXl6rVV3V3dev9fDz8eFh9VH2qurr68/q83+/Xm3HOQRAEQRAEQRAE0amYWj0AgiAIgiAIgiCIRkKihyAIgiAIgiCIjoZED0EQBEEQBEEQHQ2JHoIgCIIgCIIgOhoSPQRBEARBEARBdDQkegiCIAiCIAiC6GhI9BAE0ZEwxjhj7K4m7Icxxv4LYyzMGPtuE/b3dcbY25qwn5HCObQ0el9lxqHb58gY+yHG2EwN72vqZ1zY5xJj7OV6v7YTYIz9OWPs91o9DoIg2gsSPQRxiGGMPcgY+yZjbJcxFmKM/RtjbKzObb6ZMfYN1WOGnaRojbdKHgTwCgBDnPNxnYZ1KNFb0KkFE+f8/3LOT9WwqUPxGTPG3s8Y+2+tHgdBEEQjaOkKHkEQrYMx5gPwtwAeBfBXAGwAfghAupXj0oIxZuGc51o9jhIMA1jinMcrebHBj4XQ5sDPmD5TgiAI40ORHoI4vNwNAJzzv+ScC5zzJOf8HzjnE9ILGGO/yBi7yhiLMsamGWMXC48/xRibVzz+04XHzwD4JIAXMcZijLEIY+ztAN4A4D2Fx/6/wmsHGGP/L2NsizG2yBj7j4r9vp8x9teMsf/GGNsD8Gb14AvRo08yxv6xMI5/YYwNax0oY8zPGPuvhX0tM8Z+kzFm0hpvifcPMMb+ZyEaNscY+8XC428F8FnF+39b471vLkTQ/pgxtgPg/YwxO2PsacbYCmPsZuE4nIXXdzHG/rYw1nDh/0MHf5TyvsYZY98qnPcbjLGPM8Zsiuc5Y+wRxtj1wmueYYyxwnPmwpi2GWMLAH6szL6eZIytF879DGPshxljfYyxBGOsW/G6i4VjsUpRtcJ+woXP/d8VXvf7yIvujxfO5ccVu3u51pgL7/uFwjUaZoz9b+kaYIz9a+Ellwvb+znG2EsZY2uK9x5jjP33wvh2VPuUXrPvM5a2UzgHmwD+S7nPjalS0JgqqsIYe2Ph2txhjP2GagxFkVL1cahea2K3v587jLG/YowFC89JKYs/X7j2tqV9McZeCeDXAfxc4Tgvl9j+vs+98Hgl195jhc8xyhj7XcbYnSwfad4rjNOmPD7G2K8XxrjEGHuD1ngKr/9xxtgLhX1/kzF2b7nxEgRxCOGc0z/6R/8O4T8APgA7AD4P4N8B6FI9/7MA1gGMAWAA7gIwrHhuAPmFk58DEAfQX3juzQC+odrWnwP4PcXfJgDPAXgf8hGmOwAsAPjRwvPvB5AF8FOF1zo1xv/nAKIAXgLADuBPlPsFwAHcVfj/fwXwFQBeACMAZgG8tdR4Nfb1rwD+FIADwAUAWwBeVsn7C8/nAPwS8tF1J4A/BvA/AQQLY/r/AHyw8PpuAK8B4Co892UA/0Oxva8DeFuJfV0C8AOF/YwAuArgXapz8rcAAgCOF47jlYXnHgFwDcCxwri+Vni9RWM/pwCsAhgo/D0C4M7C/78K4FHFa/8YwMcU5yIL4BcBmJGPMm4AYKWOrcyYfxLAHIAzhWP+TQDf1LoGCn+/FMBa4f9mAJcL43MXPtsHD/gMv6HaTg7Ah5G/9pwVfG5LAF6u+Pv9AP5b4f/3AIjh9rX8nwvbf3mJ7498HOptA3gngG8DGCps61MA/lLxOXEAnymM+T7kI7tn1GMqcR4O+twrufa+gvx952xhv/8H+e++H8A0gJ9Xnd//XDiG/wf5e8wp9fkAcD+AWwAeKHymP184H/aDxkv/6B/9O3z/KNJDEIcUzvke8rUK0iRoi+WjGUcLL3kbgD/gnH+P55njnC8X3vtlzvkG51zknH8JwHUA1dQ6jAHo5Zz/Duc8wzlfKIzhdYrXfItz/j8K+0iW2M7/4pz/K+c8DeA3kF+NP6Z8AWPMXNjueznnUc75EoA/AvDGSgZa2N4PAniSc57inL+A/Mr/myo+WmCDc/4xnk+BSgF4O4Bf5pyHOOdRAB8ojBGc8x3O+f/LOU8Unvt95Cd9ZeGcP8c5/zbnPFc4zk9pvPdDnPMI53wFeWFzofD4awF8hHO+yjkPAfjgAbsSkJ9U3sMYs3LOlzjn84XnPg/gPwDyuf/3AL6geO8y5/wznHOh8Np+AEdxMKXG/AjyYvFq4dx+AMAFViLip2IceeH+a5zzeOGzraa2SwTwnzjnaZ6Pktb8uQF4GMDfKq7l3ypsvxYeAfAbnPO1wrbeD+BhVmxI8duFMV9GXvjdV+G2S37uFV57f8A53+OcTwG4AuAfOOcLnPNdAH+HvIBR8luF8/svAP4X8teomrcD+BTn/Ds8H7H+PPKC6gcOGi9BEIcPEj0EcYgpTBbfzDkfAnAO+UngRwpPHwOgOUFgjL1JkU4SKby3p4pdDwMYkN5f2Mavo3jyu1rBduTXcM5jAEKFY1DSA8AKYFnx2DKAwQrHOgBAEie1vL9onAB6kY8GPKc49r8vPA7GmIsx9qlCqtMe8lGmQEFAHAhj7O5CWtVm4b0fwP7PZVPx/wQAT+H/A6pxKs9XEZzzOQDvQn5CfYsx9kXGmHTev4L8JPME8sX/u5xzpePZpmI7icJ/PTiYUmMeBvAnivMYQj4qWclncwx5AVZrLc4W5zwl/VHP5wbVuef52qGdGsc1DOBvFOfkKvKTf+V3q9T5PJCDPvcKr72biv8nNf5WjiPMi2uolrH/uw3kj/dXVPeSY8hHdw66TgmCOGSQ6CEIAgDAOb+GfNrIucJDqwDuVL+usIr+GQBPAOjmnAeQX7WV6iy41uZVf68CWOScBxT/vJzzVx3wHi3kqA5jzIN8WtaG6jXbyKdUKVf/jyOfulfJfjYABBlj3hLvrwTlPraRn+CdVRy7n3MuTfh+Bfm0nAc45z7kU56A2+f3ID6BfIraycJ7f73C9wHADSjOJ/LHWBLO+V9wzh9E/rxy5FO9UBACf4V8tOeNKI7ylKOSz1zJKoB3qK4jJ+f8mxW+9zir3ZJbPdZyn1scebEr0af4f9G5Z4y5kE+XkzjovWpWAfw71TlxcM4ruV7Lnv9Snzvqu/a06GKMuRV/H8f+7zaQP97fVx2vi3P+l2XGSxDEIYNED0EcUhhjpxljv8IKxdaFNK5/j3w9AJBP4fpVxtgllueuguBxIz952Cq87y24LZSA/OrtkLKIufDYHYq/vwsgWigydrJ8Ef05Vr1d9qtY3nbbBuB3AXybc14UISqkUf0VgN9njHkLx/BuAFIRudZ4le9fBfBNAB9kjDkKRdJvVby/KjjnIvKi8Y8ZY0cAgDE2yBj70cJLvMiLogjLF6D/pyo27wWwByDGGDuNfM1MpfwVgP/IGBtijHUBeKrUCxljpxhjL2OM2ZFP10uiOB3rvyJfB/NqVCd61NdJOT4J4L2MsbOFcfkZYz9b4fa+i7zY+BBjzF34bH+win2rKfe5vQDgdSxv6DCKfEqbxF8D+HHFtfw7KP59fgH5az3IGOtDPnpRik8if61Lhg69jLGfrPAYbgIYYYxpzg3KfO71XHul+G3GmI0x9kMAfhz5Oik1nwHwCGPsgcJ9ys0Y+7HCd73cdUoQxCGCRA9BHF6iyBf/focxFkde7FxBfsUanPMvI1+X8BeF1/4PAEHO+TTyNTHfQn6SdB7Avym2+88ApgBsMsa2C4/9GfIpTxHG2P8oCJEfR742YxH56MdnkS9oroa/QH5yGUK+kPo/lHjdLyG/Wr4A4BuF933ugPGq+ffIF0FvAPgb5Gs5/qnKsSp5EvkC/G8XUoH+CfkoAZBPL3Qif06+jXzqW6X8KoDXI/95fQbAl6p472cA/G/kazy+D+C/H/BaO4APFca4CeAIgPdKT3LO/w35yeX3eaEOrEL+BPn6kzBj7KPlXsw5/xvkV+6/WDiPV5A35ZB4P4DPF66716reKwD4CeQNOlYArCFvylErH8HBn9tvIR85DQP4beSvQWksUwAeLzx2o/AapTvbF5D/XJYA/AMO/lz/BHmTjH9gjEULY3mgwmOQRMUOY+z7Gs8f9LnXc+1psYn8edgA8P8D8EghGl0E5/xZ5I0xPl54/Rxuuz0eeJ0SBHG4kBxzCIIg2grG2J8j72D1m60eC7Efxtg/A/gLzvlnWz0Wor1gjL0UeRe5iqzaCYIgKoGakxIEQRC6UkhTvIi8pTRBEARBtBxKbyMIgiB0gzH2eeTT9d6lcrwjCIIgiJZB6W0EQRAEQRAEQXQ0FOkhCIIgCIIgCKKjKVfTQ2EggiAIgiAIgiDaBc0eYRTpIQiCIAiCIAiioyHRQxAEQRAEQRBER0OihyAIgiAIgiCIjoZED0EQBEEQBEEQHQ2JHoIgCIIgCIIgOhoSPQRBEARBEARBdDQkegiCIAiCIAiC6GhI9BAEQRAEQRAE0dGQ6CEIgiAIgiAIoqMh0UMQBEEQBEEQREdDoocgCIIgCIIgiI6GRA9BEARBEARBEB0NiR6CIAiCIAiCIDoaEj0EQRAEQRAEQXQ0JHoIgiAIgiAIguhoSPQQBEEQBEEQBNHRkOghCIIgCIIgCKKjIdFDEARBEARBEERHQ6KHIAiCIAiCIIiOhkQPQRAEQRAEQRAdDYkegiAIgiAIgiA6GhI9BEEQBEEQBEF0NCR6CIIgCIIgCILoaEj0EARBEARBEATR0ZDoIQiCIAiCIAiioyHRQxAEQRAEQRBER0OihyAIgiAIgiCIjoZED0EQBEEQBEEQHQ2JHoIgCIIgCIIgOhoSPQRBEARBEARBdDQkegiCIAiCIAiC6GhI9BAEQRAEQRAE0dGQ6CEIgiAIgiAIoqMh0UMQBEEQBEEQREdDoocgCIIgCIIgiI6GRA9BEARBEARBEB0NiR6CIAiCIAiCIDoaEj0EQRAEQRAEQXQ0JHoIgiAIgiAIguhoSPQQBEEQBEEQBNHRkOghCIIgCIIgCKKjIdFDEARBEARBEERHQ6KHIAiCIAiCIIiOhkQPQRAEQRAEQRAdDYkegiAIgiAIgiA6GhI9BEEQBEEQBEF0NCR6CIIgCIIgCILoaEj0EARBEARBEATR0ZDoIQiCIAiCIAiioyHRQxAEQRAEQRBER0OihyAIgiAIgiCIjoZED0EQBEEQBEEQHQ2JHoIgCIIgCIIgOhoSPQRBEARBEARBdDQkegiCIAiCIAiC6GhI9BAEQRAEQRAE0dGQ6CEIgiAIgiAIoqMh0UMQBEEQBEEQREdDoocgCIIgCIIgiI6GRA9BEARBEARBEB0NiR6CIAiCIAiCIDoaEj0EQRAEQRAEQXQ0llYPgCAIol3gnCOXyyGZTMJsNsNqtcJsNsNkMoEx1urhEQRBEARRAsY5P+j5A58kCII4LHDOkclkIIoiMpkMlPdOxhisVissFguJIIIgCIJoLZo/wCR6CIIgyiAIArLZLDjnYIwhk8kUiRrOOURRlJ9XiiCLxSI/RhAEQRBEwyHRQxAEUQ1SOlsul5OFixTxOUjElBJBUjociSCCIAiCaBgkegiCICpFFEVks1mIolgkUioRPUqke6woivJj4XAYR48eLaoJIgiCIAhCFzR/oMnIgCAIQgHnXE5nA1B3VEZ6r9lslre/uLiIrq4uZDIZAIDJZILFYiERRBAEQRANgkQPQRBEAc45stksBEFoWAqalggCgEwmUySC1MYIBEEQBEHUDokegiAIoMiVrZk1N1oiiHOOdDqNdDoNgEQQQRAEQdQLiR6CIA41ynQ2xljLBYVacGmJIGWPIMkdjiAIgiCI0pDoIQji0KLsvWNURzUtESSKIlKplPyYJIKkSJARj4MgCIIgWgmJHoIgDiWtSmerFxJBBEEQBFE9JHoIgjhUqHvv1JLOpoeIkMRWvZAIIgiCIIjykOghCOLQUKr3TidRSgQlk8ki0wQSQQRBEMRhgkQPQRAdj2RWcP36dRw/frzlE33GmG6Rnkr2pYxoaYkgi8Ui/2v1uSEIgiCIRkCihyCIjkaZznbr1i0MDw+3fFIviZ5W7VstgpTudQCKGqWSCCIIgiA6ARI9BEF0LFrpbHqIjWZFaZpBKRGUy+Xk10giyGKxwGQydcyxEwRBEIcHEj0EQXQcpXrvtDLCosQo49BCqyZIKYIYY0XpcCSCCIIgiHaARA9BEB0F5xzZbBaCIOybwBtZbBgVLRGUy+WQzWbl50kEEQRBEEaHRA9BEB1Dud47eoiezc1NXL9+HXa7HV1dXejq6oLX663K+rqdxZeWCMpmsySCCIIgCENDoocgiLan0t479YgNQRBw9epV5HI5jI6OIpfLIRKJYGNjA9FoFHa7HYFAQBZBB03021n0qGGMwWw2y39riSClPTaJIIIgCKIVkOghCKKt4Zwjk8lU1HunVrERjUZx5coVDA0NYWhoCNlsFmazGX19fejr6wMApFIphMNhrK2tIRaLweFwyJEgt9u9L82uU9ESQZlMBul0Wv58JBFksVg6tl8SQRAEYSxI9BAE0baoa0vKTZ6rFT2cc6ytrWFtbQ3nzp2D1+st+VqHw4H+/n709/eDc45kMolIJILl5WXEYjG4XC5ZBHHOOybSU45SIujatWvo6+uD2+2G1WqVLbJJBBEEQRCNgEQPQRBthzqdrdJJcjWiJ5vNYmpqChaLBePj40UT90r243K54HK5MDAwAM45EokEwuEwFhYWsLu7i5mZGfT09CAQCMDpdB6aib4kgpRiNZPJIJPJAABMJlNRn6BqaqUIgiAIohQkegiCaCu0eu9USqWiJxKJYHp6GidOnEB/f389w5X363a74Xa7MTQ0hMuXL2NwcBCJRAJzc3NIpVLweDxyJMjhcNS9z3ZA+vwkQSl9NiSCCIIgCL0h0UMQRFug7L0DoKbJbznRwznH0tISbt68iQsXLsDlctU83oMwmUxwuVzo7u7GsWPHwDlHNBpFOBzGtWvXkMlk4PP5ZGMEu93ekHEYDUnAkggiCIIg9IZED0EQhueg3jvVcJDoyWQymJychNvtxvj4eEMn01pW2j6fDz6fD8PDwxBFURZB09PTyOVy8Pl8ciTIarU2bGxGQksESTVBShGkdocjCIIgCDUkegiCMDTleu9UQynRs7Ozg2vXruHuu+9Gb29vPcOtmIMiTiaTCX6/H36/HyMjIxAEAXt7e7I7nCAIchQoEAjAYjkct3KtHkGcc6TTaaTTaQB5gSRFgSR3OIIgCII4HL+UBEG0HZX23qkGtegRRRHz8/OIRCK4dOlS02ppqnWRM5vNcpQHyPcM2t3dRTgcxvLyMjjnsgjy+/1tI4LqdbDTEkGiKCKVSsmPSSJIigSRCCIIgjictMcvI0EQh4pqeu9Ug1JspFIpTExMoLu7G6Ojo201GTabzQgGgwgGgwAgN0oNhUJYXFwEY0yOAvn9/qqc55pFI843iSCCIAiiFCR6CIIwFJJZgR7pbGok0XPr1i1cv34dZ86ckYVDM6m1SWopLBYLenp60NPTAyBvtx2JRLC9vY35+fmiSJHP5zs0dS8kggiCIAgJEj0EQRiCRqSzaSGlg42NjcFms9W0jXpFS6Mn1larFb29vXJ9UiaTQTgcxs2bNzE7Owur1SqLIK/Xe+hFUDKZLDJNIBFEEATReZDoIQii5dTTe6dS4vE4Njc3ceTIEdxzzz0tn8zqGekph81mw9GjR3H06FEAQDqdRjgcxsbGBqLRKOx2uyyCPB5Py89Ns5CuNUn0kQgiCILoXEj0EATRMtS9dxoleDY2NrC0tISenh4cOXKk5RNXvdPbqsVut6Ovrw99fX0AgGQyiXA4jJWVFcRiMbhcLtkYwe12t/x8NYtKRJDFYpH/kQgiCIJoH0j0EATREvTqvXMQuVwOV69eBecc4+PjWFxcbKnYkDDaRNnpdMLpdGJgYACcc1kELS0tIR6Pw+12y5Egp9NpuPE3Ci0RJAgCcrmc/BqpUarFYoHJZDo054YgCKLdINFDEETT0bP3Tin29vZw5coVDA8PY3BwEEDrIyxKjDIONYwxuFwuuFwuDA4OgnOOeDyOSCSC+fl5JJPJfSLosKBVE6QUQYyxokgQiSCCIAjjQKKHIIimoUxna5RZAeccq6urWF9fx7333guPxyM/ZxTRY5RxVAJjDB6PBx6PB0NDQ+CcIxaLIRwOY3Z2Ful0Gl6vVxZBdru91UNuGloiKJfLFaVrkggiCIIwBiR6CIJoCo3qvaMkm83iypUrsNvtGB8f39efpp3EhlFhjMHr9cLr9eL48eMQRRGxWAyhUAhXr15FJpOBz+eTRVCtDnntCIkggiAI40KihyCIhtOMdLZwOIzp6WncddddskuZGqOIHqOMQw9MJhN8Ph98Ph+A/Ge9t7eHcDiM9fV1CIIAv98vN0u1Wq0tHnHz0BJB2Wy2SASZTCZYrVZYrVYSQQRBEA2ERA9BEA2jGb13OOdYWFjA9vY2Ll68eGCNiVHEhlHG0QhMJhMCgQACgQBOnDgBQRBkEbSysgLOOfx+P1KpFARBaPVwmwpjrCj6yDnHysoKTCYT+vv7wRiTTREsFkvDFggIgiAOIyR6CIJoCM3ovZNOpzE5OQmfz4exsbGyoqqTxYZRMZvNcqobkHfU293dRSgUwszMDCwWi2yP7ff796UkdjLSQoDJZILZbJZTQNPpNAAURYEke2wSQQRBELVBoocgCF1pVu+d7e1tzMzM4NSpU+jp6anoPXqJnnqP5zCLL4vFgu7ubmxtbWFgYAAulwuRSAQ7OztYWFiQI0VdXV3w+XwdL4KklE9AOxIkiSBJIEkW2WazuSGRU4IgiE6FRA9BELqhTmdrhNgRRRHXr19HNBrF6OhoVW5hRhI9RB6LxYKenh5ZuGYyGUQiEdy6dQtzc3OwWCxypMjr9R6qib5SBEnXbSaTQSaTAQASQQRBEFVAoocgCF1oRjpbIpHA5OQkent7cenSpar3wRiDKIp1j0MP4XRYIz3lsNlsOHLkCI4cOQIgn8IYiURw48YNzMzMwGazFYmgdheQykjPQUivIRFEEARRGyR6CIKoi2b03gGAzc1NzM/P45577pHrQ6rFKGllRhlHO2C323H06FHZkS+VSiEcDmNtbQ2xWAwOh0N2hvN4PG0vgipFSwRJ6XBKESQZI5AIIgjisEOihyCImpEseAVBaFh0RxAEXLt2DZlMBuPj43VZHpPYaH8cDgf6+/vR398PzrksglZWVhCLxeByueRIkMvlOlQiSG2PzTlHOp3eZ4ygdIcjCII4LJDoIQiiJprReycWi2FychKDg4M4duyYoQwEKk1LavQ4DjOMMTidTjidTgwMDIBzjkQigXA4jMXFRcTjcXg8HtkYwel0Gm6iX891dBCViCCz2SynwpEIIgii0yHRQxBEVTSr9876+jpWVlZw/vx5eL1eXbZrFLFBk8vGwBiD2+2G2+3G0NAQOOeIx+MIh8OYm5tDKpWCx+ORI0EOh6PVQ26Y6FGjJYJEUUQqlZIfk0SQlA5H1ylBEJ0EiR6CICpGqhlopFlBLpfD1NQUTCYTxsfHYbHod5syiugByMigGTDG4PF44PF4cOzYMXDOEY1GEYlEMDMzg3Q6Da/XK4ugapwA2x0SQQRBHDZI9BAEURG5XK7hvXd2d3cxNTWFkZERDAwM6L59o4geo4zjsMEYg8/ng8/nw/HjxyGKIqLRKMLhMKanp5HL5eDz+WRjBJvN1vAxGeU6IBFEEESnQ6KHIIgDaUbvHc45lpeXsbm5ifvuuw9ut1v3fQDGERs0WTTGOTCZTPD7/fD7/RgZGYEoitjd3ZXd4QRBgN/vl0VQPSYaB2GEc6GmlAhKJpNFznEkggiCaBdI9BAEURJRFLGzsyNbATdiUpPJZDA5OQmXy4Xx8fGG2uoaRfQAxlnhJ25jMpnkVDcg7xwoiaCVlRVwzmVTBL/fr2vqpdGRvv/S91NLBEmucCSCCIIwIofnjk0QRMUoe+9MTk7iB3/wBxuyn1AohKtXr+LkyZNyM8pGopfoMZKLXDtj9HNgNpsRDAYRDAYB5FM8I5GI7A7HGCsSQVLPnGpolpGB3miJIGW/LgByo1SLxQKTydSWx0kQROdAoocgiCKa0XuHc475+XmEQiFcunSpaS5aJDaIerBYLOjp6UFPTw8AIJvNIhKJYHt7G/Pz8zCbzXKkyOfzHapmoKVEUC6Xk59XpsORCCIIotmQ6CEIQqYZvXdSqRQmJycRCAQwNjbW1ImPUUSPUcZB1IfVakVvby96e3sB5FM1I5EIbt68ievXr8NiscgiyOv1aoqgdo30lEOrJkgtgpTpcCSCCIJoNCR6CIJoSu8dANja2sLs7CxOnz6N7u7uhuzjIIwyqWKMQRTFVg+D0BmbzYYjR47IqZrpdBrhcBgbGxuIRqOw2+2yCJLq5A4LWiJI7QgpNUm1Wq0kggiC0B0SPQRxyGlG7x1RFDE7O4t4PI6xsbGmWAFrQWKDaCZ2ux19fX3o6+sDACSTSUQiEayuriIWi8HhcEAQBDgcjo6N+JRCSwRJAvHUqVNyJMhqtcJsNpMIIgiibkj0EMQhRio8bmQ6WyKRwMTEBPr6+uTJTKswyqSJ0tsOJ06nE06nE/39/eCcI5lMYnZ2Fjdv3sTa2hrcbrdsj+1yuQxzvTYD6f5jNpthNpvl2kJlJEhpj00iiCCIaiHRQxCHkGals924cQOLi4s4e/Ys/H5/Q/ZRDRTpIYwCYwwulwtutxs9PT0IBAJIJBIIh8NYWFhAMpmURVBXVxecTmerh9wUJCEjCSAJKSKdTqcB5O3FJRFksVgatmhDEETnQKKHIA4Zoigim81Wlc5WbeqNIAi4evUqcrkcxsfHDdPPxCiTIor0EGoYY3C73XC73RgaGgLnHLFYDOFwGLOzs0in0/B6vbJFdrMcD5uJdE/S4iARJN3HrFarnA5HIoggCDXGmIkQBNFwlH00gP059aUwmUwQRbHiHiTRaBRXrlzBsWPHMDg4aLiJhxEiPSR6CIlS1wFjDF6vF16vF8ePH4coirIIunbtGjKZDHw+nxwJalWdnJ5Us7iiFEHSOcxkMshkMgDy9y11TRBBEIcbEj0EcQiop/dOpRN0zjnW1tawtraG8+fPw+Px1DPkhqDHxCedTmNiYgKpVAperxfBYBBdXV2w2+0Vb8NoQpBoLZUuPvh8Pvh8PgwPD0MURezt7cnF/7lcDn6/X44EWa3WJoxcX2o1c5DeQyKIIIiDINFDEB1Ovb13pEjPQWSzWUxNTcFqtWJ8fLymzvTNop4Iy87ODq5du4aTJ0/C4/EgkUggFAphenoa2WwWfr8fwWAQgUCg7KTzsEd6SPjdppZzYTKZEAgEEAgEAORTSiURtLq6Cs45/H6/bIxglBTTg9DLwU5LBEnpcEoRpDZGIAiiszH+XZAgiJpQprPVY1ZQLtITiUQwPT2NO+64Q7bmNSomk6kmscE5x/z8PEKhEC5dugSbzSanF/l8PoyMjBRNOldWVsA5l1fdA4FAkRCk9DZCQq/rwGw2y6luQF4ERSIRhMNhLC0tAUDJ69EoNMq2W8sem3OOdDqtaYxAIoggOhMSPQTRgejZe6dUpIdzjqWlJdy6dQsXLlyAy+WqZ8hNo9pJppTOFggEMDo6WvJ8qCeduVwOkUgEoVAICwsLRc8boa6IMA6NmOibzWZ0d3fLTYCV1+Pi4qIcKerq6oLP5zOECGpWr6JKRJDZbJZT4SR3OIIg2hsSPQTRYdSbzqZGKyqRTqdx5coVeDwejI2Ntc2qaLWRHimd7dSpU+jp6alqXxaLBT09PfL7MpkMwuEwNjc3EQqF5ELsrq4ueDyeQzmpomhX886B+nrMZrMIh8PY2trC3NwcLBZLkQhqxXe6VQ1atUSQKIpIpVLymCQRJEWCDuP3lSDaHRI9BNEhNKr3jjqyIQmBu+++G729vbrso5lUasqgTGfTwx7YZrPh6NGjOHr0KHZ2drCzswOr1YqVlRXEYjG4XC45EnTYGlMSzcdqteLIkSM4cuQIgGJRPjs7C5vNJl+PHo+nKSKoVaJHzUEiSIJEEEG0HyR6CKIDqKX3TqVIkR5RFDE3N4fd3V3dhECzqSTSo5XO1gjMZjP6+/vR398PznlRY8pEIgGv1ytPOtvxXBOVYZSJvlKUA0AqlUI4HMb6+jqi0SgcDoccCWpUZNIo50INiSCC6AxI9BBEG1Nr751qMJlMSCaTmJ6eRk9PD0ZHR9v2B72cgUA96Wz1jEOrMWU0Gi3qyaJ04uqEniyEsXE4HLIoB4BkMimbdMTjcTidTt0jk0YVPWpKiaBkMlnkHEciiCCMBYkegmhT1OlsjfpRTSaTuHbtGs6fPy8X6bcrpURPI9LZ6oExtq8ny+7uLsLhMNbW1iAIQpETVzvYERPatMtE3+l0wul0YmBgQI5MRiIRLC4uIh6Pw+12yyLI6XTWdEztci7USPdfKSqsJYIsFov8j0QQQbQG+qUkiDakkelsEoIgYGZmBolEAvfcc0/bCx6gtClDM9LZyo3jIEwm0z5nOEkELS0tgTEmP+/3+9vGWIJoT5SRycHBQXDOEY/HEQ6HMTc3h1QqBY/HI4typ9NZ0XY55x1x7WqJIGX7AAByo1SLxQKTyUQiiCCaAIkegmgj9Oq9U454PI6JiQkMDAw0dD+tQCk2mpXOpqbeCY7FYimyI5acuG7duoXr16/DarUiGAyiq6sLXq+XJlQGpl2jG0oYY/B4PPB4PDh27Bg454jFYgiHw5idnUU6nS6qUbPb7Zrb6YRzoUUpEZTL5eTnlelwJIIIojGQ6CGINoFzjmw2C0EQGprOtr6+juXlZZw7dw4+nw8zMzMd01dGOmd6pLPVe/71tCpWO3FJRehra2uIRqNy/UUwGCRnOKLhMMbg9Xrh9Xpx/PhxiKIo16hNT08jm81q1qh1quhRo1UTpBZBynQ4EkEEoQ8kegiiDdC7944WuVwO09PTAIDx8XG5TqTa3jZGRxRFPPvss01NZ1NTbXpbtSiL0DnnchG6VH8hpR5J9RdE6zgME32TyQS/3w+/34+RkRGIooi9vb2iGjW/349cLodAINDq4TYdLRGUy+WQzWYxNzeHkydPkggiCB0g0UMQBqZRvXfU7O3t4cqVKxgeHsbg4GDRc42eoDeTnZ0dJBIJnD59uqnpbGqaOWFhjMHlcsHlcsn1F+rUI5/PJ4ugZjrDdcp1RVSHyWRCIBBAIBDAiRMnIAgCdnd3sbS0hJWVFWxsbMhGHX6//9AZdShF0N7eHhhjyGazRS6dUk2Q2WwmEUQQFXK47iQE0UZwzpHJZBpqVsA5lycZ9913H9xu977XqJuTtiPKdDaXy9VSwaMcUysolXoUCoWwvr4ur7oHg8GGOsPRJC3PYYj0lMNsNiMYDGJ3d1eOQkpGHYuLi2CMFYkgs9nc6iE3BekeIdX8KB9XiyClPTaJIILQhkQPQRgQKbUBaEzvHSDfgX1qagp2ux3j4+MlJxLtHulRu7N9+9vf1mW79UxWjXROlalHylX3UCgkO8Mdxgkn0Xyk75SWUUckEsH29jbm5+dhNpuLrslOMlpRUuoeoyWCMpkM0um0/HshiSCLxdLQGlCCaCdI9BCEgWhW7x2poPiuu+6SO7CXop0jPY10Z+vUSYS06h4MBgHsn3BaLBY5Fc7r9XbshJNoPqUm+VarFb29vejt7QWQX7CJRCK4desW5ubmOvaarHRhpVIRJKXDkQgiDiskegjCIDSj9w7nHAsLC9je3sbFixcrKmI3UlSiUhrdbLTez6adzql6wplOpxEOh7GxsYFoNAq73S7bY7vdbppMVQmlt92m0nNhs9mK3Aq1rknJGa6dLdtFUaxJwClFkHSfyWQyyGQyAPILWeqaIII4DJDoIYgWo+y9A6BhP0CpVAqTk5MIBAIYGxureD8mkwmCIDRkTI2gFc1Gq6WdRI8au92Ovr4+9PX1AQCSyaScChePx+F2u4uc4dp1wtlM6BzlqVUAqq9JpWV7LBaDw+GQr8l2EuZ6NGuVjpVEEEGQ6CGIltKs3jtbW1uYnZ3F6dOn5Tz5SmmnCXqrmo0eZpxOJwYHB2VnuHg8jnA4jLm5OaRSqYqaUh5m2uW71Qz0inqVsmxfXl5GLBaD2+2Wa4KM3LdKivrrSaUiSG2MQBCdAIkegmgRzei9I4oirl+/jmg0itHR0Zomne0geqR0tnA43JB0Nr1ph3NaC4wxeDweeDweHDt2bF9TylwuJzelbNc6MaJxNCLVT8uyPZFIIBwOY2FhAclksig66XA4DCOCak1vqwYtEcQ5RzqdRjqdBkAiiOgcSPQQRJNpVu+dRCKByclJHDlyBJcuXar5h9zoRgbqdDajTFiI/U0pJWe4cDiMra0thMNh9PT0yPUXh9EZjmp6btOMc8EYg9vthtvtxtDQkNy3KhKJ4Pr160ilUkXNe1u5gNKKa0OrUapaBJnNZjkVTnKHI4h2gEQPQTSRZvTeAYDNzU0sLCzgnnvuqbvDuZFFT7ums3VqpKccSmc4URTR1dUFAAiFQlhYWIDZbJYnmz6fj1aUDxmtmuRLfauOHTsGzrkcnbx27RoymUzLmvc2I9JTDi0RJIoiUqmU/JgkgqRIEIkgwqiQ6CGIJiGZFTQynU0QBFy7dg3ZbBZjY2OwWq11b9OIE/R2S2dTY8Rz2gokkSMJ1kwmg3A4jM3NTczOzsouXF1dXfB4PB05maJIz22McC4YY/D5fPD5fBgeHi5K0dzY2EAulysSQXrcY0uhh5GB3pAIItoZEj0E0WCalc4Wi8UwOTmJoaEhDA0N6fZDY7QJeieks7XjmJuBzWbD0aNH5d5RUgH6ysoKYrEYXC4Xurq6EAwGyRmuAzGC6FGjlaK5t7cnu8OJoijXqQUCAVgs+k2rGmFkoDckgoh2gkQPQTSQZvXeWV9fx+rqKs6dOwev16vr9o2U3qZXOpsRJldGEpKtotw5cDqdcDqdGBgYKCpAn5ubQzKZLHKGa7don4QRrkWj0A7nQpmCCaCoTm15eRmcc9kZrt46NSOkt1VLKRGUTCaLTBNIBBGtgEQPQTQAde+dRgmeXC6HqakpmM1mjI+PN6QQ3AiRHj3T2aTjaeUPLf3IV38OtArQ1bUX0op7o9OOiMbQ6u9lLSjr1ID8PTkSiSAUCmFxcRGMsaI6tWru0e14PtRIv32SeNMSQRaLRf5HIohoJCR6CEJnmtV7Z3d3F1NTUzhx4gT6+/sbsg+g9ZEevdPZjCDiAIr01ItW7YW04r66ugpRFBEIBBAMBuH3+3VNO9KTTpjY6oURa1iqxWKxoKenR45EZ7NZRCIRbG1tYW5uriqzjnaM9JRDSwRJC4RKESS5w5EIIvTEmL8CBNGmNKP3Duccy8vL2NzcxIULF+ByuXTfh5JWioRGuLMZQfQYYQydhslkKko7yuVy2N3d1Vxx9/v9HTeZ7AQ6UQBarVb09vait7cXwH6zDpvNJotzj8dTdF12gggsRykRlMvl5NdIIshiscBkMnXcNUI0DxI9BKED0o366tWrOHnyZMN+qDKZDCYnJ+F2uzE+Pt6UH8RWRHo455ibm0MkEtHdnc0IgoN+tBuPxWJBd3c3uru7AeRX3MPhMG7duoXr16/DZrPJIsjr9bbsM+nEiX6ttEPhfr2ozTrS6TTC4TDW19cRjUaLHAulbIHDhFZNkFIEMcYgCAKcTidsNhuJIKIqSPQQRJ0oe+9sb2/j7rvvbsh+QqGQLKqOHDnSkH1o0WyR0Gh3NiOIHoDS25qN1WrFkSNH5O9OKpWSHbii0SicTieCwSC6urrgcrloItUCDqMAtNvt6OvrQ19fH4Bix8JIJAKr1QqTyYRAIAC3233ozo+WCLp27RpOnDghOzgqa4JIBBEHQaKHIOpAnc4G6P/DLUU9WtWTppkioRnNRo0geowwhsOOw+FAf38/+vv7wTlHMpmUm6QmEgl4PB55xd3pdDZsHHQd3OYwih41SsfCtbU1udXB0tIS4vE43G530XV52M4XYwyiKMo1P1JLCKVpEIkgohQkegiiBkr13tHbGSyVSmFiYgLBYBBjY2MtuXk3I72tkelsakhwEGoYY3C5XHC5XLIzXCwWQzgcxuzsLNLpdFFDSpvNpvv+CRI9ajjnsNvt6O/vx+DgIDjniMfjCIfDmJ+fRzKZhMfjkS2yGynOjYTS4EErEpTNZotEkNIem0TQ4YZED0FUyUG9dySBoEetjVR7cObMGdkOtRU0WvQ0u9moEUSPEcZAlIYxBq/XC6/Xi+PHj0MURbkh5fr6OgRBKOrFYlRnuHaDRE8xaiMDxhg8Hg88Hg+OHTumKc6VvavsdnsLR984DvqNZYwV2YJL6eeZTEZ+XhJBFouloQ6rhPGgOzVBVEglvXf0EAiiKGJmZgaJRAJjY2O6rypXSyMn6FI62+nTp+WC80ZjBMHBGINFCMMUeQ7cOQxub0wqH6EPUk1FIBDAiRMnIAgCIpEIwuEwlpaWwBiTRZDf76+6XxZNuggtyhk7aIlzqXfV9PQ0stks/H6/fG22+rdEL6pZWCwlgtLpNID8d9tqtcrpciSCOhsSPQRRAep0tlI3xXpFTzwex+TkJPr6+nD69GlD3HwbIRKamc6mxgiix7L+Zbz41uMwheyAmEXqwjPIDT7c0jERlWM2m/c5w0UiEWxvb2N+fh4Wi6XIGa7TbYf1xAj3PKNQbdaAyWSC3++H3+/HyMiIZoRSauAbCATatoFvPRFBpQiSfgeUkSCTyVTUJ4i+u50FiR6CKMNB6Wxq6hE9GxsbWFpawtmzZ+H3+2sdru7ond6WSqUwOTnZtHQ2Na0WPSy9DeflXwJDGsjlVxsdLzyOeM9LKeLTpqh7sUg2xBsbG4hGo3A4HLIIOowOXERt1NunRytCubu7i0gkgpWVFXDOi0TQYUvTlL6HJIIOD4frCieIKlB3iq7kZleLQMjlcrh69SpEUcT4+Ljhfnj0FAnb29uYmZlpajqbmpaLnuQyYLICYvL2gyYrWHL50ImeVkfcGoXShlhyhpNS4ZQOXMFgsGPPAVE/evctMpvNCAaDco2o1MBXrzTNdodEUOdjrNkVQRgEyQFGag5X6Q9PtaInGo1icnISx48fx+DgoCFXgPUYkzKdbXR0tKUFtnqJnlrPC3cOA2K2+EExm3/8EGHEa70RKJ3h1A5cs7OzSCQSuHr1ascXnxPVU2+kpxxaDXx3d3exs7ODhYUFOVIkiaDDNsHXEkFqYwSpJkjpDkcYFxI9BKFC3XunmslZpaKHc47V1VWsr6/j3nvvhcfjqWfIhqbV6WxqWh3p4fYepC48A9v3H4XJcrum57BFeQ4rageu7373uxgYGJCLz3O5XEfUXRD1o3ekpxxWqxU9PT1yj7RMJoNIJIJbt25hbm7u0Neqadljc86RTqdlYwSz2SxHgSR3OMI4kOghiAKleu9UQyWiJ5vN4sqVK7DZbBgfH+/oFAIjpLOpabXoAYDc4MN4ds2Pi6eC5N52yGGMFRWfS3UX4XBYrrsIBAIIBoOHMuXoMKNX+4NasdlsOHLkCI4cOQLgdq3ajRs3MDMzA5vNViSCmjHBN5KtuZYIEkURqVRKfkwSQVIkyChjP6yQ6CEI3LaxrMSs4CDKiZ5IJIKpqSnceeed6Ovrq3W4hsdI6Wxq9BA9giBgY2MDHo+n5h/7rCkAMXCprnEQnYdW3YXSGc5sNssTTZ/Pd+hW2w8TjU5vqxZlrRqQj+KHw2Gsra0hFos1xbDDSKJHDYkg40Oihzj05HK5A3vvVEMp0cM5x+LiIra2tnD//ffD5XLVvA+jI6WzdXV1GSKdTU29oieRSODy5cvw+/2IRCKIRqNwuVwIBoNyV3SjHTPRvlgsln0pR+FwGJubm5idnYXdbpcnmh6Ph669DqLZ6W3V4nA40N/fj/7+ftmwIxKJYHl5GbFYDC6XS742XS6XLsciCIKhhOBBkAgyHiR6iENLpb13qkFL9KTTaUxOTsLr9WJsbKxtbti1YMR0NjX1iJ6bN29ibm4OZ8+elcUN5xyJRAKhUAhzc3NIpVLwer2yCOqUhoCEMbDZbDh69CiOHj0KALIz3MrKijzRJAHeGRgt0nMQSsOOgYEB+b4YDoexsLCAZDIpuxYGAoGar01RFNs2xbOUCEomk0WmCSSCGgeJHuJQUk3vnWpQix5JBJw6dUpeqe1EjJzOpqYW0SOKImZnZxGPxzE2Ngar1Sq79zDG4Ha74Xa7cezYMbkreigUkhsCSjUZgUCgbX+wCWPidDrhdDqLJpqSAE8mk/B6vbI9tpG/l8R+jB7pOQjlfXFoaKjItVBaHPJ4PHIkqNIG1a2uc9ITae4hHQ+JoMZDooc4VCh77wDQ/eYpiR5RFDE3N4fd3V3Di4B6MXo6m5pqRU8qlcLly5fR29uLU6dOFb1fK79c2RVdaggYiUQQCoWwuLgIk8mErq4uCILQUT/gROtRC3DOOaLRqOwMl81mZWe4rq4ucoYzOJ10f1C7FiqvzZmZGaTTafh8Ptkiu9RvZiedEzWViCCLxSL/IxFUPSR6iENDrb13qsFkMiGVSuF73/seent720IE1EM7pLOpqUb0lDo+6fqpZDtms7moF4ZUk5HNZvHss8/Kxb/BYFC3vHeifWikkyBjDD6fDz6fD8PDwxBFUXaGW11dlZ3hpD4sRmuMfNhpp/S2atG6NpUCPZfLwefzyelwUppwJ4seNVoiSBAE5HI5+TVSo1SLxQKTyUS/H2WgOxxxKKin9041RKNR3Lx5ExcuXEBXV1dD9tEqlFGNdkpnU1OJWOGcY35+HqFQCJcuXao49aISpJqMlZUVjI6OyjUZi4uLiMfjlI5ENAwpyijdmyRnOCkKyRiTn292M8pW28gbkXZOb6sWZYR8ZGSkSKCvra3JacI2m+3QnBM1WjVBShEkpewHAgESQSUg0UN0NHr03qkEQRAwMzOD3d1dDA0NdZzgkYQCY6zt0tnUlBM9mUwGExMT8Pl8GB0dbejET1n8Ozg4CM45YrEYQqHQvkaVXV1dtBLfgbTy+6PlDCc1o7x+/XpT+7AY2Yq4lRzWc6IW6FL/qo2NDezu7uLZZ5899FFKtQja2dlBLBaTF8sYY0XpcCSCSPQQHYxevXfKEYvFMDk5icHBQXR1dSEejzdkP61EqlUKhUJtl86m5iDRI6VWnDx5Um7I10wYY/B6vfB6vRgeHi5qVLm8vNzSlfhGQKv7xkLdjFLZh0WyZtfbgliCRA9xEFL/KlEU4XK5cPz4cUQiETlKzhgrEkGH0TBGcraTjl1a9FW25DjsIohED9GRSGYFjU5nW19fx/LyMs6dOwefz4dbt2517ERubm4O0Wi07dLZ1GiJHs45lpeXsbm5aag+SupGldlsdt9KvGRP3G49WtpprIcVdR8WpQVxIpGQ3beCwWDdKaAkeohKkGp61FFK6d54mJv4qu28tdLhstnsPhFktVphNpsPhQgi0UN0FM1KZ8vlcpiengZjDOPj43JovVRz0nYmlUohFovB7/e3ZTqbGrXoyeVymJychN1ux/j4uKF/HK1WK3p7e9Hb2wvg9kq81KPF7XYX9WghCL3QsiCOxWIIh8O4du0aMpmMXHheS38qEj1EJZQyMlDfG6VUzZs3b+L69euwWCxFqZpGvs/XiiAIB6b5McaKRJGWCJJMESwWS0MXjFsFiR6iY2hU7x01e3t7uHLlCkZGRjAwMFD0XKeJHsm9zOVyYWRkpCNugErRI32WJ06cQH9/f4tHVj3qlXipD8bs7KxsASv1B6ImqYSeKFMxjx8/DlEUsbe3h3A4XNSfSnLfKldzQaKHqARBECoSLOpUzXQ6jXA4jI2NDUSjUdjtdlkEtVuUvBSCIFSVhaElgjKZDNLpNID8fMZqtcqRoE4QQSR6iLZH3XunUV9MKQXqxo0buO++++B2u/e9plNEj9qdbXp6uiOOC7gtetbW1rCysoJ7770XHo+n1cOqG3UfDOUkdG1tTXb1kSahhzHnnWgcJpMJgUAAgUCgqD9VOBzG0tKSXHMRDAbh8/n2XX8keohKUKdwVYrdbkdfXx/6+voAQHbNXF1dRSwWk1sHdHV1we12t+W1WKkgLMVBIkjKnFGnw7UbJHqItqYZvXeAfKj8ypUrcDqdeOCBB0p+2TtB9KRSKUxMTCAYDMrpbNU29DQynHNsbGzA5XIVpSZ2GupJqNKeeGFhoaheqNHOXMThQ92fSlmPNjc3ty/diEQPUQmiKOoStXY6nXA6nRgYGADnXBZBS0tLiMfjcLvd8vXpdDrb4tqsVRCWQimCpN//TCaDTCYDIP8b84UvfAGPPfZYW5wfgEQP0cY0q/dOKBTC1atXK3L0anfRU6oZZ7sfl0Q8HsfS0hL8fj/Onz/fNjdqPVAX/krpHuvr64hGo3A6nQ1z5iIIdc2FOt3IarUik8nItWl0/RFaNKI5qVbrAMm0Y35+vsi0w8j1koIgNCyCL30f1SLos5/9LB577LGG7LMRkOgh2g5lOlsjzQo451hYWMDOzg4uXrxY0Y2uXcWBKIqYn58v2Wy0EyI9m5ubmJ+fx+DgoFykeZhRpntIK51SFCiRSMDr9cqmCO3s1kcYE/X1J00wlSvth9mUgyJf2jRC9Kg5yLRDqpeUmkgb6f7YSNGjRjknaKfrlEQP0VY0q/eO1IAzEAhU1aCyHUWPVjqbmnYWPaIoYmZmBslkEuPj49ja2pILNetBj3NilImNcqVT+pGPRqP7mqRKpgidmhJItAbGGOx2O1wuF86ePSubcoRCIUNPMhuJ9BtHFNMM0aNGy7RDEkFXr16t27lQL5opeiTabV5Av1xE29CsdLatrS3Mzs7W1ICz3URPqXQ2Ne12XBLJZBITExM4cuQITp8+baj6JGkcRpzYMMbg8/ng8/kwMjIiN0kNhUJyUbrUn+Ww9MAgGovyu6A05ZAmmdFoVG4eLIlwyZTDarW2ePT604rJfTvQiom9GpPJJN8fh4eHi0xjNjY2WnZ91mtkUC2ZTKbtXEFJ9BCGp1m9d0RRxOzsLOLxeM0NONtFHJRLZ1PDGGuL41Iiidd77rkHXV1d8uNGET3thFaT1HA4jM3NTczOzsr2r8FgkOoxiJo4aAHAZDLB7/fD7/cXiXCpRxXnXF5l9/v9LZ8U64FRF0RajRHFoNI0BsiLD0kESdenUgQ1KlKut5FBOaQ01HaCRA9haJrVeyeRSGBiYgJHjx7FqVOnat5PO4ieStLZ1JhMprYRCmq7baPWJxllHLVgtVqLemConY+kot9gMAiHw9Hi0RLtAOe84smsWoTncjmEw2Fsb29jfn4eZrNZvv7atRGlESf3RqAdzot0/UmLbblcThbpSvt26Z9eQqXZ50YyHWknSPQQhqRZvXcA4MaNG1hcXMQ999wjr9TUitEnspWms6kx+nFJpNNpTExMyLVYnVafZFTU9q9Svvu1a9eQyWTkVc6urq6OTEWqFboOb1NPZMNisRQ5w2UymbZvRFmNCDxMtIPoUWOxWIrs25XtAxYXF+VIUVdXl2YPq2po5rUtLXC1EyR6CMOhTmdr1JdYEARcvXoVuVwOY2NjukzGjPpjWm06m5p2iGBJ1uJ33323PPnRwiiixyjj0Butol9plXN1dVVORUqlUm23Skg0Dj0L9202G44ePYqjR48CuB2JXFlZkVenjd6DhYwMtGlH0aNG3T5AShfe2tqSe1hJjXyriVQ2+/eE0tsIok6alc4WjUZx5coVDA0NYWhoqKN/XGpJZ1Nj5Ak65xxLS0u4detWRdbiRjkWo4yj0ZhMpn2pHuFwGOFwGAsLC9jY2JCtiQ9bk9TD8PlXSiNrWNSRyHg8jnA4jLm5OaRSqaJ0TKM4w3XC5L4RNLtYvxmo04XT6TQikQhu3LiBmZkZ2Gy2oka+RrlHUqSHIGqkmb131tbWsLa2hnPnzsHr9TZkP0ah1nQ2NUaN9GSzWUxOTsLpdGJsbKyi6+awiA2jIqUixWIxeDwe+Hw+hEIhrK2tIRqNwuVyFfVnMcoPfKPo9OOrlGYV7iud4Y4dOybbD0v27Nls1hDpmGRkoM1hEIN2u70oUplKpRAOh7G2toZYLAaHwyGbIijTNZt9vVBND0HUAOcc2WwWgiA0NLqTzWYxNTUFi8WC8fHxjnD4KYUoipibm8Pu7m7NTnRKjCgUdnd3ceXKFdx5553o6+ur+H1GORajjKPV2O129Pf3o7+/X+6EHgqF5FV4ZZPUdrNHJSqnVZN8pf3wyMiIZjqmVG+hZ9F5OQ7D5L5WDtt5cTgcRfdISQRJ6ZoulwuBQACiKDb1e0SRHoKokmb13olEIpiensaJEyfQ39/fkH0YBT3S2dQYKdLDOcfq6irW19dx4cKFqleajCI2aBV3P8pO6NIqvNQkdX19HYIgyLnuzZyANgpazb+NUc6FVjqmVHS+sLAAs9ksX4ON7FFFRgaEFoyxfemaiUQC29vbyGQy+O53v9u0mrVEIkGihyAqoVm9d6R6j5s3b+LChQtwuVwN2Y9R0CudTY1RhEIul8P09DRMJlPN0Tq9jkWPHxIjnFMjo+zPcuLECQiCsM/1qN2tiYk8RhE9atRF55lMBpFIRO5R1ah6CzIyICpBWigym82IRCK49957S9asdXV16dpCIBaLySl47QKJHqLpcM6RyWQablaQyWQwOTkJt9uN8fHxpk6Imv0Drnc6mxrGGARB0HWb1RKLxTAxMYHjx49jaGio5u0YRfQYRUiqYeltsOQyuHMY3N5jqO2azeYi61e1NbGU6x4MBuFyuQw/aTTqRL8VtMu5sNlsRUXnynoLqSZNmmDWcw1SehtRDVJjUnXNmrKFwMzMDNLpNLxer3yN1jNXoPQ2gihDLpfD4uJiwzu37+zs4Nq1a2XtixuBNJlt1g94I9LZ1LS6OenGxgaWlpZw/vz5us0n9BIb9W7DiBM8y/qX4XjhCcBkBcQsUheeQW7w4YZsF7hU93aV1sScc9maeHFxEfF4XP5xN5IrF6FNu4geNep6i0QiITsTSuk/Uk1aNavs7Xo+iNYgCIJm5oNWC4FoNIpwOIzp6Wnkcjn4fD65Zq2aukkSPQRRAmU6WywWa5jtorIfzaVLl1rSDV6qf2nGKl2j0tnUtKqmRxRFXL16FdlsFuPj47BY6r9lGSnCYpRxAPlIjOOFJ8DEJCAmAQCOFx5HvOeldUV8Sm3XfPdXAej3g8kYg8vlgsvlwuDgoLzCKbly5XK5IlcuPa6leqGJ7W064Vwoa9KGhoaKrkGpUa80wSxnzEGRnv0Y6X5pNEqJHjXKlGGlcUckEsHa2hoEQZDvk4FA4ED3QqrpIQgN1L13zGZzQ1KlkskkJicn0d3d3bCIRyU0QyA0Op1NTSuEQiKRwMTEBPr6+jA8PKzb52kU0WOUcUiw5HIhEpO8/aDJmk9Jq0f0lNiuLbMB4HjtAy63X8UK5/DwMARBkF25lpeXwRiTJ59+v58mmC2mE0SPGvU1KIoi9vb2EA6Hi4w5pAmmUoiTkcF+OvEa0YtKRY8apXGHVDcp3SdXVlaK3Av9fn/RNUrNSQlCgbL3DnDbZrIRouDWrVu4fv06zpw5g2AwqOu2q6XRokdKZ2umuGv2BF36PM+ePYtAIKDrtvU8lk76EebOYUDMFj8oZvOPN2C7GdsADm4jqy9msxnBYFC+P2SzWUQiEflas9lschqSsvdFI+mk66deDsO5MJlMCAQCCAQCRcYc4XAYS0tLRUI8l8uR6FFB0a/S6NW0VX2fzOVysgj62Mc+hn/4h3/AAw88gJe97GU1p7eNjIzA6/XCbDbDYrHg2WefLXqec453vvOd+OpXvwqXy4U///M/x8WLF+s+NoBED9EgDuq9o6coEEURMzMzSCaTGBsbM0Qfj0aKnq2tLczOzjY8nU1Ns9LbRFHE9evXEYvFGvZ5GiXCYpRxSHB7D1IXnoHjhceLam/qNTMotV0h09rFCavVit7eXrnmT937wu12FzVJJRrLYYxsqI05stkswuEwbt26he3tbVgsFmSzWdmdsNNFYTlI9JRGMjLQG4vFIl+j73//+/H444/jH//xH/G3f/u3+O53v4u3vOUt+NEf/VE89NBDeOCBByrOOvna174mOyKq+bu/+ztcv34d169fx3e+8x08+uij+M53vqPP8eiyFYJQUK73jl7pbfF4HJOTk+jv78fp06cN84PQCIHQ7HQ2Nc2YoCsjWBcvXmzY52mU68Qo41CSG3wY8Z6X6u7eprndxUVDiT51Qbpk+zo7O4t0Og2fzyf3B9JTjBvxOmgFhyHSUw6r1So7w0n9VaxWq+wM53Q65UhQI42AjEqtKVyHgWadm97eXrz+9a/H61//erzyla/EF77wBTz33HP44he/iHe9613o7u7GQw89hJe97GUYHR2tqXbyK1/5Ct70pjeBMYYf+IEfQCQSwY0bN3TpsUiih9CNSnvvmEymukWP5OZ19uxZ+P3+uralN3qLnlaks6lpdKRHcttrRgSLMWaoRquV0Cgbac0x2Xtq2ke5Maq3a+QJm9r2VVmLsba2BlEUi2oxap1sGEn0tRoSPcWIogiXy4UjR46gr6+vyJ1waWlJTi1SNqHsdCjSUxpBEJqe6ZJIJHD8+HHccccd+Nmf/VkA+bnZ1772NXz2s5/F3/zN3+DDH/7wvvcxxvAjP/IjYIzhHe94B97+9rcXPb++vo5jx47Jfw8NDWF9fZ1ED2Ecqum9Yzab5Tqfasnlcrh69So457q5eemNngKhVelsahoV6eGcY2FhATs7O01z2zPKxKrSc9ooG2k9aYcx1oO6FiOXy8lNUhcWFory4KtNQzLK9dhqSPQUo073K+VOqIxGKi3ajZDqrTckekrTiigY53zfPgcGBvCGN7wBb3jDG0q+7xvf+AYGBwdx69YtvOIVr8Dp06fxkpe8pNHDBUCih9AByaygVDqbmlojPXt7e7hy5QqGh4cxMDBg2B9IPUSPlM62t7fXknQ2NY2I9EjNYz0eD0ZHR5v2Y2akSE85GmUjrSeaY3z+UcR958G9p1o8usZgsVjQ09Mj56Sn02nZkUudhnRQg0qK9NyGRE8x0gJiKUr1XwmFQrhy5UqR9bBRLNrrhURPaZp9bur5vg4ODgIAjhw5gp/+6Z/Gd7/73SLRMzg4iNXVVfnvtbU1+T310v7fAqJlVJrOpsZsNlc16eScY3V1Fevr67j33nsN7wtfr0BQprNdunTJEBMBvSM9kUgEU1NTOHnypNzZvFnodSz1fi6VjKNRNtJ6IKWzIRPZP0aehvtfH0Tqwic6KuJTCrvdjr6+vqI0JCkKlEgk4PV6ZVME5QIGiZ7bkOgpptpJrLL/itp6eHl5GQCKLNrbsTaGRE9pWhXpqfY7G4/HIYoivF4v4vE4/uEf/gHve9/7il7z6le/Gh//+Mfxute9Dt/5znfg9/t1SW0DSPQQNaLuvVPNhV+NKMhms7hy5QrsdjvGx8fb4kZdj+iR0tmMYL2tRK9ID+ccKysruHHjBu6//364XC4dRlcdRnFNq2QcjbKRrgatWp3idLYMIKaL3wMAYtpwUalmoExDkhpUSivwyiapwWAQDoeDJvoFSPQUU6+bXSmL9u3tbczPz8NsNsupcF6vty3EhF62zJ1Is0VPrb+hN2/exE//9E8DyJcrSIYIn/zkJwEAjzzyCF71qlfhq1/9Ku666y64XC78l//yX3QbN4keoirUvXeqFTxA5e5t4XAY09PTuPPOO9HX11fTeFtBLQLBaOlsavQQCrlcDleuXIHVasXY2FjLBKxRRE8lNMpGulIsS5+D48qT+X1zIW813fPSfelsJc+mQaJSrYQxBp/PB5/Ph5GREXkFPhQKIRQKIZ1OY2FhAcFgED6f79BO6kj0FFMuva1a1BbtUkrmxsYGotEo7Ha7HAlqVp+qaqFIT2maLXrS6XRNNbh33HEHLl++vO/xRx55RP4/YwzPPPNMXeMrBYkeomIO6r1TDeVEgVTcvr29jYsXL7adK021oseI6Wxq6o30RKNRTE5OYmRkBAMDAzqOrHqMInoqHUejbKTLYV36HOyT78pHbYR8JMfxwuNIjP3FvnS2kldsk6NS7YByBT6RSGBubg4ejwebm5uYnZ2VJ5/BYPBQ2RKT6Cmm0X2LlCmZAGRnOGWfKqUznBE+m0b1oukEmi164vE43G530/anFyR6iIoo13unGg6K9KTTaUxOTsLn82FsbKwtV3WqEQhGTWdTU49QWF9fx/LysmHqsfQSPfF4HMlkEn6/v6brtJpx1GojXSssvQ371JP7xQwzF1LXDnZf5ABgcjQ1KtWumEwmuTcLgJK2xFI6XKdCoqcYvSM95XA6nXA6nRgYGCjqUzU3N4dkMlmyLq2ZUKSnNM0+N5IwbjdI9BAHokxnq8as4CBKiYLt7W3MzMzg1KlTJTv1tgOViB6jp7OpqUUoCIKAq1evQhAEw9qL18rGxgYWFxfhdrtx/fp1OBwOefXeKKui9cCSywCzASiu1YGQhOgcQursB+GQokAqOID0ne9G7s4nSPCUQWuir558SrbE165dQyaTKXLkslqtLRq5/pDoKaaVE3ytPlWxWEyuS8tmsy25Dkn0lKYVkR4jLGJWS+fMQgjdqab3TjWoIz2iKOL69euIRqNtIQDKYTKZDuxD1A7pbGqqTW9LJBK4fPkyBgcHcezYMUMdYz1jEUVRnnyOjo7KEzXJrWtubg6pVEpeFQ0GgyUnBEZJs9OCO4cBrnUNC3D/64NIn3wPYPYCQvT2ewCAWZE+9zSyI29p1lA7Gi1bYsmRa3V1FZzztnfkkiDRU0yj09uqwWQy7atLk5r1StehHs16yyGKYkctnulJs6+XRCJBkR6ic9AznU2NcgKdSCQwOTmJ3t7ethEA5ThIILRLOpsak8lU8QT95s2bmJubw7lz5+D3+xs8suaRSqVw+fJlHD16FGfOnJEdDNVuXcp+GWtra/LENBgM1pwK12y4vQfpk78K+8zvFUVzJFc2+8yHAJNqYmOyIf6Sf6u4N48ttQBPfAbM+bKO7edTjmon+iaTSRY5QN4cJBwOy45cFotFTkGqtklqqyHRU0yz09uqQXJ+U16H6ma90vN6mnO0wpa5XWj2AhqltxEdQa29d6pBivRsbm5ifn4e99xzj3zz7AS0RE+7pbOpqaShpyiKmJ2dRSKRwPj4eEel3uzs7ODatWsViVV1vwxpYnrr1i1cv34dNpsNwWBQtnw3KrnhX4B99mmApzSezSJz4pdgW/g4wDP5h0QO894kchUIGPvkr+LupU/n/1gAsiNvR/r80/oN/pBgsVj2OXJJYjsajcLlcskiyOhplyR6immnVC51s95MJoNwOCybc9hsNnnhpx5nuHY6J51OLBaj9Daivamn9061+0kkErhx40bHTY6B/aKnHdPZ1JQbsxQF6e3txalTp9ryGLXgnGNxcRHb29u4dOlSTYXk6olpKpVCKBSS05Ru3rwpp8IZSQxzew9S9/8pHN9/BEBmX/0Ot3gAzhSPZ+F44VHEfecPjNyw6AysS58u2p516dPIjPzioYv46D3Rt9vt6O/vR39/PzjnSCQScjG6Mu2yq6sLNptNt/3qAYmeYtr5fNhsNhw9ehRHjx4FkL/nKZ3hXC6XHAlyuVwVHyeJHuNANT1E26JH751KicVimJychNlsxoULF9r2pn4QStHTruls1SAZUHTaMWazWUxOTsLlcmF0dFS3H1uHw4GBgQF5EupwOIoaV0q58V1dXS1P5cgNPoy47zzcX/8BAMWOi/brH8Y+owMxDfe/PIjU/Z9AbvBhzW2aI8+WfLySKBFRGYwxuN1uuN3ufWmX6+vrEAQBgUAAwWCwoXUYldLOk/xG0EkTfIfDoSnGFxYWkEgk4PV65XveQQtLnXRO9KQV3x0SPURbok5na9QXh3OO9fV1rKys4Pz585icnOzYHziTyQRBEDA7O9u26WyVwDnH/Pw8wuFwxx2j1FfojjvuKNkYt97rV3q/VKg+PDwMQRAQiURky2KTySRHgVpVo8G9p5C6+Bk4nn8MYCZATMi1PWoYAPA0HC88jnjPSzXd24TAqOZ+lI+z9HbT+xK1gmbm4avTLqVrLRQKYXFxUa4Xkq61Zk8uSfTspxPPh1qMc84RjUaLHAp9Pp8sxpURSRI92rTivMTjcQwODjZ1n3pAoucQ06x0tlwuh6mpKZhMpo6zLtYim83i1q1bOH78eNums5Ujk8lgYmICPp8Po6OjHXWMGxsbWFpaanhfIa1zZjab0d3dje7ubgD586xVoyFZY9c9hgrFhdQg1bzx13BceU/5DZus+e1qbJN7TyE78nZYpZqePSBn/Skg3Q14Acv6l+F44YlCA9QsUheeKRk1qhYjiqlWfXe0rrVwOIyNjQ1Eo1E4HA5ZBFWTglQrJHoOJ4wx2RlueHi4yKFwbW1Njkh2dXUhm82S6NFAEISmnxdybyPahkb03inF7u4upqamMDIygoGBgYbtxyhsbW3h2rVr8Hg8uPPOO1s9nIYQDocxPT2Nu+++W65T6QSUdtTNEuflVvptNpvcNV1KCwmFQpidnUU6na6rV0a14oLbe8DdJyvbuJjN216XIH3+aSzZfgxH/tcX4f/Qf4fF9s+wZM8i/ZEPwe57CkxMAmISAA6MGlVDI8VUPRhloq+sw+Ccy01SFxcXEY/H5RSkRtWekeghAG2HQkkERSIRTE9Po7u7W7ZpJxGU/+1qdnpqLBaD1+tt6j71gETPIYNzjmw2C0EQGp7Otry8jM3NTdx3331tuSJQDcpeQ+fOncPKykqrh6Q7nHMsLS1hc3MTFy9e1CXSYBTUdtSVfi9Yehum9GpNkYNq+/Qo00KkhoHKni0Ainq2HDQZYOltOF54ompxIfrvyzctlRzbVOR79TiQuvBM2fMhJALwfehvwFJpIJVPlbO/8z3Ax6yAS/HCA6JGlVLr8TYao/ZpUtqwDw4Oyk1SlbVnSsGtxwIBiR5CC4vFIkckY7EYTp48iXg8LrthWq3WorTMw3gNtcLKmyI9hOFpZO8dJZlMRi4AHx8f15x8SRbIbblKk94Ciy+Du4cBe688Ye7p6cGlS5eQTCYNO5mplWw2i2QyKdtRt+XnVgLJjrpa63Tz6l/B8f1Ha44c1NucVL0ims1mi6yx7Xa7nAqnTk9iyeXCuJOKDZYXF3lHt0/C8cJjADMDYiavdCxOQMwgffJXkRv+hYqEhG1jA1w9WbZagZtZ4ITisTJRo0qo9XgbCdvehnVqCtYaHAGbjbJJqlR7tre3h1AohOXlZTDGKhbcpSDRQ5RDFEXY7Xa43W4cOXIEwG1nOCkF2Ol0ytei2+0+FNdUK0QPGRkQhqUZvXckQqEQrl69ipMnT8o3JS0kh7N2mzybVr8Ey3OPAMwK8CxCJ/8QlyNnipzLDmpO2o7s7e3hypUrsFqtuOeee1o9HN3gnGNhYQE7OzvVGzGkt2B9/lFDRQ6sViuOHDkif++SyaTcLFBySJKtsZ3DgJgt3kCF4kKq75FqYwDUVCeTGRgAy+VUGxeQeujDcKy8t0hM1ntOeR3H2wgsX/4yHE88AZfFgt5MBulPfAK5h1ufalcp6uaU2WwWkUhkXy+qrq6uivuykOghyqE1Z1A7w0lpmUtLS/LEXLpWOyk7QUmrRA+ltxGGg3OOTCbTcLMCyckrFApV1M9EalDaVqYG6S1YnnsETEgCyE90AzO/grGXT8Pmu23V3CmiR3LcW11dxb333ouJiYlWD0k3OOd4/vnna7ajZvHlvPBF7ZGDeiM95XA6nRgcHJTTkyS74itXrkAURYz0/RaGN38HzGQFxFxV4oLbe4peW4soEYJB3PzAB9D3G7+Rj/Bks0g98wxy5x9G/O5X62o4wO09SF14Bo4XHtdVTNUC296G44knwJJJuVeR4/HHEX/pS8F7jGGuUC1Wq3VfLyplXxa3213UJFULEj1EOTjnB6ftlkjLDIfDch2kkXtV1UorjAwo0kMYDim60+h0Nqn5ZldXF8bGxiraTzsKA62Jrslshz13Axy3rRvb8djUCIKAqakpAMD4+Li8itQJE5NoNIp4PI4777yzpB11Obh7GODGiRyUQ+mQNDIyglwuh0hkBFdc40iHZ5CzDcGbuwPBaLRoZZ5tb4MtL4MPD+s+IWeMIfbjP474T/3Uvn2oRZUeqCNULUtrW17Oi7ykQjBbrflz0KaiR4169T0ejxdNPLUsiTvh3kI0lmoXiZRpmcePH9fsVaV3bVoraIWRAYkewjA0M51NSmc4ffq0bH1aCVKkp53g7mFwUdWZnmfzE2AF7S56pAayx44dw+Dg4O0JcCEy0c4Tk/X1dSwvL8PpdMrdwmvC3ovs/Z+AVVXTU81EutGRnoOwWCzo6elBT08PgHGk02mEQqGilfnhb34T/b/1W8VRmAakYPGenqZN9hshpqoew/AwkFUJ5mw2/3gHwhiDx+OBx+ORDTj29vbkOgxRFBEIBOT2Ca1ukmoEOq0m1Cho9ara3d0tqk2T7LH9fn/bXIutSG9LJpNtmS5IoqfDaFbvHVEUMTs7i3g8jrGxsarDxO0mDERRxPWlEOxdT+Lu8B/kJ7o8i9ylTwH2YtvmVk5m6+XGjRtYXFzEuXPn4PP5ip5r5+MSRRFXr15FNpvF+Pg4vve979W9TeHYa5Hu+iGYUitNcW+rhUojNXa7vTgvfmUFvb/5m2CplByRsD/2GNIPPghzjdGxTqXaaBjv6UHqmWfgePzxvJFDJoP0M890TJSnHCaTCYFAAIFAACdOnChEHSO4ceMGXnjhBVgslkPvxtWO9a7tiNlsluscgdu1advb25ifn5evxa6urpY07K2UVogeAIY9HwdBoqdDUPbeARp7MSYSCUxMTKCvrw+nTp2q6UepnSI9Uvped3c3hv+fp5DJvK3IvU1NO/5ISz1q0uk0xsbGNHu+SEK1XVa/JJLJJCYmJnD06FEMDw/Ln48eUStu74HoqL5XEUtvwxGfRNY2CKAxXa2lYvlqIzWMMXi2t8FsNiCVkh/nFgvm/umfECuYdkiT0nb84dOLWs9x7uGHEX/pSxGdnMS2x4ORsbEmjNaYSFFHh8OBsbExpNNphMNhrK+v73PjakaTVCPQ7hH1dkVdmyZdi1LDXrvdLgtyIznDNbs+up2vTxI9HUCzeu8AtyMBZ8+ehd/vr3k77RLp2drawuzsbJE7G+y94Bpip11RioKDetS0Y6Rne3sbMzMz++yoTSZT61LLCk0ynTCD8RzS1j/VvUmmslheitRUUyyvlYJlEgScfuUrkfH7iyYCDodDFkFOp7Ntfwyrpe5z3NOD7JlB2LavgaW3W552ZxTsdntRQ95SLoRdXV0NaZJqBCjSo02z7y3KaxHAPmc4t9td5AzXqntfKyI97Sp8SPS0Oc3qvSMIAq5evYpcLqdLt3qjR3qUzUZrSd9rFyRRV0mPmnYRqsBtO+pQKFTSjroVokfZJFP6iWqE1XW9xfLKFCxlFIP39MAKyNbYyknp3NwcUqlUR7ojaVHvObasfxlDLzwODgtMy0LVfZ4OA0o3rqGhoSIXQmWTVMkUoV0L0dWUcyk7jBjht8fpdMLpdGJgYKDIoGNubg7JZBI+n08WQc0U5M3OwGi3xU8lnXGHOIQ006wgGo1icnISx48fLypsrwcjT6DVzUbbcTWjHJxzXL9+HXt7exWLunaJ9GSzWUxOTsLtduPSpUua3w29Ij2Vrnax9DZYchksE2lKk0w9iuWlFKyD6lXUk1K1O5JUpB4MBtuqMLiSOp16zvFt8Xs7fbDVfZ7aAbULobIQfWlpSW6SGgwG4fP52lY4SDW5xG2MllqtZdARjUYRDocxPT2NbDZbJMi1Usb1otmRnmQyCZfL1bT96QmJnjakmb131tbWsLa2hnvvvVdXe0Kjih7NdLYOI51Oyxbj1Yg6o35mSqRGqnfccUdZO2o9RE8l505KZ4PJCggZAKpz2ACr64MiNdVup5r3qN2RpCL17e1tzM3NFTWtNKqArrROp55zzJLLTRG/nY5WIXo4HMbm5iZmZ2cNW4NRDkpv24/Rz4ny3qcU5FK/Ks657AwXCAR0FSnNFj3xeJxED9EcJLOCRqezZbNZTE1NwWq1FvVp0QujpbcdlnS2UCiEq1ev4tSpUwW74soxeqRnfX0dKysrFQl0PSI9lUZ4pHQ2aYLLYQU3OSDCkq/paVCTzEoiNY2m2Bo7H0WV7GEjkQicTic45wgGg4aoz6i2TqfWc8ydw4DYPn2e2gWr1SqnXgL7azA8Ho8sgso10G4l7Vov0UiMLnrUqAW5tAAk1aeZzWY5Fa7eqGSzm5PGYrG27NEDkOhpG5qZzhaJRDA1NVVX88ZymEwm5HK5hmy7WqRC/t7eXl3T2Yz0w8U5x+LiIra2tnDp0qWafvCNKnokO+pcLoexsbGK8/qbcSyaK/oWJxKXPo9QVEQMPRgavL9h+29mDxwlUjqf2srb4XBgYGAAAwMDWF5elmsSjVKfUUudTi3nmNt7kLrwDOwvPJav6YFQdZ8nojzqGoxYLIZwOIxr164hk8kUNaZsZPpRtbTbBL8ZtPs5US8AZTKZoqikMgqubBBdCc1O/ZNMHNoREj1tQLN673DOsbS0hFu3buH+++9vaPjSbDYjk8k0bPuV0qh0NiPZO0s1Li6XC2NjYzX/cBgxvS2ZTOLy5cvo7+/H8ePHq0rVa4boKbmi778PGYjIJRINH0OzKUrnKzRt1SrQZ4zB6XSir68Pw8PD++ozTCZTUX1GMxYQmtk4NDf4MG6yc8iGr6P/rheR4GkwjDF4vV54vV4cP34coijK6Uerq6vgnMsCqNX1Z2RksJ92Fz1qbDYbjh49KjfJlqKSUoNol8tVsVV7s9PbEokERXoI/VH33mmk4Emn05icnITX661rYlwprZ5ANzqdzSiiZ3d3F1euXNElatfqz0xNKTvqSmmK6Cms6DteeLxIBOQnuLcavv9mo5XOV2mBvjodJJPJIBQKYWNjA9euXYPL5ZJXQhu1IKNXLVSlCJYgUu7zJHhagCSqpXtHLpdDOBwuakwpXW/NbpJKRgb7aVUDzmahjkomEgmEw+Eiq3bpelVnajT73MRiMYr0EPrSzN47Ozs7uHbtGu6++265KVejaWVNT6PS2ZS0sg8MkL9+VldXsb6+jgsXLuhygzJKelsldtTlaOax5AYfRrznpfvSvYxyPvWk2gL9g47fZrMV9WtJJBIIhUK4fv060uk0fD6fPCnVMzXJCLVQRPOxWCz7GlOGQiGsra0hGo0Wie5G92TptKiGHhymc8IYg9vthtvtLrJq10rNDAQC8nuahVQf146Q6DEgzeq9I4oi5ubmsLu7W3OdR620KmrQLHe2VkZFcrkcpqamYDabdTWhMMIkPZvNYmJiAh6Pp6QddSU0+1i4vWffpL8TV3KrKdCv5viVkwDJHnZvbw+hUAirq6sAUGSNXe/kqFm1UOq6v0qssonmYLfb0d/fj/7+/qKV92b0ozJSPahROEyiR43Sqn14eHhfamY8Hsf169dlEdToekgyMiB0QbqxSqHMRn7BpWhHT08PRkdHm36DbXakp9nubIyxloieWCyGiYkJDA8PY3BwUNdttzq9TbKjvvPOO+U86FoxgoAD2rvJmxYHp/Pph8lkytu/5nJg4TAyAwMIWyy4desWrl+/DrvdLk9IjWxVrPz8K7XKJpqPeuVd3Y9KEARZdOthR3yYJ/iloHNyG3Vq5ne+8x0Eg8F9/aqk+jS9z1sikaD0NqI+pN474XAYOzs7OH36dMP2dfPmTczNzdVcC6EHzZxANyOdTU0rBMLGxgaWlpZw/vx5eL1e3bffSqEg2VHfd999HZOqZ4QxNIJS6Xx6oxQJrmwWjmeeQW9BJCSTSXkCEI/Hi1blG22NXcq5ruTrGavaKrvTaLfvgboflSAIiEQiCIfDWFxcLDLhqGUBkyb4+2m2LXM7YTKZ0N3dje7ubgC3+1VJi0BWq1XX+rR4PC7vq90g0WMAlOlsZrO5YZNlQRAwMzODVCrV8l40zYr0tKrZaDNFjyAIuHbtGrLZLMbHxxsW2m6FkBNFEdPT0xAEoSo76nJ0quAwClrpfHpSTiQ4nU4MDg5icHBQzocPhUKyNbaeq/JKKnWu23c8NVhldxLtns5lNpuLJp3SAubGxgai0SgcDocsgso5cQHtfz4agRGMgYyI1u+Yul9VKpVCOByW69OcTmdV16MasqwmakKr947FYmmIGIjH45iYmMDAwADOnDnT8htqoyfQrW422iyBkEgkMDExUbVlcy00WyjUakddCa02mgBIeNVDNSJBmQ8vdUpXNgm0WCxFq/K1Xme1ONdJk9t6rLKrjSwZkU6b5CvtiDnnsh3x4uKiHHmUrjmtyCNFevZD50SbSiJgDoejqD5NioRLznBS017JpKMc0jXcjpDoaRGleu80IgKyvr6O5eVlnDt3Dj6fT9dt10ojRUEr0tnUNEP0SKHrs2fPyg4ujaSZkZ567agrodWCo5Mmec2mnEg4SAioV+XVLl1ut7vIpatSqnWuKzqeGq2ya40sGY1OEz1KGGNwuVxwuVxy5DEWixVFHpVNUi0WC/Xp0UAUxZY0LDY61UbAlNej5AwnNe2dnZ2VnTGl61FrwZgiPUTFlOu9o6foyeVymJ6eBoCGpj3VQqPS2yQh0Mp6JaCxAkGKYsVisaZGsZoRmeCcY35+HuFwuGY76kowSpTFCGNoRw4SCdUKAbVLVzweRygUkicAfr9fFkEH3UOrca6T36OY7FdrlV1PTySj0cmiR42ySarUlFdyIlxeXpazPnw+H0U3FNC50KbeHj1aTXv39vYQDodlk46NjQ2kUim84hWvQDAYrNqyenV1FW9605tw8+ZNMMbw9re/He985zuLXvP1r38dP/mTP4kTJ04AAH7mZ34G73vf+2o+rlIYZxZ8CFCns2nd5PUSA5LTVSNcvPRAb1HQKiFQikaJnlQqhYmJCXR3d+PixYtNnSg02pFOLzvqSjCC6KlnDGRtrC0S6hUCjDF4PB54PB55ArC7u1s0IZXSknw+X9E1qodzXTVW2fVElozGYRI9asxmc5ETVzabxezsLKLRKJ599lnYbDZZdHs8nkN7nkj0aKN3Y1LJGTMQCMgmHdlsFl/84hfxx3/8xzCZTPB4PJicnMSpU6cqanVisVjwR3/0R7h48SKi0SguXbqEV7ziFbjnnnuKXvdDP/RD+Nu//VvdjkVzLA3dOiFTKp1NTb2ih3OOlZUVbGxs6OZ01Qj0vHEr09maLQRK0QjRIzWRPX36dEucU0wmU8PMJ/S0o64EvUVPLSKk1uuUrI1voxYJegsBtTWs5Iq0ubmJ2dnZfQXq1TrXVTPZV19jtUSWjMphFj1qrFYrXC4XPB4Pent75SL0lZUVxGKxmtMv2x1yb9NGb9Gjxmw248UvfjFe/OIXA8ibQ73xjW/E17/+dXzkIx9BIBDAy172MrzsZS/D6OioZjRciqQDgNfrxZkzZ7C+vr5P9DQDEj0NRpnOJoWtD6KeAutMJoOpqSnY7XZdm1IaGaOks6nRU/RwzrGwsICdnZ2mN5FVYjKZ5LRMPVlbW8Pq6mpTRbqeoqceEVLtGA67tXE5eNwDXE8D3QCk8kUdhYDSFUldEJxMJhXW2Od1jTaXusaa0ROpFqo1VyDRU4wyqqEuQo/H4/vqLyQRZLVaWzzyxkHubdo0Wwz29vYil8vhox/9KFwuF27cuIGvfe1r+MxnPoN3vOMdOH78OH7+538eD5f4DVxaWsLzzz+PBx54YN9z3/rWt3DfffdhYGAATz/9NM6ePav7+En0NBDOObLZLARBODC6owfhcBjT09O46667mrJS3mqMls6mRi/Rk8lkMDk5CY/Hg9HR0ZaudOkdHREEAVevXoUoirraUVdC1ceytSWvsqO39/Z26hAhtZzPeqyNOz0lThYGZhN4BsCb7MAIkHrogw0RAuqCYHXDSlEUZWtsv9+vOWGrpGD9oGusWT2RgMqFTC3mCiR6iil1PpTpl8eOHSuqv1hbW6vommtXKL1Nm1aIwXQ6Ldfb9vf34/Wvfz1e//rXg3OOxcVFhMNhzffFYjG85jWvwUc+8pF9ploXL17E8vIyPB4PvvrVr+KnfuqncP36dd3HTqKnQSh77zRS8EhRgO3tbVy8ePFQhLuNmM6mRg/RE4lEMDU1hZMnT8p++61ET9Ej2VEPDAzg2LFjTf8MqzkW05e+BMsjj8ir7LlPfQria1+b306T+6vUam1s1JQ4va6nImEgbfuzacDjhuP334vUM76GH6+6YWUul0MkEsH29jbm5+dhsVgQDAYRDAarqs046BqTnpc+f1PkuYaIn0qFTK01VSR6iql0gq+uv5CuuZ2dHfma08OO3QiQ6NGm0eltpdD6LBhjuOOOOzRfn81m8ZrXvAZveMMb8DM/8zP7nleKoFe96lV47LHHsL29jR6df0dJ9OiMVu+dRpFKpTA5OYlAIICxsbFDcUMwajqbmnpSwaS6rBs3buD++++Hy+XSeXS1oVf0SmoY2yyrbS0qFj1bW7A88kjRKrvlHe9A5qGHgJ6e+vqr1CAia7E2NmpKXKkJWE31URrCgAFALA6gNcdrsVjQ09Mj/2irazM8Hg8452X7XZS6xsyXL8P+qlfljzuTAt7GgR9y6m5dXY2QqbWmikRPMbVaVquvuXQ6LbtwKZtSdnV11dSUspWQ6NGm2aKnloUqzjne+ta34syZM3j3u9+t+ZrNzU0cPXoUjDF897vfhSiKDaldJtGjI5xzZDKZsmYFlW7roPdLE8dWFbXrRaU/dkZPZ1NTq0DI5XK4cuUKbDYbxsfHDXWTrzfSo7SjbvVnWOmxaK6ym0xgL7wA/vKX19xfpZoxqKna2rjJ0ah6qDUilRcGmdIvMMDxqmszYrEYFhcXsbm5iZs3b8ppSYFAoCjVU/Ma++AH4Xjve4uELD4N4GwW8OlrXV2NkKnVXIFETzHSHKJe7HY7+vr60NfXp9mU8nYNWlfD2gPoBYkebVoV6anm+vy3f/s3fOELX8D58+dx4cIFAMAHPvABrKysAAAeeeQR/PVf/zU+8YlPwGKxwOl04otf/GJD7gkkenRCiu7okc4mTZi1LmRp8h+NRhvax6QZSE515Wo52iGdTU0toicajWJychIjIyMYGBho0Mhqp55Ij1Sb5PV6MTo62vLPsFLBobnKHo/D+rM/i9hHP4pr990H34tehO5nn4Xr1q2m1ctUY21cTzSqmdQTkeI9PUj/wa/C/iu/B2YCkFK9wGDHK/XG6OrqQm9vL44cOSJbYy8tLcmucVJaUu7hhxF/0XmY556FcNco2GZsv5A1A9hC3sBBR+vqaoRMrbbdJHqKacQEX6sppVSDpmySqiW8jUCrJvdGRxCEpi4g1iLIH3zwwbK/t0888QSeeOKJeoZWEca6qtuQRqSzSWJA/QVPJBKYnJzEkSNHcOnSpbb/kahkEt0u6WxqqhUI6+vrWF5exr333ltV069mUmtkYnd3F1euXDGUyUbFx9Lbi9ynPgXL298OpFJgKKRNpVJw/dIv4eg3voG4KGImFEJaEBAIhRAEKpo0NKtXUD3RqGZSb0Qq97pfgN32NHAzBSwB+G8ANwOAA+mPfAjMsgykYQh3Mwlpsm82m+V6HyC/SBAOh7GxsYFoNIoh4f/i5M6H82LmhRxSxz+4X8gKACSPDR0d66oVMrWYK5DoKaYZ54MxBp/PB5/Ph5GREQiCUCS8D+pJ1Qoo0qNNs40MksmkYVuhVAKJnjqotPdOtWj16tnc3MT8/HxL6yD05qCeRO2WzqamUtEjOZgJgoDx8XHDra4pqSXSI9lRX7hwwVA3ymoEh/ja1yLb1QXr614HxOPy4ya7HT2xGIKnTslOSurV+l4APbEYnGfOFLm+NZtqU+JaQb0RKW7vQeolf5qfnJ+yAg9kkO76NfDBbjhWngK+VbmbWKux2Ww4evRofpEgtQXP//lxMJ4ChHwIy7b8JLZ+9zfR+1sfKK7pCTorjq5UQ9X9h+w91TVlJdFTRCsm+Grhre5JZbfbZRHkdrub/nnRNaJNsyNgsVjMMHXGtWDcGZaBUfbeAbRdLOpBKQYEQcC1a9eQzWYxPj7eUT78pSbR7ZjOpqYSgRCPxzExMYHBwcGWOJhVSzVCQWlHbcSeUdVGWfiFC4D681RNyNWNLPGXfwn3O98J0WwGy+Uw/+u/Dv5zP4dgMAiHw9G0SI98DFWkxLUCPSJS6sk5ALj/6WzVbmLNopKJnCm1AphsgHg7Z4+ZbIj84Alc/4u/gP3GDdhPnULXcBcClgjgHmnIsVUrZKraNk1oi6jVyEBPlD2pgPzvcjgcxtLSEuLxODwejyyCmtU7jq6R/TRb9EiffbtCoqdKmtF7RxI9sVgMk5OTbTMprhYtYdCu6WxqyokeKXJ37tw5+P3+Jo6sdiqN9CQSCUxMTLTMjroSqhUcUYcDN9/9bpz5oz8Cs9mAbBbZT36y5IScbW/D/a53gaVSkKYud33wg5h96CFcu3UL2WwWHo8H6XS643PVS/V3MWdDsOdugKVt8uN6RKSUk3NT5Lma3MSMhFZNDeM5DNz1YvSf7UEul8uvyIdCmNkVYbOtIRhMtGxFvhZI9BSjl5GBnjidTjidTgwMDMhGHOFwGDMzM0in0/D7/fKiTyctzhqdZjcnlZwn2xUSPVXQrN47JpMJm5ub2NnZwfnz58tamrYryohWu6ezqSklEERRxMzMDJLJZNtF7ioRCkawo64ExljFqXo3btzA4uIi7v3lX0b2kUeKJ+QlbMnZ8jKgSlVkNhuOCQIGL12CIAi4desWIpEIvv/979fcw8XolOrvYln/Mu6aehycWWCaFYpSzvSMSNXqJtYsKpnsl6upsVgseTMEHwNLxpBg3QjFzVh7/nnwxUWY77wT3jvuQDAYNKzxDYmeYoxevyIZcXi9Xhw/flxO7Q2Hw1hdXQXnXBZAndYk1Wg0u6YnkUhQelun08zeO7lcDjs7O3A4HIav8agXk8kEQRA6Ip1NjdakWjrOI0eO4PTp0213nAdFeoxkR10JlQg4URQxOzsrC1SLxQJ4POBSbQ7ntxdAVL1lTJcvA9Fo8QYV6XBmsxldXV3Y2trCvffei3Q6jVAoVNTDpbu7G8Fg0PDnshQl+7v4zucf5ymg8BE0KuWsVjcxo1Gupsay/mU4nn8cMJnhEgX4198E6/u+AFit4JkM1n77tzH9Az+AXC5XZI1tlMkoiZ5i2u18qFN7peijujFvV1dX2zdJNRqtqOmhSE8Ho2fvnXLs7u5iamoKXq8XfX19HS14gPzELxQK4dq1a22fzqbGZDIVTaqlCMiZM2fkQtF2o5RQaJgddXoLLL4M7h4G7PqaAJQTPZlMBpcvX0YwGMSpU6cOPCZ1b5n0hz4E+3vfC+U7OID0hz5UFMFQbtNut+/r4bKzs4OpqSnkcjk5d76dVk1L9XcxR57NPx5J5i2WewEE9Us5UwvQWtzEtLbTCKqZ3JaqqWHpbTi+/wgYsnkHtz3A+hufBssASCbBABx7//sRnJpCrqsLkUhE7tWiLF5v5WS03Sb5jabdz4cUfewtLBBJizpra2uIRqNwuVyyCHI6nW19rK2Ganqqo7Nn1XUimRU0Op2Nc47l5WVsbm7iwoULuHXrVklXs05BFEWEQiGIotgWkYFqkaIinHPMzc0hEom0fV8lLaEg2VGfPHlSLngtSRUixrT6JVieewRgVoBnkbv0KYjHXlvvIcgcJHqkY7r77rvlH+2S29HoLWN/z3vyAkiJxwPh3nv3vV9rDMrUkZGREeRyOUQiEXnV1Gq1IhgMoru7u+Fd1UvV41RCqdQyITAK/N9kvpmmGfmJ+tuT4C8rnXJW6ThKNTettgi/1iaprYDtXgagOM9byJ9XJQXrb3NPD7q7u+WG1ul0GuFweN9kNBgMwul0NusQ2n6S3wg66XyoF3USiQTC4TDm5uaQSqWKmqRqzQWaafjSbjQ7FZJETwfSzHQ2aZXc7XZjfHwcJpPpQCvnTkBK87JYLOjv7+84wQPkRU8ul8Ozzz6LQCBgiIac9aJOb6vGjroqEZPeguXZd4CJKQB5IWF57h3IHHlIt4hPKdEjHdP9999fUd5yqd4y+3uoCPuslys1U7BYLOjp6UFPIdoguSgtLi4iHo/D5/MdOGGolVL1OJVSKrUM6W7gswzIKF78WQa8G4DGmkCl46inuWkjtlMJekz29727F3khqaSE9bfdbkdfXx/6+vrAOUc8Hkc4HMbs7KxcnC6lwjWy/pBEz+GBMQa32w23242hoSGIoig3SV1fX4cgCPtSMI1e49RKmv3dicfjhmo/US0kelQ0qveOFqFQCFevXt23Sm42m5FOpxu231aidGfb29trnxWcg6IUGs/t7e1hb28PFy5cKBstaBeklD1BEDA9PQ3OeWV21OktWJ57BExIohIRY174bJE9LwCAWfPnuEGiRxTFon5JlaQLMMaAkRFNgZP68IfheO97D7RervXeonZRikaj2NnZwfr6OkRRLEqFq3WiULIep8q6G63UMtNzzwE2B5BSqB6bQ7MJaTXjqLe5qd7baRai/z6A2QBeOJ8+AL8I8M8AsDkBAUg98wwAwPTccyXT9Rhj8Hg88Hg8+/pOraysAEDDmlXSpPbwYjKZ4Pf74ff7ceLECQiCgEgkIi/sSM+LokjXiQGIx+OGaTJeCyR6Cqh77zQ6nW1ubg7hcBiXLl3a53HfiZEeqSg8Ho/L6WzxeLwtjvOgKIX6uezFT2JBGMfNmzfhcrk6RvAA+e9ENpvF9773varsqFl8OX9+oJhElhIx6S2Yr314/+q1mMmLSp1Qip5UKoXLly+jr68Px48fr+p7z3t6kPz4x+HUSIWKv/rVZWtC6hX9yq7qJ06ckAuIpcUFh8NRlK5U6bGVqseppe5GnVpWTRPSasZRa3NTdepcvU1Sq0GPVVpu70Hq/k/C8cKjgJgGA8BeDOAcwHdExF/zDZi/OQH32bNF16jwEwfXOamL07WaVUrXVr1plhTpISTMZnNRCmYmk8GtW7eQyWTw7LPPwuFwyOK70em9xH4ova0DaEbvHYlUKoWJiQkEg0GMjY1p7qvTRI/StUxZFG4ymWSRaVgOilIA+54zP/t2ZE9+FePjL8K3v/3tFg26MWxvbyMej2N8fLwqO2ruHga46nPmWU0Rw+LL+xoxcgBi/4/pamYgiR4p2lqPwUTu4YcRf+ihfQKnnPVyI5qTqguI1bnzylS4g9KVGmn1LDUhtT/2GLjFApMglGxCWs04SjU3BUpHOEqlztXbJLXZ5AYfRsLaBdez/wEQ4vkHfQCCdrDU2v50vcceAbImIGCrOHVRq1mlZIiQSCTkuoxaHAdJ9BClsNlsCAaDiEQiOHv27L70Xq/XK4ugdq6ZrZVmf28SiQSlt7Uzzeq9A9xO7So3weok0XNQs9F2OM6DohTS/5XPMZMNpwbt4B0UgleaMbhcrur779h7kbv0KViee0dRtExLxHD3MCBmih5jAEw3/heQ3qpO+GxtyUIEGhE3KYVCK9paLXr2ltETl8sFl8uFwcFBiKKIvb09hEIhrK6uAsinK3V3d8Pr9RaljTTa6jn38MNYO3UK1vV19IyOlo6EVTkOdXNT89e+ti/CIRkSHJQ6p0eT1ErQU/Ry/30AV1nKi1mwLexP12MZsJsAPPnFhVpSF51OJwYHBzE4OFiUZnnlyhUIglCV4yCJHuIgpLQ2xljRPU1yugyFQpienkYulytqktrpDriV9prTE7KsblOU6WyNNiuQGlImEomKnMraQQyUQyudTc1BfV+MQtkoheo5hpyuaVitJpPJYGJiAn6/H6Ojo/jWt75V03bEY69F5shD5d3b7L0QTj8J8/RvF6e4mWxV1fSYvvQlWB55RJ7s5j71KYivzackCoKAxcVFZLNZvOhFL2pZjngjIj0HYTKZEAgEZNEqpSvduHEDMzMzcDqdRalwtVo9V4oQDEIIBg+OhqW3IbpPIP6SfwUTYhWNQxKg5QwJyqXONUvI6jXZLyUQRft9QEpVI5dF3vBAosbURQmtNEul42C55rucc6rVIEpSqpZH6XQ5PDwMQRDkhZ3l5WUwxoqapHbaNdbsxqQApbe1Jc3svROPxzE5OYm+vr6KG1K2u+gplc6mpi2O84AoRS6Xw1rPr2Pk1u/BZLYfGMFoR6qyo64Ee29FokW4420wX/twsZlBiXQ4Tba2YHnkkaLJruUd70DmoYeQcLvl/jutKoqV+78cO9ZSIw9lupJkIxsKhfY5d3V13deSFVPN1LPApYrff5AhAbwAMpF9UUV16lw9lt2tQEuosu1tQH2dqS87nVIXJdSOg+rmu263WxZBDoeDIj0K2sbcp4kIglDRvVpq+qysQ4tEInLGiZQq19XVpSm+241m9+gBIKcUtiuHTvQ0M51tY2MDS0tLOHv2LPx+f8XvawsxUIKD0tnUtEOkB9COUsRiMUxOTuLYyL9HbuzNDWui2SpWV1extrZWsXWzrth7kRv9dEXpcFqUmuxGXngBUy4Xzp07B1EUcePGjQYdQGnU/V+OvPvdwOho08ehRmkjq3buWl5elovapVS4Rk8W9HCPK2VIYLJdhuOfXlUQUwI4swFmx77UuSLRFckg3fVryI2+RdfoTyMm+2rjCLa8DDidxefC6QTfEYGgve7UxUqEobpPSzwelxtTZzIZmM1m+Hw+5HK5jk9JKgcJwP3UGtGwWq1FNY6pVArhcHif+JaapLYbrRA9VNPTJjSz904ul8PVq1chiiLGx8ervom3o+ipJJ1NTVsdpyJKcePGDSwuLuLcuXPw+XwAoJuVcqup2o66QVScDqdmawsIh4FM8Qo+z2SwyLl8bUYikaavqGqlW51++mkk3/pWw9UDqZ27MplMUUd19Uq93rDkskZ0QoT51v+GcORHK5qgaxkbpD/yIThWnioSUxwOJC99HqL/Pnm7RaLrG0ngM4Dd/Luw4w+ReuZPDduoVAtN8ScA8dd8A8wdA497wDZjYPbtqq/DWno5Ka2xjx8/DkEQMDs7i2QyiRdeeAGMMfnaUteaHQbIlnk/ep0Th8OxT3wr+1JVavRiFCqNgOkJRXragGb23olGo5icnMTx48cxODhY077aSgyg8nQ2Ne0S6ZEQRRHXrl1DOp3G2NhYW9wUqyGRSODy5csYGhrC0NBQ61cbK0yHkyiq4xFFcKsVcDohptNY+0//Cfe9/OXyD0Sz62kA7QgUN5sN2/9Fic1m29fEUlqpz2az8Pv96O7ulpsJlkJO7Stj/8zNHoCr6lB4Go6JXwXw7oobpaoNCZhlGfiWqo7HbAO3BYqjI1K9TyQveJCRmoCmdG1U2oxV/SLxZzbnDR0++EHwwVMwqyKPSqOHcujVy8lsNsOTTMIXCiFw4QLSPh/C4TA2NjYQjUZrtl1vVyjSs59GCEGtvlR7e3sIh8NYW1uDKIpyk9RKzDhaQStqejKZTFu75HW06Gl2753V1VWsr6/j3nvvravQS2oC2Q5Uk86mpp3EXTKZlHu5nDlzpuLrqF1+wKTP8dy5c1WlYrYcyaHN49lXx8MdDlz+jd9A8Id/GH3nzhW9rRWiR2vFnQmCLACUgqDZIqia2hWtlXopFW5hYaFk0bo6tc//5JNI33MPmN2+vympEANMziJxwgBAzNsxO55/FHHfecDWXXbcRYYEaVRkgS1bZW8BUM8pDNyotBS5hx9Gem8P9iefBGw2ON77XqQAON773pJGD+XQq5eT5ctfxunHHwe3WmHK5WB+5hnYHn4YR48eBedctsaWbNeV1tidtvAEUKRHi2acE6XRi9KMY2dnRzbjkBwJm5HiWwmtSG8zwnHXQ8eKHnU6WyM/qGw2iytXrsBms7U0JaiZ1JLOpqZdIj21Cjvp+Ix8PUh21Lu7uzV/jrqT3qoora0ospNOA6ofRcFkwp0uF2wa3aNbIno00q2uvvvdGO7p2ScIqllxr5daUpSUmM1meRIK7C9a93g86AVwhyq1r+/974fodsMkivuOt2xRPU/D/fUfAJi5qB6n3LiLHM6YGeBZpM9+aH+jU+l1e48BQqrYSbDGRqVawrJZiyJsexv2974XLJ3Of1cAOAoCqIgqBJ0evZzklM9USnaYK3LYU1gUDw0NQRRFRKNROdWScy6nYQYCgY4QCyR69tOKc6JlxhEOh7G+vo5oNAqn0ylfe61qktps0cM5b5sF+VJ0pOhpZjpbJBLB1NQU7rzzTvT19TVsP0ai1nQ2NUYXPaIoYm5uDnt7ezUJAqOLHqUd9aVLlwyxgmNa/RIszz1SZGAgHnvt/hdqOLSpb8XmRAKe974XeM97iiyrAf1ET7XbUadb3VpcxEgZa+VGUjJFyXe+yCK6mkiQumg9Fosh+a//CsFkKvrBYQDM8ULkRnW8xeLEBAjxIuGR/78AcAHIZW6Pu4LUqtzgw0hl9+C48iRgssI+9RS41btPMOUGH0b8Z14KS+ZzsL/nacBqq7lRab3Csl5KmXuoa9+qEXR69HI6yGFP6xybTCb4/X74/X55NT4cDsvW2FarVRbgbrfbEPe0ammX7IBmIopiyw0u7HZ7UYqvsjlvMpmEx+OR64Galf7VikgP0N7Rno4SPep0tkauDHDOsbi4iK2trdY4XLWIetLZ1Bg5vS2dTmNiYgJdXV01CwIjizrd7aj1IL0Fy3OPgAlJSA1fLc+9A5kjD+2L+GhOlhwOiIIAbrPBFC9MkmOx/HYKltVSk9JWRHokitKtFhernvjpiWaKEgD3vzwImPPOXtnjb4R15Qs1TdilPhq+Bx6A+YDvArdYgKUlQHG8kv2y+db/huPyL++v8VFTYWoVS2/DMfVeMJ4GhELUo4Rg4vYeZH/+Pcj92C/UnHp4YO0L5zBnQzBF5jQFZaVis1xqpLaZgYD0H/wB7E89VRRhrOb46u3lxIeH9/cQSqUqFl4Wi2WfO1coFMLS0pLcT0QSQe1Sh0CRnv0IgmCoz08dgZSa86qbpAaDQQQCgYYJtmYbGbTCOEFvOkr0AGhKOls6ncbk5CS8Xi/GxsYadhEYacVHj3Q2NUY5NjWhUAhXr17FqVOn5NB2LRhR9HDOsba2VpcdNeccLLOtu003iy/nIzxQdo63ajYl1ZrEiZzj1t//PXoWFmD65V+WBQ+A2yLCAKJHTSlr5VpSqKret2aKUjIvGHN5QWBd+nShnqb2YvV9xfSxWFHkhmcyeG5nB47paXmSarPZwO09EI78KMDY/lCeGjGdN0AoQy21KLU2KmXb2zBP/W8gZgaUX7XC/oJ738DR+d8HM9n2CcpKo0OVpEZqpVZKr8v9xE+UFXQHiS/JIpttb8N05bnqheE+l77av5cOhwMDAwMYGBiQo4zNnojWi5F+942C0YWgsjnvyMhIUZ3j0tKS3CQ1GAzC5/PpdizNjoC1u1010GGiR7KibuRkZnt7GzMzM3VPiMthpNQovdLZjI4yenfp0qW6rXiNJnokO2oA1dWeKWpsTCYT2MoXYXv+sfIpaFXC3cMAV03ASzUl7e1F7lOfguUd7wA3m8GzWSQ/9jEEf/AHId59N/Af/2Px61UiwlCip8SEtBmF8vtTlNIAZwdHVQoTdgDFTTDLRCWUqX3miQnYnnwS3GKBSRCQeeYZ3P8jPyKvll65cgWiKMoTBdO9H4Nz4pfyGxKTAHPkU9sYy9fmiEmAM7j/9SVlI1F61KJUgixGLBYgHQN+EcCLFfsze/KNjXk634h3D3D89SOIP3we6OmuyBlNywa9VGqkOrVSTiUsI+gqEV+11qSV6iGkR5RTijJ6vV4MDw9DEAREIhGEw2EsLi4W1aIZpTAdMP4EvxW02zlR1zlms1mEw2Fsbm5idnYWdrtdvrfVk4bZ7AiY1Nuoneko0QM0bjIj1Xfs7u5idHS04RealPrVatGjZzpbyzmgQD6TyWBychJut1u36J2RRE+tdtTqGpt+z6OwPf+JilLQDkTrs7D3InfpUxU3Jc09/DCuDQ6CLS/jxEMPwdrfn39CIYikSVjuU5+SU9sAY4keoPSEtCn7VqYomT1w/8sPHfwGMQtT5DIc//YqeSJcKgVOLYSkCbZ46RLWxsZgXl1F79hYvmgdKFotleo1bt26heu7I3D3fwlH3Al4uvrhsgqAawTI7MD9Lw/mo0Y8BXDA8ULe2Y17T2kOX49alHIUiRFpv58BcMEDeASkz34I5siz4Lxwf/8m8tbY5gzcv/cipJ9+CjhSPhpVbWpktRGrSmypNYXXo48ifv48+Cntz0AeTxOjnGazGd3d3eju7gawv/eUy+UqssZuFe02wW8G7X5OrFYrjhw5IqeSJ5NJhMPhojRMSQRVs9ja7DmiNNZ2puNETyOQIh09PT0YHR1tyopQq+tdGpHO1koOKpCX6lvuuusuHNVw+ipJGZcxo4iemu2oNWps7tz9WN4tS0mJFLRSHPRZVNqUNJ1O4/Lly+gdGMDIi1+87zspvva1yDz00O2eML3F2zGa6AFqT6HSZd+FFCXL+pcBiIpMMiuyI28pEjTpsx+CY6q4wadWClwquwfH1HuLhIWgqP8QgkFkA4GSx6yu15Cti2+GkExm4fVuYcC+DpfZDlZIxQMAiGm4/+VBpO7/RMmIT721KOXQFCN2D1J9fwgMp+GYegpgFgAJYA+qXkA52H/tD4E/YYByfqFlrd1g0VBJKqDmsabTcD/4IFKf+MSBER8pyml79FEwmw0sl2talFPdeyqRSCAUCrW8USWlt+2nE2pJlDidTjidzqI0zHA4jJmZGaTTafj9ftkZ7qBrr9nnRXLjbGdI9JRhc3MT8/PzTY90tFL0dFw6W4kC+XTvS7F6K4X19fWq61sqcRlrteip145aq8aGwwpwleNTqRQ0LSoxKyjTlDQcDmN6ehqnT59Gt0cECz+nLZB6e+Uann3HZkDR02rkVX3F58tNZmTufgqZu5+SBUIp8wPsId/XphdAlxmOK0/mzQIkIfT8IwBMgDlfv+I9/nvY7fp3FY/P6XRicHAQg4ODsnVx5GYc/UrBg4Jw4OmytUeS0GsEmmIkJ0C4axTuF16SF4zSa7fyGXpFWC1Id70bduHpA6NRjU6NrCQVULP/FACk0xW5EOYefhhXBwZwTBDgPHOmJcKfMQa32w23213UqDIUCmF1dVW2xpYaVTZyotnuUY1GYJRU/0agTMM8fvw4RFHE7u4uwuFw0bXX1dW1r0lqKyI97W7aRaKnBIIgyKq7FZGOVomeVqSzNXplq1SB/NwL/4S0597qeytV6DLGGGuZ6NHDjlqrxoYxAckzH4Lz2q9XlIKmphqzgn3jKTQA3tjYwMWLF+He/p+w/N8K7K21xkGiZx/aq/o2sOQyxMCl4gm3eiIsp2cBEAC8PQk86JTd0QAAPFMwSMjXC/Ut/wZingcADFU91tvWxaPIuD8BxwuPAmK6yBxBhBmZ8Cysfc2fRJcSI8wd23+Oe5E/Z0pyAnKjb0HO+5ay0ahGpkZWkgooH+ujjwLp4s+gUhfCbCCA3PHj4AZZRVY2qgRu12RIv492u12OAultjc05J9Gj4jAJQZPJJIscAPts2aXmz11dXZTeVgMdJ3r0uPnEYjFMTk5icHAQZ86caUmko9mip1XpbM0wbNCavIu5NDx953By5FzV26t04t5oU41SSOl6d999t5waVBMaNTYLwafQc/znYT7xcE3ubVWZFShQmjCMjY3BnAtVbG+thabo2doqmQ53GKi0wH/fRDiSAT6bBTKK+9VnTcD5XHF6lgbWzDqAe+sad27wYcR95/M22/y2yGI8i43laYSWYnB1DcsThWa5HWmJEZbe3n+O/Vbwt3Pg07m8aOQ2pD/yYTDLMjiGIQYuld1XI1Mj5VTA3ctgAET/fftf8/DDiJ8/D/eDD8rNTwFUnGpn9JQurZqMUCiExcVFJBIJeL3eItfBepD6CxK3OUyiR406zVdq/ry2toadnR0IgoCenh50dXXB6XQ29NqJx+Pwer0N234z6DjRUy/r6+tYXl7GuXPn4PP5WjaOZoqeVqazNcWwQTF55zCDC1nEz38U/SO1TbYqnbg3O71NioTUkq5XCnWNTejqOoKiCLgOTkErSSVmBSrxkUwmcfnyZQwMDODYsWN5S/rdZY3B8opri9Six/SlL8HyyCNFxgfKZqaHgWoK/JU1MexaBE77zwPpvdsvsDmR7nqnIj0rnf+ngPEURJM+qRLcewqp+z9RaGZqBoQUGBdwNvR+QMxip+sPsLb3IJaXl+WVVMk+ttz9rprGrPvGpRIj6nMsCmlEz/wxLK94JdhbL4NtAcy9DMfKU8C3WtPIVAvz9tfK22ePdCP99K/V1MjV6KJHjTLVUtmj5cqVKxAEAYFAQLbGrva37TBP8EtB5+Q2yubP3//+9zEyMoJoNIq5uTmkUilZgHd1dem+eJ1IJCi9rVPI5XKYnp4GYwzj4+Mt9/BvluhptTtbs4RBduA1mN7phzm1ihPnH4LdYgELPVtbn5kKXcaaKXoEQcDU1JR8/eoqIhU1NibTjbqjVweZFajFR/jpp/HC6dP7rk9u8QCCqqYklQL7u/8L/obRsmMoEj1bW7A88kiR+5S6melB2+kkqinwl/uz3LWtWUyvTM9CJgLXd/99kRU2hx0mMaHr2NPZPdin3gMgV0ilywux7rn3wPHyKfA770Qmk0E4HMbGxgauXbsGt9str9KrnZMq7ZVT7Tilczy9nMDx/ksw213gd/4w2NA23P/0+rJW1Y1Cq8lpJQ5u8nk6YgU+AqS73onc6Fsqjj61m+hRotWjJRKJIBQKYWFhARaLRRbZlVhjt/O5aBQkerQRRREejwd+vx9DQ0NyrWMoFML6+nrdAlxNLBZraTBADzpO9NRys9jb28OVK1cwMjKCgYGBBoyqehoteoziztYMYSDZNQ8ODuPYsQdhXvursiYEAA50Z6vEZaxZokdpR33s2LGG7ku3Y9IyK9AQH/53vxtj09OwqQQ5y8WAtB1AOl8PYQbYpwHLC+9H5kf+Q0ViRRI9B9r+tlD0aE1Am0G1Bf7liumlHj77GowyhqxtULdxs/Q27FPvBRMz+59UOI7ZbDYcPXoUR48eLXLtunbtGrLZrNzAssuZg6eCXjm1IJ3j3NqV4mOooXGqXlg+9zk4nnwy/xkKgtxrp9yY9okiD2AXnkbO+5aK991JE321NbaUjrS6uir3OZFW4rWssWmCvx86J9qo679u1zr6ceLEiX29qZRRbq/XW/U5jcfjGBzU757dCjpO9FQD5xzLy8u4ceMG7rvvPkM1XWqk6DGSO1ujxd3NmzcxNzd32665QhOCStzZyrmMNUP01GxHXSONNADQEh8mux32GzfAVTda7h4GbgD4CAAv8q5hUQA+236xolGro7zm67H95ZznO9GvrOgqTmpt9tgqyhXTa6XObR7/PQjWoG5jKOkqB5RsPqp27VJ2Ut+Z/TYucFPxj2QZAVJtKpx6st/oxqmlxmf93Odgf9e7ZNc1QNHk1HvwmA4SRdLzZaOGHSR61CjTkTjniMfjCIVCmJmZQSaTuS2yC/VmZGSgTadeH/VQ7nuj1ZtKinJHo1E4HA5ZBLlcrrLnmIwM2phMJoMrV67A6XTigQceMNxNxmw2I51Ol39hlUgiwCjNRhslDEpFsioyIahQGJWjkaKHc47r168jGo02NVLXSNFTlfiw9yL3w38Ey28/AbalfH2u6PUV1epU0My0FOa/+iu4VBGOesWJZrPHCqx/9aSWOpZyxfTq1LnoTga8wgWPSqJeWoKBAwBzVNx8VNlJnR0LwPxPv1IUnRKFNDb3HPA70/saVOuRCtfIxqmlxse2t2F/8knsm+6YzWDLyxAvXTpwTKWEmnnh32D/xiuBo1bAIxx4PjpZ9ChhjMHj8cDj8RTZE4dCISwvL4MxJjvHUXSD0Bt1lFtqkrq4uCibFEgiSH1/A/JZJUYKDtRCx4meSm6coVAIV69excmTJ2U3FqOhdwREFEXMzMwgkUgYqtloIyI9qVRKbiarjmRVYkJQj62yEpPJhKx6Eq8DmUwGly9fRiAQwMWLF5s6WWikkLvFOXbf/W6c/c//uSLxIV54G3KfACyP/wpgswHZXPHrq6jVKdfMVJOtLdgff7ykOKm1AP7AdLs6RE+l6XJ617Go9ys3tWQ3Knp/pVGv/YIhg/TJX0Vu+BdqEg1aAmT31NNIcg/Wp6eRy+XkfPkuZ65s3YvmPjQm+41onHpQXQ5bXs5/f9SLbIoFh4PGpHWesssPw/7zv5nvPySkgV8EHCh9Pg6L6FGjtifOZrOYmZlBJBLB1tYWHA6HHAWqZCWeOFzUcz0wxuByueByuWRDjlgshlAohOnC/W1rawuRSASvfOUrEQwGa4r0/P3f/z3e+c53QhAEvO1tb8NTTz1V9Hw6ncab3vQmPPfcc+ju7saXvvQljIyM1Hxc5eg40XMQnHMsLCxgZ2cHFy9e1MynNQp6igFlOtvp06cNdeM0mUy6ip6dnR1cu3Yt37iyENItogITglptldU0QiBEIhFMTU3tt6M+oP5ITxoR6VE2Ub33ySeReeSRisWH+Ia3IfMjP6n5+qprdXwA7gBQ4ULWQds3pytwuyqBZsQrna6rh0mlwqGSovVG7LcU1Ua99BYMucGHEcd5mOeehXDXKKyDpzAMYHh4WM6X35ufR2LqH+C2M1iUbq511OLo3Tj1wLqc4WEglyveP4DUhz+8z3muZK8g5XmPe+B+y4NgytKqzwC4YC55Pg6r6FFjtVrhcrng8/nQ3d0tW2PPz88jmUzC5/M1zJmLaC/0/h1WNkmV7m/PP/88/vEf/xEf/ehH5Xqhq1ev4sKFC5qRIDWCIODxxx/HP/7jP2JoaAhjY2N49atfjXvuuUd+zZ/92Z+hq6sLc3Nz+OIXv4gnn3wSX/rSl3Q9NiWHRvSkUilMTk4iEAhgdHTU8GFjvUSP0dLZ1JjNZl2EAecc8/PzCIVCuHTp0j4XJiVlTQgqdGcrh56iR9mYU21HXVH9kU7oLeSy2SwmJibg9XpvN1Ht7S1rIFBEiddXky5Xyzksuf0BDxzfr104FBkDcA6kUoDJBPdLXlJT+lw1wkHPQnq2vQ3H44+DpVI1p+nVEvXSUzAcJNrMZjOO/vM/Y/iJJwCLBUgngF8E8OL8e0UhjZtRJ/yuTMkJarMm+wfVCnG74nozm4FsFuk/+APk3lLaiEAraiidd9P8c3nL6pQicmQGcLN0bRKJnttIaW3KlXgtZy5RFOVIo9/vb2qTSqL1NPo7YzabMTo6itHRvBvq1tYW3vKWt+Cf//mf8dGPfhRHjx7FD//wD+PlL3857r33Xs059Xe/+13cdddduOOOOwAAr3vd6/CVr3ylSPR85Stfwfvf/34AwMMPP4wnnniiocfWcaJH60RtbW1hdna29Oq/AalX9Bg1nU2NHpPoTCaDiYkJ+Hy+ygVtGROCStzZyqGXQJDsqE0mU74xp/LHTaf6o0rRM9ITjUYxOTmJO+64A319fbpss4hKa3VqPYe9vUj/6Z/C/thjRZNi5ozVLRyUzR4ZUFdtTzXCQc9Cesuf/VlesCmpMk2vHpOJeiknFouel8b7aQB3uoF+Ebt3/yHiggurV67IE9Tu7m74/f6mL7qVqxUqZ0ShpFz0TjNyJACphz5c8von0XObUkYGameuXC6HSCSC7e1tzM/Pw2q1ylEgj8fTMeezFQ2+24GG9zdUIWWW/PEf/zG6u7uxurqK//N//g+efvppTExM4J577sEb3/hG/NiP/Zj8nvX19SJH2aGhIXznO98p2q7yNRaLBX6/Hzs7O+hpUP1qx4keJcpi9tHR0YrCcUahHtGTSCQwOTlpyHQ2NfWKOyndqyH1WWWEUTn0ED3xeBwTExM4duwYhoaG9j2vV/1Rpegl5G7cuIHFxUXce++9DXWDqaRWp55zKPzszyL6Qz8E8+qqPFlk6e2qhEOpCR+LxQC7vbjWoobanmqEg16F9Gx7G/Y/+qP9xfGZTFWCpZwddrVUYwNeTixqPp8F8FQG6Y89DevJN2EEwMjIiDxB3drawvXr12G32xEMBpFTiYOKj6OGerFyqX/ljCiAyqKGRZ+ZpRA5+pM/QO586cgRiZ7biKJY0bmwWCzo6emRJ4epVAqhUAgrKyuIxWLweDyyKUc7zX3UkKGDNs0WPUCxe9uxY8fw5je/GW9+85vBOcfU1BRisVhTx1MLHSt6EokEJiYmcPTo0ZbbMtdCrWJASmc7e/YsAoGA/gPTmVon0ZLd+Obm5r50L6NQr0CoxI5ar/qjSqkp0qOwjBa7uzE7O4tEItG8JsBl0uXqOYe7u7tIm0zounBB/gHSSzjoFeWoVjjoURcjF8enlM1IgfSv/VrVgqWaKMRBVFtfVO78az3PACCThf2dTyL3ip+Qx6qeoEq1GolEAs8//7xsWxwMBmG1Wg8+jhqMJpRiT+y5dOBr9703OgNz5FkIgVGw5VhFUcOiz6zPA+aOgaW3D7yW2u03ulHUOsl3OBwYGBjAwMCAZlG6dI0FAoGWN1+vBhI92giC0PTzksvlNLOGGGM4d+7cvscHBwexuroq/722travz4/0mqGhIeRyOezu7jY0I6t9rvwKYYzJq8j33HNPW0z8tahW9EjpbMlk0tDpbGpqsebO5XKYnJyE3W7H+Pi4YW+I9Qi6iu2odao/qpRqj0ltGT3znvfA+trX4v777zfOJKeGc8g5x9LSEm7evAm3242FhQV59T4YDMI18BoIFQgHSURqnQs9oxzVCod662I0BYPDcWCdyIHbqyAKcRC12ICXbboqPf/oo0A6XRzVSqVg+dznkH3PezS37XQ6MTg4iFu3buHs2bOyCFpbWwMA2TbW5/MV3d9qMZqox0zCPvmrsC59Wv47639j5VHDnp66DD0OK3pEvbSK0iVr7KWlpbqbVDYTEj3aiKJo+DqusbExXL9+HYuLixgcHMQXv/hF/MVf/EXRa1796lfj85//PF70ohfhr//6r/Gyl72soXODjhM9giAgHA5jbGys7IqZkalG9LRTOpuaaifRUh3IiRMn0N/f38CR1U8tokeyo+7q6qrYjlqP+qNKqSrSo2EZffeHP4zs294GNPIa1XKyUz+W3oI3cw1InwTsvRCPPITsi/4K4ADvunDgORQ2N7H4ta9BPH4cl8bH5RU3aeK6sLAgOy11dx9HlydQ8422nijHPqtolXCoJtWrWvROS6uXWm3Ay51/ufbqB38QyNy2K2MA7E8/jdwv/EL5lDHGimo1stksIpEINjc3MTs7K9sWB4NBeDKVGU3In63HU3PPJxadgXXp00Vizrr7BWR/7mdh/fMvy49l3/hGzW3p7QR4WGjEJF/ZfwrI/86EQiFsbGxgb28PLperqEmlkSDRo02z09uk3/1q5pcWiwUf//jH8aM/+qMQBAG/8Au/gLNnz+J973sfRkdH8epXvxpvfetb8cY3vhF33XUXgsEgvvjFLzbqEPJjaujWW4DZbMbZs2fbvvjNZDJVdAztls6mphpxt7a2hpWVlYbXgehFtaKnpB11JdRZf7SP9BZY+AWAATxwQRYB1ViMa000mc1W2jJaB7Rc2ABe9Jgw/PMwL38eF0QTzH/3K/LflTi35b7wBTieeAKnrVaYBAGZT3wCwk/9FIDbq/eDg4MQRRF7e3vY2dnB8vIyTCaTPOnwer3yD0clIrKWKEe51f3i5zNI/8GvIve62vrZlEJPwVYvmpGnCuuLyp1/fuoU0k8+Cfvv/m5xtIexsqJKa1XfarWit7cXvb29cgPBUCiEubk5CIkd/KCQRtFUR1UvVvTZptP7FxgqrAszR57d/+AeYP3L/14shL7wBWSeeqqhToCHiVJGBnpis9nQ19eHvr4+cM6RSCTkayyVShVZY7d68bgVaVztQCtqemrhVa96FV71qlcVPfY7v/M78v8dDge+/OUvq9/WMDpO9BwW2jWdTU0lwkAQBExPT4Nz3rw6EB2oVLgeZEfdCkyrX4Lle2+7XedisiE3+lmIx15bVaRHOHYMFnXaTyOdt7Rc2J59OwCAiSn5MfPCJ8FQuPkJt/8u59wWnp1F7xNPwJxOy+YCtkcfRfrBBwGViYbUVV1aiJBWVtfW1hCNRuUiY865/v0WqnEdKzxv/5Xfg932NFIv+VNd048aIdhqHUfqmWfgeMc7bosfUYT561+ve9sAkHv1q2H/3d8tfjCZrKu3ElDcQDBvW3wO4bmnEZz9VYgwg3EBawP/CeaUFV6rCHMotO+z3Xd1VfgdFAKj+x/cAmBSLXqYzQ13AjxMVGpkoBeMMbjdbrjdbhw7dkxesAmFQlhdXQXnXI4CtcJ5sB3SuFpBs8VgLpdrm7nXQbT/ERxClCYN7ZbOpqac6JHcy4aGhjA0NNRWx1qpoCtpR90oSjUyLUR3LN97O5iysF/MwPLs2/MpdBWKnlQqhctLS7jrAx9A/2/+5sGW0bWiMEhAb28JF7YazqfKuY1znndE+uY3cVRaPZewWmFaWQE/cuTA6IR6ZTUWi2FnZ0cuZJeiQHpMKmpxHWNmADdTLU8/qqX2plKEl740P0GXRE8mU9G2K4k6sVgMcDiKLbodDrBYTBYdmr1tqhC8LL0NS3IZ5uEfQ2L4x8CSy8juOmCa3cCt5BQWYjH0P/887jCZiiNBTie4KOadAKtIM+TeU8iOvL2opof1AlAHemMxmCcmIF4qNkjQy9DjsNHqdC71gk0ul0M4HJaNdex2uyyC3G53w3+TW30+jEqzIz3xeLzlC7J60JGipxFd441Cu6ezqTkovW1zcxPz8/M4f/48fD5fk0dWP5UKulJ21A0ZU4kmnPLjMAFcw1iCmcHiyzCZhsoKuVAohKtXr+LMmTMIPvAAMq997YGW0TUdh8ogIfepT0H8yYc0XNhqsENXOLdJUUbGGM5nMoDakjObhXj8OKxVRCeURcbhcBinT59GLBaTJxVSDUd3dzecTmf1w6/BdQwCgF40Pf1IfZ+utfamErQc5cptu9KoEx8e3p9Gxhj48DDY9jYsn/sc7E8/nd+/ajsV2RNrOLbhmxz+J56A32rFUCoFcA5us4HF48Vj4xzRf/kXWJLJqtMF0+efRmbkF2Fd+BhsK/8V8AH4DwA+pzhMAPannkLuJ35if82TDk6Ahw2j2XdbLBY53RK47Ty4tLSEeDwOr9cri6BGWGOT6NGm2REwyQa93elI0dNJSDfATklnU6NVIyIdayqVwvj4eMtzimvlINFz8+ZNzM/P49y5c80TdKWacPrvVTxeAi6Au4dhypVOx5JsxG/evIlLly7B4XDknyhjGV01GgYJlne8A5mHZrVd2ArHebum580wL/85BJHBbOLy34AFEDPInfxDwN6LVCqFF154AQMDAzjmcMD65JNFqXocQObDHwaAuqITykmFVMOxs7OD2dlZpNNpuet6V1dXRT9yFbuOPf4YgBSYAOAXkZ/QNjH9SNOxroHNSKvddjVRp1Ln3Py1r+UfS6Xy105BcEnbqQRNQ4B/fQx4F8CSqeKIXeH4OAB4POC5HJbf9z4sJf7/7J13fBz1mf/fU7ZJqy65yd3ggiu4ACkEQiA5cgnJncOFdAjYgEkgxjYQSCc0Y1Ih2CSQ9suFkLskdznSQ5LLJQTbYLnjLndb0kqWdrVtZr6/P2ZntLM7W9QL+rxevIDdKd/vzGj2+Xyfz/N5Os26snCYaq/XWVdWoOePKJtFcvonTNIDMBXwA+l9Z/OQx946Ab7eMNSD/PTaRSEEHR0dDmts631VWVnZJ0H5UL8egwVd1wc0Dkzv0TOcMUp6hjCsoDkej48YOVsmFEVxEINoNMr27duHpRNdJtxIj2EY7N+/n3A4POAOg7macMqhzVmfO2iN7EVbsgl8dUhSkyuRs2R6iqKwdOnS3v9IZUjXHPPIkw0wlrg72WV+pl/waXa+9EsuWHYNcmAsYt8FqA+vgXMe1Na1nNsg88qsWWa2qroaacuW7HMGg4hFi5CPHi06O5EpccrMSqfXcFj6+ra2NkKhEIcPH0ZVVTsLlE9aUpTr2OWXo255Fl/reqj0Dgn5UX+6vnX32N3NOmVec4DSuXOR0jNLmccpYlXf1RCgWcn/6x0MElu/Hv3tbzf7A5FdV1ZaWsoU/s74xvuzLKWznP/SpW51gIGzVi8RRdQk3ccyDNGfzoaFMNQyPfkgSRLl5eWUl5czdepUdF2331eHDh1CVVWHNXZP5jVKetwxGPK20tLSATtff2FEkp7h8sIoBEVROH36NEeOHBkxcrZMpBODpqYm9u3bxwUXXEBVVdUgj6wActXFpCEzoLXIa3fsqPsSuZpwGtVLsz+X/SSXfhs8FQ73NjfpaGdnJw0NDX0m03OVrl3X5aZWcMXezcku8zNfHWHfHHNeTU2ot6xDisYBU9pX9tk7Wfa7f8dTquc+p65jTJ6MoWlFZRDcpFLS+efnleKmu76B+Qy1tLQ4pCU1NTWuTS3TTQTcVvNFbS3Jd6xFi98wpORHfdWMtLfH7knWKf2aKy/9AdQcwZp1nCNHCo7Z1RCgVgctz066jv72tzvml1lXFm09St3f7kMSMZtQ+bbdhnixjcBd92VJ+iypm9K2BekrLfjWfBnk1LP/8SSlO64m2bGC+PzHCs7JgkUuPG1tRe9T6Fi9fWb6w0ijuxiuMYyiKNTU1NjNJePxOK2trQ6ibZGgYqW7w8WlbKAx0EYG4XB4RJCeUfo8RGEYBp2dnZw8eZKlS5eOSMID5ktS0zT279/PkSNHWLJkyZAnPPKx5/D+aiae/70G769mIh/7ScF92tra2LJlC1OnTuW8884bnB+1VBNOoQQQajlCCZjyr/LZ2Z8v2YSYuBwx9ioHqcvMXjU1NbFz64ssqI8zsa4Xeu54E1JoC5zYa0vXpPZ2pGgUdeVKaGrq2rauDm3jRkQggCgvRwQCPTJIsGSj9oq+hUtAXp+gdM9Huu5vnnNaGYT07zIzCOlSKWte/lWruh3s+Xw+JkyYwLx581i2bBn19fW2scnWrVs5dOgQ586dc9wj9cTzlP5+LiV/v5bS389FPfFTxzGFrxajcrFNeKR4M3LbVqR4c7fG1pcQtbUYixf3yyp7oWNLzc3IW7cCFLyvuaCeeJ7A4eshnlFfAwi/3z5OMav6liGAkFN/n3KA2GVPEnviya6xeb0Ij6focUqSRFBuRlKc8hjRLuP71N1Zz6nUnHoW4jUYZ2ej/dMHiPzjF3B3EulrIL3BzPx4jmxC6nit4PUBk1yUzp1LybXX8oYPfAD1pz8tvFMRxyqdO7fHx8r1d2rPfxTdgs/nY9y4cVxwwQUsW7aMqVOnYhgG+/bt4+WXX+a1116jqamJZObiQhpGMz3uGOianlF52yj6DVYQo6oqs2fPHjH1O25IJpOEw2FqampYsmTJ0F/hylUX42JxDF11LqdOneKiiy7qUWF6XyJXI9NiG5xamR4hBIcOHUI+9hxval0PocI9bnLBYa5wIA5Kxg+cJQVKIzXGddeRuOKKXhkkWM+aY0W/DFgBkhcQEdC77q/rOVP1aIUyCLmkUv7Tp3tsuuLW1LK1tdXRcHBMucSMncU3iMwsmI/PfRi9ckFRWaDBlAT1FdxW+SO7dnVrXnYNTjBm1ko9DUIBhI/4mrVFNSzNhKshwHKccrrwAdTdf0C74ErE1EsKHtMtgyQ36eD1Q7zrc02SOPKnP1HT2sq4NCfG5BevhxnZx1VP/TfJsll5z51ZL6XQc5e+vnT8608jjdc7JEkiGAwSDAZt6e65c+cIhUI0NjYiSZKdBSovL7eJjmEYI8Iqua8xGPK2UdIzij5Hujvb8ePHu9XccrihtbWV3bt34/V6Of/88wd7OEUhV11MusWxBU3TiMVidHR0DJwddTHI1ci0iAansiyjaRrbtm0j6Ikyp229I6DORwBdkUkiq0EkMrbJJSnqpUGClelp83hoWr2aCzZsQJqqgB7JaDKZdn8zzplO0vP1pMkllYqPH99nTpMej4cxY8YwZswYhBBEIhE6j/8ZXcjOF72sujq0uRXM+3bcAUoQhG7Xe7hhKEiCeoucwfOuXVl2zK77pySEUqKtqwbnDcA8IFRK5zU/xJhxZfZ+RS70uBkCiNpaKAPvjk/hOfULAHw7HilKZuZqKX3FQ/i/fK9jO1UI6qZPp+4d7zDrkyzzkM/8P/gKpgFGGnyvfRE5fjrv+fuSXPTlsfrTSGMUTsiyTFVVla3ssBZtTp8+zb59+/D7/VRVVRGLxSgrKxvk0Q49DDTp6ezsHJW3DVUM+WyBCwzDYM+ePZw4ccKWs+Wzcx7OEEJw+PBh9u3bx0UXXTSsVnFy1cVYFscWIpEImzdvRlEU5s2bN3QITy8RjUZpampi3LhxzJzgTRFAoB04CHSY1tYF0dSEtGUL0tFtXccAM4C6xY/wexGlpQi/35SRlWPK3+JpMjdLEmd9lvn/BSBJEidPnmTPnj1MXLOG5P79JL/5Ywj6nRu63F/7fK1bipKB5ZLAaf0k5bRWVcdOuxhFyjDT0BLsONjBsWPHiEQiNumyC+bTjwNIehjJiOLftsp1riNFEpQlcwQ7eC6EdAlh4OXrQU8zLygHphuIiQv7dsDWeX87B8+pX5j3iu7JzLT65UTetovOS39B5G270Obf6PqcBoVAylAcGIrKmfbLTMle6rNiz99bcmFJEKXm5j4lKsVIVUfRP7AWbWbPns2yZcs477zzkGWZlpYWDh06xJ49ezh9+jSJROaq2OsTA016wuHwiCCfwyfaHMGw5Gzjxo1zOJaNRNKTTCbZsWMHgUCgb1y+BhqpupgsW+S0zEa6HfXOnTsHcbB9izNnzrB//34qKioYP348Iq6aBPBvwNOAAuhhJHkb4oOpbu6WC1swaDZqnDIF+Y9/NOt0FAV0DW42IF2NI3TS12Mk6e94f5VpQy0c/Yb0KR9FafxeVv+hXIYThmEQDoeRZZlly5aZPx6BAKLuKrRjm/LeX3BK8nxprlf54CaBk3bu7NeeYm6r+YlFTzC9+mLbYSkajVJeXk5tWQlTMwvm05Gjh89IkQT1xNJaamxEjAtmZcgEHoTsB7n/XPHszBwufbUApW0LWgGZGWRnkFyf0+bmrGujGAbB932X8OlvEzz8YPaBW/4BOc7vcNJTVYxEgkSKXBSSSbplFfvS8a8/jTRGUTwsF8toNEpVVRWqqhIKhdi5cyeGYdjW2BUVFSNmUbE7GGgjg0gkwpQRkPEcJT2DjHzNRkca6Tl37hy7du1i+vTpjBs3brCH02PkrIsZRDvqPkUGURBCsH//fjo6OliwYAGHDx82t/PVoU1dj/qR25FSi28SoK66i8T8C1D+588oqT42xGIQCIAQkEwipT3X4ikFMdUHMRX8GjwtkGIpJ7UyUJLfQpLBrqHafDNIEpIRsz9TDj2VkqR1yey05DnU7euyiFA8HqehoQFVVZkzZ07WD2bB+iaXuq58dTLpyJTADUQjZbd6kADYvTYMw6C9vZ2WlhZ2l61mdvtjIKnIRobML0cPn5EiCeqOpbUj8E7EYYUEF6dvEKBz8ffAW5m3Hqo3997VyjoNeuWSHh878znNd20k33vBhfTsOh2ks3mLXadRUVHhCNIscmEcOsTOcJgFCxbgeeQRfBs2uDZxhfwSxO7WXrkhnXAVI2kcRf/DKthPr1/UNI22tjaam5s5ePCgbeVfXV1NMBgclmqf7mIwjAxGgrxtRJKe4fDAF9NsdKSQHiEEx48f5/jx4yxcuHBE/OFk1r9YdtTV1dVZdtRDuu9CBsFxmAqIJLGFT/BKy0wqKiq46KKLiEajjkBNaIvAXwaJjq5j6nG8b3graBm9PKLWKngGkjrcK4MPSAgok2AC0ITZEyTzT0DEgQLmHpKK2rAGyYiTToSa/YvZse80s2bN4tixY7mDzjz1Ta51XWlZkKFY0J++mp85PlmWqaysNBddZtzNufCHCZ/eida0mSnNX0VIKjI64blfRXJrXtmPvXUGGtry5UTmz0fZsgV9yRLErOxMhVvgLZ4C5tBV32IkERULoQOknY0whT6/Hm5GBNbTnJy6AlFElqc7yJUBcfTwSTv/BfP/BU3TaG1t5ezZs+zfvx+/328Hp4FAAGpr0SoqqH38y5T+23UQT7g2cbWt1/P16Oql299QqEvr7wWQ4Qg39zZVVc3+U6n7HYvFaG1t5ejRo7a1svWc2U2yRxgG2tWus7Nz1MhgFD1DLjlbJkYC6dE0jd27dztlRC4Y0sSgACxDhlmzZtkvYQuWxfNQTL87CI6RQD/vEygHvuHIoHhfuZWpF79E7UTTaCLTstpc5Xc2DJF60KNQGp+EUBIuBLECk+gowA9S/07fFhAZbgdZT46RSBGTLumPgcLR3X/hwqX/RklJCcePH+/+QMlR15XKghQbOFlF7x4jNiCBjkV05IYG/Pfem3d8nuAEqs6bAOddTWdsBbGWvTRFS2kKgZFj5X6kSIKKuX+ugbc3gGgxoNpny9mU/3qx8LF68c7Lki5qcRKTP0Ry+i0FCU9PiXkus470Hj565RL7/KqqUldXR13K/KOzs5NQKMSBAweIxWKUl5cz6ewLnL/+cTtb7ECGTLK/sop96QDXGwzn38H+QjHBvd/vZ/z48absWgjC4TCtra3s3buXZDJJRUUF1dXVVFZWDqsa4qGEUfe2UfQI+eRsmVAUhXjcXa89HBAOh9mxY0fBppWWxGe4veyFEBw9ejSvHfWQJT0uEi1l3/qszSTVS11JxF5BlmXZGaSn+teoK24GKW4GLt3h6TJwE7AUk9zITl8D8WHQ5atQ+F0WsRGSCkJzfG6NTJ90Pcrx55w7GElmL3kHakmJObeUe1u3kVnXlQpy6aCowCndFnqRHqe95CsQ/FD3x1Ek7EBeVaGjw7xexQZ2/jr89XVMAiZBzpX7mpoaAnnc63JhKK1sFxv4ujeqhci//hWpNGxmYDqg9Pa5/R5Eu1pZF0B/ZTRE2ayCNURWncbEiRPNurrmw4z5zZeRcr0eMwhNf2UVh0pd2mhPmmx095pIkkRZWRllZWVMnjwZXddta+wjR47YrnHV1dWUlZUN2+s90PFSOBweJT1DFUMxeE6Xsy1btqyoeo/hnOk5efIkR44cYf78+QUdP6x5DqeXj6Zp7Nq1C1VVWbZsWc6xZ2ZGhgrcJFpufzWS0BzOZW5EwbjuOhJvXID3xxfD6Th8I+3LMhB1mFI1zSSF+sc+hvLssxCU4P4oUrqKLDMGLilDzP0n2P67rLElF30dz7Y7HR7X1hyU48+hLViP2rAWXcjI6OhLNqKWjnfMpaew6n5E+DAJdQJSYIzZ0LJA4JRpC60AFXtX0znxHX1e6A4ZgbwbuhnY5Vq537dvH/F43C4uLmZFNd/1HwyJYLGBb87Au35W1+LAzvzPgjW/7jandYOblXXOOQ5ARqPYeyfLMpXeNhjrBT1bpmd4vZz+0pfwlJaSvpzUm6xirrENlbq04bj419/o7aKhoii21A0gkUjY/cw6OjqyJJej198do/K2URSNYuVsmRiOpMcwDPbu3UsikWDZsmVFpZJlWUbX9WFT+B+JRNi+fTuTJ0+mvr4+77ZDlfS4SrSs7wCUUsBAW7DerPkB8NXlLryvn4123dOo/5XK+AhMR7Y0qZrRcg3a+78KdXXon/408r5fozatBr2rHkiQQb6MToyxb7VrjGxIHiR0kExvBDL3kzyEPeezs/aHzKz3Ulm/KMuQoNcmAr46hKcGkTDrEIoJnFyLzyV3V7S+gGsgn2d83UXmyn1bWxuhUIjDhw/bxcU1NTWUlpYW/d4brNoK9/uXgMo2pHhzQYezwscyr7X6/PMmYVIU3pBMknjqqaLn11sy2N2MRnfP1917JwJTkMp0u4krCggN4net5twHb+CcLBNKEWpLolRVVYXag6xivrENlbq00UxPNvp6QdTr9TJ27FjGjh2LEILOzk5aW1ttyWVZWZlNgoZqTDIYGfJIJDJqWT2KwuiOnC0Tw430pJO7OXPmdIvcDUVi4IbTp09z6NAh5s2bR3l5ecHthyrpsSVaW1aAEcvK8ugzbkOUTkbdvtbhfibXL889n78JuA/weBC+JKwAyddFRuSa/0D685WIt7wL6uowyt8Bv/qk8xhZGjbz+deWfsccq6SA0NEWPoa6fS2S4d6zQRgJXtt+kgXlE/GXzXZtltrXzmnFBE5uxecId1e0PhmTS/AtAIJB0PU+DexkWXasqMbjcVtSYv1g1tTU5A0mBjMTkXX/EjH4uE7Jvo/C3mxb8rzNaHM8CwD+W29FSvUaUQD/LbcUNb++IIOuZCweR7is4Hb3fD25d8JXS/vYLxIcdx/yw17o1Ihd8Qja/BspAUqASZMmYRiGLVFqbGx0PGtlZWUFf2uKGdtQqEsbzfRkoz+JoCRJlJaWUlpaai/ctLe3EwqFOH78OEKInO6Dg4nBIMej7m1DGEPhpdETOVsmhhPpsTT+PSF3Q5YYpMGyo45EIt2yox7KczMmXUe87nI6tz5A1elNNt+QAOWAqVFLNzVQt65Er7vcnSg0NaHedBOSFVBNtPlKF3TwfPY2OH4n2re/jXHddc7aGD3qmn2SQ5sxpn7YYSPtJs+zMlTC0Dl94p1c/Jnb7IBN27gR47rrHMftK9KT/r4pmAHIKD439DhtMx/D2w9ZHnAPvuMPP4y+YEG/B3Y+n89RXNzR0UFLS4sdTPh8Prxer+MHvL9rK9wCeT3tfln3Tz7QQODw9UjBGGjmM1nIljyTTLk9C8of/gCZzRUTCeSGBvQrr8w57r4ig47nQQjTJU2WKb3sMgep6cn5enLvzKzXZ9EVD+ga8a89ijb/hqztrDqMqlQz30QiYQemHR0dBd26uiNdHEwTjtFMTzYG8po4nCzJrmH0er32c9ad7HVfYzDKAQzDGBEmEMN/BkMQPZWzZWI4kJ7M3jRu1tuFMNTnmW5HfeGFF3brfg5l0qPrOrv3n6HE+waqlB+BHu760q2yWPIgdx51PZb05z87V5CbyHZdU1KfJxKoK1aQWLAAEZ5OYunfkQJhhJHA++e3Zh3bqF5q/odlIx1vgkSr6dCWDtnPrpov4VPP47yP/5sjYFNXriRxxRVQ15XxySI9ViPVKVMc23UXhQKn9OLzAyc1asbMLmTA3SsM9Aq2WyZFkiTKy8spLy9n2rRpJJNJDh8+zLlz59i8eTMlJSXU1NRQM3YsJf1UW+EayN9yC8hyVl8YoVZCqxe0WNcBcjRnhdxZkax+NznGVoh69yUZtGy5S9/0ppymFj05X0+au/pvvx0pFrMDEd+d96Bd9S5nLyuX58nr9TJu3DjGjRuHEIJIJEIoFLLdutJryxRFGTI1O4UwSnrcMVjkIrOGMRaLObLXwWDQJkE+n2/AxqXr+oCaIw0lw5neYsT+dQ3WH8mZM2d49dVXmT17NlOnTu3VOIY6GYjFYmzZsgVVVbnooot6RHhgaBOD1tZWtmzZwrRp05gxY0a372fW3OJNSKEtZuA+iIhGo2zevJmKigqmz38rWZZrQs9O1Yikw9TAgvzcc3g++lHnhx0gbQLiIDrNf7PJ/BwAw8B78cV4rrkG77xLkX5/CGrfgD79VgTY/+jTb4Xy2V3nOvYc3l/NxPOPD4LQEbIXoZYj5AC7y+6iYtZypniqzYAtHamALR3ppEd+7jm8M2ea45k5E/knPyl4DTMhxZuRD/4B5aU/mB3s80D4ajEqF6OpVQPygyJqa3vdx6QYqM8/T+ncuZRcey2lc+ei/vSnrtt5PB4qKioYM2YMy5YtY/r06Wiaxp7mZnatXo3h82EEg4hAoM8keHYgn45EAikWQ2pvR4pG8a9ahdTc7C5DzNGcNZ1MZR4nE2LhQtdnUyxcmHfsfR20S+EwZAZpaX8jPTmflUUSgQCivLzgvXO9H4qC8pvf2NeumOdJkiSCwSCTJ09m0aJFXHTRRdTU1NDa2sorr7zCq6++ypFIhLbPfQ7h8yH6+LnqS4zK24Y2/H4/EyZMYN68eSxbtozJkyeTSCTYvXs3mzdvZv/+/TQ3N/d73DZYjrAj4dkczfT0EfpCzpaJoUx6Wlpa2Lt3L7Nnz6ampqZXxxqK8xRC0NjYyJkzZ3LaUReDdLezzMaf2uKNGJOuK3CEvod17y644AJbLuKQmaXGBmR/5qsD9ncdrKkJdeVKJLf79xKwC7PBaBNdhAfMYBMgZcluZWL0C7+CPmOlKWmrXuogPG4220Lyc3bmN9nfHGTOhZcRDAYRklR0wCaEMOdwyy0FM0P5oJ54Hv8Tt8CmpJnhEl5iTxYuUO/ruqLBRE8lWOm6+smTJ6PPn8/x976Xzt27aSothbo6ao4do7q6mpKSkh7/8LoG8plIa3QZm/wQ/hfvhrEeCOrEFj3hmuVxzYooimtWRNTWEtu4Ef9tt4GiYKSMDAoF331daF+I1PT0fN3JKrqOIRzGv3YtrF5N/OGH8d1zT7efp0y3rng8TuL736f8c59DVxTkRIKz992H+q53MXBr88VhNNMzfJBujT1lyhR0XXcYuaQ/h8XUnXUHA53pGUkYJT19gL6Ss2ViqJKBQ4cO0dLSwuLFi/uk2/FQy/Sk21EvXbq0Vz9C9txcAnZ160oSY65wLbLvDwghOHLkCGfPns26d5YFs1UzY43J7bN0SI2NkO/l24GT7JCS8vh8NuEBuqQzdXVQPhsjnexY52rdRmZy2kDlTGuCRZdc1bXQYPUOWrnSUdOTSWCsnkNSYyNUKTCeLnKWPp5CiDXh/8sqpE1JsBV3Cfyrbut2zcVgWDX3FfpKgqUoClUzZ1I1cyb1mFnJUCjEoUOHiEajlJeX28FEdzTmWYF8NJoddCcSXQ5rt98LqheSCbPWpN6dwOYK3pXt2zEWL87aPp0YbG1p4cKrry5q/H0pUyyG1DjONy6IVBrOcrFze14zJX1Sx2tZTUvTx+C77TaELCN3dpoLIWFTZutbty5nxrY7c/d3dFBz//1Iae+bMQ89xD8WLSIaDDoK1Qc7kBzN9AxfKIpiSnRTi8CZdWclJSUOa+zeYKBJTyKRGLJOdt3FiCU9A7WCevr0aQ4ePNijAv5CyGoEOchIJBLs2LGDYDDIkiVL+mxFaiiRO6uhajF21MXADqwjR7MK75E8JqEYANKjaRo7d+7E6/XmJnJWzUyhz9IgpkwBTcv9PWBccAHyoUNm7YSmoX3hC6hf/KJzwwLSGfnYc6hbVoIRc3wuCY2ZF12NlPFCNq67jsQVVxSs0RFCIAVehQfCtrU2m4CGAtKhVP0PkyYhcxSalawaJlT31X7H+NPeU4XcsqR4c7eaUA40eiSJKuL9FggEqK+vp76+3uGudPToUbvAvaampqjVVNuooKGBwPXXdxlvkCLjuo763//dlWFIwa3WxN6vtpbYQw/hv/NOhxmI75570N6Vex9RW0vy5ZcLzj9zP6BLhpZOLrr5fBRDokRtLUr8RbuZLm0J4lVr0ZbcgPLiiwXd3Xw71uA5ssn+/+TUFcTnP+YYQ+jCC0n8/OdM3bDBJjxAl4NeOhIxRDCIvHVr0cTPjYxLXi8LystJLlpEa2srzc3NHDx4EI/HM6iF6qOZnpGDzLqzzJ5mFRUVtjlHdwnFQBsZjBTnNhjBpKe/0R9ytqGMc+fOsXPnTs4//3zGjBnTp8ceKpkey466mIaqxcKam2tfnBw1Mn2NovoKxZvyZnRyoq4O/ROfQHnsMffmpoB84ACJl19GCodtAqLV1xfMxKSPTd16S8pJzoQAhORDX7oJyZ/jeayry5upkWUZKd6Mun8dUprORawATV2fPZ7UNZJ+9yrqLeYqtCeZJPaVh6BWzyqLQtML1lyora14jh5Fqq/PKw1TTzzfFXga2fbJhTAQGaTuSqJ6ElCmuytNnz49azW1mMJiUVuLqKw0zQtiXc+UBJBM9ijDYCxcCGVl0JGW1pRlpIYGRD5Xtm5eg1zEuKfPRyHTDUcz3b9G4WnwKV/CJx4FQ5jW2zmkZ1LHa3iObHK8FzxHNpGYerMj46NXV9Px5jfDo486T65r8GHgB5gLCjpwmUbpZW8Gjzcn0cqaYx4yrigKtbW11KbGnFmobtmsV1VV9bhmtTsYJT1ODKVF394gXcKbbsHe2trKsWPHAGwCVIw19kDX9ITD4RHRmBRGSU+P0F9ytqEIIQRHjx7l1KlTXHjhhZSUlPT5OazmpIMFwzDYt28fnZ2d3bKjLgY2ofONc6+b6ecsj2W1OW/ePCoqKtzHmKvWqEgiZCxalJXkcCCRQPn5z9HvuadrnyIzMYCrPTVKCdqlzyHGXpXvzHkhSRJK7Fj2sUvKEJctcmxrXyNU0DuQFgIvpYK9T91L7PcP41+x1lnT88STeQNK9fnnmXvbbeD1msFj+g9dGTBVRjrUAL6J+F+9DUnE7aamheyTM88zUM0+eyPBKjZTkb6d11frWE0Nh8OEQiF27dqFYRhUVlZSU1OTFUjkre9JXScHCmWs3DKekQgl119P7Mkn0ZYv731j0Vw1U5fO7yImPXg+8p7TaqbbZhIeElYrrXi241wGMVTatrgeU2nbgpYucxMCvbo621r90bvwjfk6LGk3Zad+4H4DKRGDaKxr/gUkpN0h41ah+oQJExw26ydOnMAwjH7v2TIqb3NipF6PTAv2ZDJJa2srZ86cYf/+/fh8Pnvxxq2OcaDlbZZT3UjAiCU9/fWH0p9ytqEGSxLl8XhYunRpv/2RKYpCslCBcT8hHo/T0NBATU1Nt+2oi0F6FitX3Uzxgy0+GyOE4ODBg7S1teW3Es9Ra6Qlz6FuX1eU6YK8bVvesUiA8sUvor/nPTA7rVanQCbGnotblgyBqFxUcF8HMq6fJEkkffUux9acGTjHNQJ8wApMk4ZU/Y+RWEjkc68hfbwBqQmM8xbmX0G3Ath43K5tsoPIS1LH1yOUnHkfnMVsU5+OHPbJmYH1QDT7zERPep0Um6nw7HkG34G7QfGC0BzbZRYWa5pGW1ubTfz9fn9XIGEFwrfdBrGM5ry6TuyRR/Dfe2/RRfzC5XgSQCyGf9UqYu3tWcfrLvHMVTOlHNiSum5pn+ex1+4ObBc7Fwv6LGQQQ71yietmmZ9bgW0mYaYMfL/fAOWY/xx0GUORNT49IeOZNuuZPVuswLSmpoZAINAnvx2jmR4nBqMfzWDA4/EwZswYW0WTXsfY2dlJWVmZ/e7yer0DTno6Ozv7ZcF7MDBiSU9fYzDlbIOx2tHR0cGOHTuYOnUqEyZM6NdzKYpCLBYrvGEfo7W1ld27d/eJA10uZEn3CtTI5DxON5zfkskk27dvJxgMsnjx4rzPjmsWRVJRG9YgGXEKmi40NaF885uu0jYHDAPvsmV2U9Is5OmRoylVNFbfw3nNDyEp3sJZMhdy6Hb9JGkJulpdMAPneo10TFe6DuyO9sJXi5hxJcwodDFyBLB+PyJowMoEktdaUdfcm7i42Ce7ZXSMadP6tdlnX8AhocqTqVCffwafdKcpRdTjObezt1dVh3TJ0tTv37/f1NTPn0/d//4vdT/7GYHHH8/q1RN597u7FSRry5cTraoi8KEPQSTS9YWi4L/7bpPgZhLPbsh3csm09POWwLbi7LW7C7uZbvttoGeQQ48HoSiO6+boSVQ2i+TUFVk1PenSNnD+vmUS5vRGvoxNgDBIcwvplm13bxuPZvZssQLTgwcPOgw2iq3RcMtsCiFeF0F+sRgsa+bBRnodo5VxDIVC7Ny50yaClZWVA0Z+RuVtrzMMppzNCpoH8g//xIkTNDY2smDBggF50Ae6pqev7KiLgSzL7lms7tTQdMP5zSKr06dPZ9y4cQXH55pFMRKpID/NXS3DdMEKVKTGxqzaCAFZJEgCsynpzTeTWLDAkfGRn3sO9ZZbHPU9FjGKRCI0NDQw7fyPkLz4xoLXzJUcjrnC9fqpF/weIQIFM3CidAroUednHqDDD8RAkrI62heCawArSUR/tonAsVUgIu77AUi+LPtk6dRr+B+7DUmNQ3taYP2Xvwz5poy2hCpPpkJqbsb/2N1IqzN3VorOaJSUlFBSUsLEiRNtTX1LSwsHr7ySwMUXM6az0+wJM2UKEj0Lko2FCyHzXRaPm38jLm6F3UFOmVb9LGKkkYNUpqyvzC60+uVE/uVy1MQz+NY95qin0efPR9myBX3JEsSsWVn7xuc/RmLqza7ubfa88izqpTfyFYEpKJ4/9Zltd2+RabBhSeHSazSqq6spLy/PIjK5MpuGYYxIOVdPMZr5cmYcp06diqZp7N27l3A4zCuvvIKqqnYWKBgM9svzM0p6hgH66sYPtpzNcjYbCNKj6zp79uxB13WWLVvWLTvY3mAgSU9RLmZ9CLe5dbdfj3s2Jtv57dSpUxw+fLh7ZNVXl53pWLAedfta53ZppguW45gkSTlrI9yIDwDxuDPjk6dHThOwb98+5s+fT3l5OTQ1QSMwBTPLknVsd3KYvOQnrtfPmzyJIcbb1yFvBk6SnBkXrweEYUuYoHuyMYddr6oi67qZmZl9GRw33LM7ALKPyGV/dQSP6onn8b9yG6yOd7nPvYQZWIfDRdUzDKYzXDGNQKXGRjjnASXu3E7vWUYjU1Mfj8fN/hotLURefjlLTlL0XNKJiRDms6EozswP2MRTOny4W+POJdPKJAd9fQ+Fr5bkR9ehvfNG+9zKiy9SetllBSV7omyWo4Yn6/sCSgbhq7Xn05e23X0JWZapqKiw6yatGo3Tp0+zb98+AoFAl7RSjuTMbI4G+U6MXo9sqKrqkOpa766jR48SDocpLS21v+uLliIw6t72uoAlZ4vFYoPqzjZQds6dnZ00NDRQX1/PpEmTBnS1aaDmGA6H2b59+4BI9ixkkZ4e9Osp5PzWWyMGt0yH5inPKfmy5iTLsunEtn496u23O+x6bbelDNgZn49/3DYycJNfnfi//+Nkfb1dj+SaDcowQshFDs1l++zrl/TWI+eTF1m21NWtIPvNDJgNr9nXpznts27KxrTlyzk8fTrloRCVixbZ+zkkPVrUJFyKv2sFPy2AtKVhxMGSXFv1RqnAWlu8OG+g2FtnuN7CllDlyVSIKVOgVTcJ3QpAB6FAfO6j7g1Du2ka4PP5GD9+POPHj3cUsO/YsQMhhB1EZK7au51HW76cyPz5lL7pTQ5SLACCQbNmyCKe3SQ9kDsDlU4Ouotir5d17r6sFeuufLu3MrWBQHqNhhDClsLt27cPb7iBi5CdwZeV2RRjRjM9aRglPe5IXwjPfHdFIhFCoRB79+4lkUhQUVFhyy57upDd2dk5mukZyRhK7mwDQQjOnDnDgQMH8jp89ScGItPTH3bUxSBzbsVmbRxwy8akSEgikaChoYHq6ureGTFkZDrySb4ye2BJTU3dP18yibRtG2LRImemqAyM8VFiNR4WL15s/uC5ZYNuusl0O0vVE2gbN2Jce4U7Oaxc5Hr9hKjNWVPhIFm+BHzFcKauJB1OZ+yUqu0pFlK8GV/gFInZ05xNITNW7YGcK/hu0jBhABN9xNZ2ZXRyBYqu9TSv3kqkfL6rFKmoefXApaxQpsKRQblfgYoksbWPoM25IetYvXWryyxgt1btT5065Vi1n/DnPxP81KdczyMfPw6q6pS0BYPE1q9Hf/vbh1TQrj7/vHldFcUkZCm3uXzoq0a0MHIduixIkuSQVoroeOQ/rnFsI/QE7Vo1uq4PiDX2cMHrtaanEHKpfyRJIhgMEgwGmTx5Mrqu29bYjY2NSJJkL+CUlZUVTSjD4TBjx47t62kMCkYs6enpS9SSsw0WAchEf5Ke9AzBSM1m9acddTHIJD097dfjRkKs3kkzZ860i2v7FDkkX445NTWhPPxwtpSt2NtZV4e2cSPqypWISwykD8eRPApzWj6EdsKU/bkGWImEYxXdlMTty0kOs64f4Dv4DzT/RGCic0wuJEts8iBu9YOcZqSwwTyvLWOS5aJre6zsylQUJKER1x7GqFxoB/yZq/a5VvBdpWEBH5Gf/xUxvjBpca2nEXFK//ImYou+lZXxKSSD6w3hKJSpKEba1B9udZmr9p2dnbTt30/pHXe4mhMoL75okohMcxZdzyI8g92HRGpuxn/rraZlegr+W24pbAPdg0a0OY81wklPJqTAGOKLnnRkNs9OfZBjTTFaWloIBALout6n8qThiteLe1t3Uex1URTFJjlgNphvbW3l5MmTdHR0OGRy+RwIRy2rRyAMw2Dv3r3E4/Eh1Wy0vwhBLBajoaGBuro6Zs2aNag/Ov2V6bHsqGtrawdtjm7ubT3u15NGQo4fP86xY8f6rXdSPqRnepRvf9u5mg1mj5k6TJvbjsy9U/B4zCwPZs+es3MnUbvnamQJJGKgg7plBYkxV+TvqZJ2PKmxEWNJHlOC1PWzaqrqUZAMDd2zyVFT5UqytgVITvh/cH5V13EnQXz+fHyXXGISsCKD7PTsirVW599xJyhlWTbMhZBTGlYE4QF30iQBGPEsZ7RCMriBsMcu2EyzDzMQrsdPNRks03Vkn8/x7OuSxOH//E/m3n8/UoaxBz7foBbd54Lc0ACJhPPDRAK5oQE9T1PV7jaizYfXG+mB7Mxmqa+WOcDBgwfx+Xx2sXoymXTIk15vWY9ReZs7epoB83q9jB07lrFjxzpklwcOHCAWi9m1jJnNeDs7O0drekYS0uVsc+bMGVIv4P4gPc3Nzbz22mvMmTPHXgEYTPTHHAfCjroYuBG63vTrMQyD/Tv+hhI7xtIFl6MOgne+PaemJpRHHunK8pQBbwXeA2g4i+pTsNe1JQn5xRfR3/c+jhw5QjS0jzGeAGhpAXg4hvKLr2O87Vq0jetRV641A6xEAnTdSYTSV5nzmRKk1VRZPxlSRk2VCAaziVwyiZi+CKqdx5XCYcgIfgsF2W7ZFQlANxmiRTbooCiZmO67nM66HyGNAWP8wm7Vddik6dVbQcSdGbs0Z7RibKX7m3AUNZ8+zEB09zyKEEyaNAnhppvPkdEZ7N+aXHmmYvJPfWUq8HokPeCe2RRCUFJSQnV1tUOeFAqFOHLkiGPlvr+cuoYSRkmPO/rC3CpTdmk5EIZCIX7+85/z7LPPcvHFF3PVVVf1yr1t7dq1/Pd//zder5cZM2bw7LPPupqCTZ06lbKyMhRFQVVVtmxxb27cW7zun6bTp0/z6quvMmfOHKZOnTrkXiJ9SQiEEBw4cIDDhw+zZMmSIUF4oG8zPUIIjhw5wr59+1i8ePGgEh7IMzdfHaJ6SbcITywW48ifHmLewX/mgtN3UPK7C5CP/aQPR1scbPc2y64azIaaXwPeB3gxi+pTTTxFGQhJsh3dJEBKJFBXrmT3n/9MNBpl9pJ3ZBgFmIkwRVuP5y//hOpdi/bX9SRfeIHE/v1o3/kOIhBAlJcjAgG0jRuz+vu4jt2uqXKeSIqY9sHyc8/hvfRSkGUEIPz+vMfvSZDtKklLh+xB/e9nTPvka6+ldO5c1J/+1HVT9fnnze3+5aMELv4Ayn//KfdxMyDFm5HbtqLXXk7kLX8F2efcoDOM/PsGc1uLqGWMU4p22S4PFOHIBysDkf5s9EeGJdd51KVLkTPe19bz7r31Vo5u3Up7e3uPZG1SczPy1q1Izc19NAsQCxeaRDUdHo/5eTH719ZiLF7cq+v7eiU9bsi8FhbJOe+881iyZAkXXHABXq+Xo0eP8vLLL7Nr1y5OnTpFPHORZoRglPS4oz8cfS0HwmnTprFy5Up+9atfcemll/KLX/yCX//619x7771s2LCB7du3d+v9ddVVV7Fz5062b9/OzJkzeeihh3Ju++KLL7Jt27Z+IzwwgjM9hV6iQ1XOlom+Ij2JRILt27dTXl7OkiVLhtSPTF/NcaDtqItBXxG6UCjEgV1/543n1iOLGGipWpYCzm/9AWtOdpBbhumo5XPZWAepDkRMAVlxZEQMw2BMZyd1F1xgblp1N8qpLyBpmG8miVQjSjMDou5fS+Kf9pk1OtddZ7u/uTU0zYW8NVXptTzWV0KQeOklR1+hdESDQU5/9rNM/sIXkDwe0LSCQbbw1ZKMLMcj/wBJBwI4TRL0OL67H0OKxvLKxHoiJ5PizcjnGpCb/4Lv8LfMGqWUVC0+/VF8u+4wx6SAtAn8DfcSufzdiLLCttLdlTxlGmL0FQbK1li//HKiP/qRSY4XLuxy33viCfy33gpxZ+ZM8nopa2nheCBAR0cHpaWlJJNJ4vE4Pp/bH08XemvOkAuitpbYxo34b7vNYWQwkDK8UdLThUJBfqZTVzgcJhQKsXv3bjRNo7KykpqaGioqKkaEFG6U9LhjIP5mqqqqeP/738/73/9+PvKRj/DJT36SI0eO8PDDD7Nr1y7mz5/PVVddxVVXXZXXDffqq6+2//uSSy7hpzkW8AYKI5b05MNQlrNloi8IgSX16reC916iL4KfwbCjLga9JT1CCI4ePcrp06e56PwqpJAXjLQC6ULOb/0A+35ZJgQP3gRGwimNases6SlP/RsN4prjOHI8Tt306fb/63NvQvnwI1AWMzNFdwDpaxGZc62rQ3T3eU6rqRKoYCTRUzVVUuOWLnmWVZcUM3vduD2dVrAx8b3vZefSpcRfew190iQqzjuP6s5OSkpKXJ3MpOZmPLf/xEw41WH2HfooiBTRQ2gw3wN/TDuZi0ysu3Iy9cTzKRmbmVEza3dSPYa2raKz7kdwfxD84a56rPJUrdTixRm1Qwni5XeZ26TF60Olj0p/2xrnIyHpltXpJF/SNKovuoiq2lo7YG1oaLAD1qqqKjtgzbTF7s9aqcG+Z6OkpwvdCfIlSaKsrIyysjKmTJmCruu0tbWZDXcPHsTj8dhSuNLS0mF5jXVdH7B+gaPIjUgkwsyZM7niiiu44YYbMAyDhoYGfve73/Gxj32MZcuW8cADDxQ8zjPPPMO//du/uX4nSRJXX301kiSxcuVKVqxY0dfTAF6HpGeoubMVgqIoPU5dCyFobGzkzJkzXHTRRQQCgT4eXd+gty/jwbKjLga9IT26rrNr1y5kWTYzV8mWHjm/9TUcJFUImKxDusnQ34CnMWt64kDK7TmLOPh8yJs3Y9TUmJmaujq0DY+hPnQXxFVQIk4iJZKIaBBpy5ZuZXcyYdVUhY6+QrtRzdRJS83DW5mrS7B7waCEkQLbECxxHOPYsWOcOHHCttWWJkyAZcuIRqO0tLRw4MAByv/nf5j7+OPg9SKlMkDa8uUpsuKFjrhJGpqAD5vteEzoiA/psJkuIwgXmVh35GRmTc4qJJHI+g4wpWpjMHvhpHGo9ONZxdfqfz+D7+7H8MW/ji+5ISvrYJO7xkbH/w8XSCdeQzmwBf28JYj6bEOIYkiImDWL2Le+lTPrZRki+P1+LrzwQjRNo62tjbNnz7J//36Hq1JwAGqlBrP3TX+u5vfEPn0w0RsCqCgKNTU1tqQ7FovZtUCRSKTHDXcHE6OZHncMNIG1nh8Lsixz4YUXcuGFF7Ju3Tre9ra3MW/evKz9vvzlL3Pttdfa/62qKh/84Addz/HXv/6V+vp6zp49y1VXXcXs2bO57LLL+nwuI5b0ZD4Uw0XOlomeZnqSySQ7d+7E5/MNGalXXyO9gexg2FEXg56SHqtZ7MSJE5k0aZL5YW+c3/oQ6UYG6pqVSI+kAmUFaMMkPC6xddZrOh5HXb0aPvlJs2bmUoHqXQuf84KRQB/3MZSzz9lz1RMfxTvvUmeD0uuuyz5RMfDVoVVciNbenhpLE5LSiLbxs6javaasLjVmdc9qEudda8rqUu8RTdPsv6tEmvtVIBAw75nfT+lXv2paGqcWLby33srR88+nsraWknSyUgdCz7g+3gBiogEnfDllYvnkZJnBnhRtBCmP3MVIYoxfWFie1gG+VRtM6R2pLFFawC/Fm21SRNzbp1KsgYBv4xo8922ym+smH1xBfMVjjm2KzbB1J4Oiqiq1tbXUprbp7Oy0XZX0lhbeGI/juHsDXCvVn+ShvzI9/SUJ7E/0ZZDv9/uZMGECEyZMsBvuhkIhdu7ciWEYVFVVUV1dnZVZHEoYJT1DA9FoNK9L7O9///u8+3/3u9/ll7/8JX/4wx9y/q3X19cDMGbMGN773vfy8ssvj5KenmI4ydky0RPS097ezs6dO5k2bRrjx4/vp5ENLmKxGNu3b6e2tnbQG8jmQ09Ij+WuN3fu3CyXk944v/UVHEYG4xQzI3IK08hAwpXwWLDse+16h45Uvc7qm+EbElKadE85+xyJt/4dqfkYvHIMz013mv1ErAalK1eSuOKKHmd8wAy4LAtrJA+oMdN5Lh3RBNKJbcQnvsW2QLdMT3LJMt0CY8nrxXfqFG2//z11WtdJxFkZKaAAaURIgcjP/4p0Mpw30HQLrN2CPf1dl5vMKnP+ALLftLn21RYM1PMF/Er8RVP+pseQHgE2xeClvret7kukB/PEW/Dctwkp7fn1fHoTiXfe7Mj4dCfDli+Dki/Yd7gqzZtH62OPUb1mDYaiIOk6Jz7/eVSfj+AASMP6mzz0B+npjiRwKGWD+osApjfcnTp1qp1ZbGpq4sCBA3i9XjsLVFJSMmR+T0dJTzYGo7dXb5rE/vrXv+bRRx/lz3/+c07iFIlEMAyDsrIyIpEIv/3tb/nsZz/bmyHnxIgmPZIkcerUqWElZ8tEd0mP1b9lwYIFI6aZVCZCoRB79uwZdDvqgmjfi9ryMr64B1MzlR9CCA4dOkRLSwtLlizJXdycz5J5AOAwMjitm6viR4CWIg/gRgLL45D04VjOljwov/oZyk0PuvfpsVbXe0J64k14ww14ojrqLtPC2tJ1iUzlhwrRo1G2ntpcdF2cW2AsaRq106ZR+tGPIqWRHiOhcED5BOeJJ02ZWapXjxg/C1HEmoUVWEvxZpQjf8B/zyoXE4RdxBY9if/VW+yaHlCJz7oXbcoNzkao+QL1XAH/hCD+V243Sav1u7YC2IVZizWAttXFwgzmV4GqgKaTWPUB5/MHoIByYAtaOunpwx416cgVfMuyjO+jH6Xzne9EamwkNm4cMVkmdPQo4XDYVbaUlenrYWA/IL2XMgL9viAhxWbjhlo2aKCC/MzMotWv5dChQ0SjUcrKyqipqaGqqmpQFRSjpCcb/eHclg+9JVm333478Xicq666CjDNDJ566ilOnjzJTTfdxAsvvMCZM2d473vfC5iGVB/4wAd4xzve0euxu2HEkh4hBLt37yYWiw0rOVsmiiU9uq6ze/duhBAsW7ZsRDi3ZMKyoz579iyLFy8e0t2qlVfvRDn0FAAXA/qrL6Nf+JWc22uaxo4dOwgEAixZsmRIv+gdRgYbNqE+8TF41XDIsyx76qx9wZXASE0gtLgz6ExGUe56GClXY9IeSnyszE4dKnVGnCzn/kRqoKleQ+K7KjvuVFl42cKiFxJyBcZSOJwVjMlKkgnCx6H5f6SzeTdtyUoCHZOpO7mXam87ctn0gr137MahQoZHY7CRrv5IqWBPW2zW5MjnGkzHsYru9fTJO69AOKv3EDqmUcOpgZViFQOpuRn/qluRYl1pHe83vg+ZfFwH/bwlZKIviv/Tg/1igm+LjHqB8WA7eHV0dNDS0sKOHTsQQjD9H/9g0he+gOQ15YXJD38Yzw9+0KPAfiB6L3X3OhR1zCKyceYzsMpsJNtPhK67GKwgPxAIUF9fT319vaNfy7FjxwBsKVx5efmAjq83GYaRisF6Rnqa/Ttw4IDr5xMmTOCFF14AYPr06TQ0NPR4bN3BiCU9kiQxfvx4KioqhkyqticohvREIhG2b9/OpEmTqK+vH5bzlSQp7x/zULSjzon2vSiHnnIE/cqhb6HPWAnl2dbHlvPccJEjpkv2jOuuQ//Rj1ASv+72cRzEqAOzkaltIAA8a0Ak+xUlADwetPXri5O2xZu65IDNLaibVyClNeLMWsfyehH3S1CmIk5p7L/9XhZddVW3F07cAmOpuRmSGf2IdCjt+Arjxt2MmDIPwzBIHvgB1a+swUBFQuPExC/iOe/Dri5MjsahYPZJsrIsHTiCPeGrRR9zZdb+Vmf4TBLk9l3mvCgD+VwD6M55CRXo8PdLn5zeQj7QgEPHBiBraDdfg7rpBUdNj5uZAfRd8X9vsinpsqVp06ahnTpFxRe+gByLQcyUino2bTKf9R4E9gPRe8kiPX2ZVXIj5/GHH3aYa6jf+Y59jWwMcDPdTAwFJzurX4vVsyWZTNLa2srp06fZt2+fw2QjX51HX0DX9aH9Wz8IGG6ZnqGGEUt6AKqrq/us6eVgoRDpsZzL5s2bR3l5+QCOrG9hBdJuL7ihakedC3Joc87PjQzSc+bMGQ4ePDgknedywVHLsncvyu9+l5XVsdzauvXz/RJmoF6H6WgWlUCLum/r86GuXYtWXp7XzMBRr5OMwc90eIfeJcECUAIIYZjNOVPmEIkX38yhP/4Rado0zrv00qICEVuWMyGIFAibRCEjMBa1tcQfXYPvrgdMXwEduBmo9JrkwleLkgxRvn8dkogjY5ogTDz+OV6R53Mu5s2SntiNQ9OyLEIHppbCASMv6bAzRLLH7tej1S8v+J01L/XE8/j/kdoGA4EH1IBpaT1mDdo/bhxyhAdA1GFe+3TokPj4zcRXfSGve1ufjSH1N9SX2RTvyZNmhiczmE8/r6IgDh+GjJ5Pblmr/pLyOcZjkZ4+ziqlk3O5oQH/Pfd0zeGhh/Bt2JD9fkokBjUrORTlXB6PhzFjxjBmzBiEELYU7sCBA8RiMSoqKqiurqaqqqrP7aWH4vUYbAw06UkkEgX7iA0njGjSMxKQi/QMB+ey7sCaZ+ZL89SpUxw+fHhYkQKjemnBz4UQ7N+/n46OjmF3/2yC+txzqB//OOQg5TkzKanvMuVwgJmZsGya0Vy3kQDCYaCAmUG8CXVrWr2OBLwTVyaWuPIfSFoYUTqFTr2UhoYGJl9+ue0okw+SJJmynHtWwRUG/FMCPAFQcBAFex4ltfAoZi+jOsxeRmmNPuVTDaZMLf0cipc5kwPoFRfR3t5OS0uLLT0ZUyExO7NPUtBP54YnkMZVYIxf6D7u9AxRijD5t60iUnt56r8zvnv1VjqPyY5jZm4jZD+di7/XI+ncQEJMXAgrPLApaWd1WOFBTDTHrfUj2UmHJEmIyZNzZlO6W9/impnJRDLJtrY22LaN6upq6v/yFyruuiunpKy7Ur5ujzlFevojq2Sd33/NNc4M0rp1pqFKGjkUQHzt2kEl6UMh05MPkiQ5TTYMg3PnzhEKhWhsbESSJDsLVF5e3uu5jJKebAw06QmHw5SWlg7Y+fobo0/TEIcb6YlGo2zevJlAIMCiRYuGVcCcC5kuZ4ZhsGfPHk6fPs2yZcuGDeEBoHw2+vRbzbqJ1D/69FttaVsikWDr1q1IksRFF1007O6fJUVRV65E0rT82ZwpQF0AVNUmOjm3L+ZF7rKNJVnJ+jzSaGZ40qEDPwfiIKJgCK9p+10+G1G9hFBE4dVXX2XOnDlFER5IyZO+dwvSIzGkdyeQvCBJUSQjavbGiTc7t739XqQfAxMBFUQc4tMfRvhM57XAlddDPOI8SYoUSZJERUUF06dPZ/HixcyfPx9PcAIHaz+Njg9NKsWQ/MTrP0RJy60E9nyU0t/PRT2R3QXbzhA5PlRQXvmNSbwyvzPilJz6GIHN76X0N7NQG5/N3kb2greyR4RHijcjt211XK+eQGpuRt661ZQS5oDw1RJbtRHxDT/i/lLEN/zm/3dz3H0xZiubIvx+RGkpwm9KApUXX6R07lxKrr2W0rlzUYvoZG4fKxBAlJcjAgGSK1Y4/j/+rW+xKNUHw9feTtmnPoUUjSK1tyNFo2adS8a1E7W1GIsXFyQE6vPPd3/MFulJH3swiPD5iD30UO8NIqwMUjoSCejsdH7m96PdcEOvztUXGMqkJxOyLFNVVcWMGTNYsmQJCxYsoKSkhJMnT7J582Z27NjByZMnieXJPObDKOnJxkBL/sLh8IgyxRrRmZ7h9PLIBVmWHZrKpqYm9u3bxwUXXEBVVdUgjqxvkU7uLDvqurq6IW1HnQ/6hV9Bn7ESObSZrY0qF154PdBlJ37eeecxZsyYQR5lzyDLMsqxY/lJigzcBCwFjkbhwfzHlACRQe5dJXOZWaVoFJHjhSxKp2Q3c1WAP6b+qfey+2v/j/MnvQuAo0ePcurUKadJRno9kJtjXlMTym9/CjckzVqaTMgeW7YGaQHYS1Fbymd0lqL/+4K0moaUEcGKlEwt2GUpnQmv18u4ceNg3J10xj5ItHkPoY4k5x94PxLxrAyOw6UtMAWMjOvTGca/ei0kNfiKs3mQJGH+YqgASXyvPehCjLoyVoUgJ1sIdO5HipehNL+YU0rXHXSnEF6rX07kXy7PWc9U8Fx55H/FwLGqn6mbb2/Hf++9PapvccvMJO65Jyv74vf7KYnHkVMW8hZ0Wea13/wGzxveQE1NDWVlZUXLO3tSk5N+HbTly4m3t+O7+27wevHfey+x8vJeOaqJKVNMkpM+VkDIMsLjAa+3X2R7r0d4PB7Gjh3L2LFjEULY/aZee+014vE4lZWVVFdXU1lZWZQUbrSmJxsDbe4QiURGVKZnRJOekQQhBAcOHODcuXMsXbp02HRULhZWpmfY2FEXg/LZGOWziZ78G7Tv5dyh33LoXB0Ll7x7WL9EJEkiWV+fU9YGwAeBt6S2nwDCkhDlQyCASCRA14uvBfL7kcJhVwmd3cx1ywoIx5BU4PtAh5l9C3/8y8Sqp9pZRV3XWbJkif2DIh96GrVhTSpbpKMtWI+oXISIBpFOhZFefRV13TqYIcEduJOeDBLgkPCkpHySTzOlTI2NXUTSqm+aWkrnV36IUX9l5pFdrkUdgYl1TGrbinzEB1pXIKsJicM7X8Rf/2bb2lj4aoktesLsrYMM0QjSJuCkKR0U/yEjluery9KJT7sD3+FvOQL/YsiDeuJ5Jm9bhUBFPqwBBpJI5CVphdCToFv4anuclcolDUw/Xj6TiKxxp62G+1NBvwPdqG9xqyVz289NUqYYBtOuuIKQonD8+HE6OjooLS2lpqaG6urqnPr+ntbkpJMeqbkZ3733Opr69oWjWuKDH8T7ne84n+VAgOj3voeorBwSPXpGGiRJorS0lNLSUiZNmoRhGLS1tREKhThy5AiyLNtSuFzEejTTk42Blrd1dnYO63glE6OkZxjAMAy2bNlCVVUVixcvHpaZj0KQZdn+gR3qdtTdxYy2r+H93S+owyzh0Pfdmte+eqhDkiTafT523XUXcx991A6aHE/lj4E3YNarlAMfAb5DV2FOHRABMhQm2ic+gfrVrzo/LEttHwUCmCYHVt2PVQuQA8ak69D3b0eJP2b2/vwIEAN2laEvXIimaWzZsoW6ujq74SikCM+rn0jNyQy+1FdvB+GHRAx+4IMXUw5wjWT1dxECULIzNJmF4SKR4NjnPkd1bS3qj35k1ypBao4HDMR095qcXHDL4KiSoG7qUpo6YuzcuRMlFGJMZyfBeRdRGXmQwPp1cDLtugL83oB3AXlqWI3aNxOZcXu3siVdhMFZT+FARoasGLgG3YqC8pvfoL/97X0a1LqZR2SOuVAmqFABf2Z2oq9d0yC3UYFnwgTGgr1iH4lEaGlpYffu3WiaZlsYV1ZW2kGpCAYdGaNix+wgPX1sZmBn/tyCxGQSY+HCUbIzQEgnOWDKvEOhkP27HwwG7e/TifUo6XFiMGp6RuVtwwQjgRyEQiE6OzuZPXu23UhspEHTNFpaWggGg0PfjrqbSDZvZ1L0F0XbVw8HtLW10dzczLK77iJxyy3IP/kJ6v33OwMVFbNhaSkmYblMRfy7hjQNU/b2cxB/SjuoLKOtX4+6dq3TvEAGvoRJnLxpLsObQLxEYdvqeBPKuW+ayRpLibUCuDtBR00NLadOsXDhQmfD0XgTasMaV3kdUswkAR+Owxa6jBc2gbgVk/xIgKwSn/uIq9wpXX7UUlZGq6Iw9pln8N1/f5ZxQ+y++7pv15uewUkLuEtrplJaA+dt3oz/9tsxFAUSCSTDcDRLteebZiMuvCl5m2MDD4ZlWNAB0s5GmELhQnc3wpCJbsjk7Hm7FcKHw/jXroXVq/u08aSrNDBtzMVmgnKOW9eJP/oovnTHsX6SXxUyKpAkiWAwSDAYZMqUKWiaZr8DDhw4gM/nY+pLL1H/mc+YixAAgQBAUWNOJz29MTNwa8hqZ/6scwEEg6Dro3K2QYYlzR03bpxNrEOhkE2sKysr0TRtwIP8oQ43w6f+xCjpGcWAQAjB4cOHaWpqoqSkZPhLvXLAsqMOBoNMnDhxRBGetrY2Wrb9lDku37nZVw91WI6B4XCYiRMnUhKJIDU2Ylx5Jdx/v2NbEQUex3zD6MAUzazzWQ00A3/KcG8zDIQsQ+b994FkuZyl/h8wiUtjKWLRorxjltq2YXpDp51LB3GjRtvJ/6Cy6hon4SHdACFj1TodVuNNKzOyCzAwJXQAQsO36x608e9yzVZYciMRCuHdvxXft9YileHMtAD+L32J2IQJ3Q7WtXqzEWlmBiY9ELSuSt4uDJbM7h1exHt0bI2i5CF24UbbfKE7DSVdCQMehKyYRghGgvj5d3VrvpCRtVAUCIcdTn992niyA+LKGnzt66HSmyXtKyYTlFnAn5lt0ZYvR3vXu3rVALVYdKfnkKqq1NbW2otw0aNHqf3MZxzyPEPX6fjLX5AvuKDwudNJTx6L7HyucG7PoDFtWnbWKBgktn59n2f+RtE7pBPryZMno+s6bW1tnD59mldeeQVVVW15pVuvstcTdF0f0PKGkSZvGzkR5ghCMpnk1VdfJZFIsHTpUlRVHfb9htxw6tQptm/fzvz586moqBgxcxRCcPToUfbu3cvEBf/iuk0uW+uhCstxzufzMXXqVMr/53/wzpyJ55pr8F56Kfq//RvC5zNdl7xeU5KSBClqZmek/SBVY8bMB93P4fnEJyCS4Vqm0UV40qEDlZpzBbipCWnLFmhqAswePZ6/vQ905zGlAMgLdOaFv4KP9qz9XA0QMqEATSnC4AcmgJT5Nk0FuflQ0vQLZp+5HtYk4WvAJWnjBKR43NVNqxgIXy1G5WJnjUmmk1UZMD3173zH2gGts39O58U/I3rxz4hc9Rpa/XKTRN2zCml8FEnkdv/KHFds0RMYsh9dDiLkALGLNhJ5227iMz4JAnwHv267znXHIU1bvpzIrl3E1q83V/TTkZJJ9RaWQ5nvpq/BnRA/+0kib9vllK4VyATlGnfnL35BZNcumzQW65pWLPrKIS/9eGWH/xcpwx1NeDyc+M//ZMcf/8iRI0fo6OjI2eQw06bZ7Vrkc4VLJ/LpDnQiGHTNoI0SnqEPRVGoqanB7/ezdOlSLrjgAjweD42Njbz88svs3r2b06dPk8iUgL4OMNBGBqOZnmGE4bgacO7cOXbu3MmMGTNMVya6nM1GSoo3vcfQsmXLUFWV5ubmvE1Yhwt0XWfPnj0YhsHSpUtRFIVTZf/K+I7/6Nomzb56OKC9vZ0dO3Ywc+ZM6urqaN6zh/Gf+5y5sptaRVW++11T0pJIoN9xB8rXvpZ9oLOYZGGG+3mktPsvwOyj8dG4KW3LgFBAu/3ztrRNfu451FtusVd6tcc+i1r1eSTRla0RAiSDrvobWaXuVz/D+/i/du23cSPGddehLdyA+urt2X2CDHM7vouZCLpRgak6jCH7bVpAoiXFm6k5cDeylOhqlroCM7OSnvERwq5p6G4PlEw45EOXpM6np4wmfu0163jCsnmxksmubWRBxWvL2VO+huSEf6Valan0GPj2fAcejZnkVMGUw+1UCtZgaPXLOSvPJx56jQnnvQHhq0WKN+PbvwFJxEAzswb+V1aCnQEqziFN1Naiv/3tsHq184s+qIlxM0vwrduA9s4bHbVPuSSGjvqujGC/O9mWnqC3bnM5jxdWIB52fCdHIsz51rfgG9/g1AMPcPQNbyAcDlNWVmbXbVir1W69adKvRSGDipx1QOFwvzdW7UvkIoWjAJ/Px/jx4xk/fjxCCDo6OgiFQuzcuRPDMByucCNJLeKGgY4FI5FI0e0bhgNGNOkZThBCcOzYMU6cOMGiRYsc6cRcDUqHI2KxGA0NDYwZM8ZhR60oyrDP9ESjURoaGpgwYQKTJk2y53ZszFpKFq6hNLrTzPAMI8Jz8uRJGhsbHc+k58QJRIamWIIuAvT44+7OblFM6+pPA1cAL+aXVukf/zhKw7cQlwjz+F4gkdrnux7UzV9EK6vHWLAAdcUK0/EpNQZ1w71wL11kApDiwDcxg/hyoC3JxMe+hxRPdO1nNTv1lIHkQVgZH8mDVr8O9eVHzdqWtwHXgyh3zlMAKEEQekEnMylqyejSCvpTsjkpnfTEYohgsCgZWSG3MFs+dPdtsDJm9hSyvrs2Ae/xER/7GXzXfMmU263AlBiSBAFzOx7naOl7aG5upnHfFt7Yth7JS5dz3QrgjjDK9u0YixfnnDuA4akhGpiXXxJGEslIQsr0oFhXt3wyqd6gO4X2uSSGgwH3GqPbuu2Q53q8EuBmEE8DnlKIRMxnqsN8iMfffz/lu3ZhXHBBVrBaVVVVcLW+0DXPVwekLV7crcaqg4mh3ph0oJGLBEqSRHl5OeXl5UydOtVRY3bw4EG8Xq9NrEtKSkbcNR11b+sdRknPEICmaezatQtFUVi2bFnWAz1SSI9lRz1nzhzbwcWCLMtoLsXUwwUtLS3s3bvXtX+SLMtoJdMwxl44SKPrPgzDYN++fUSjUVtiaX83eTJSvq7v+Z7VQ8AngLEgSkC/+j0YV1+N57bbsjZVvv1tpISA7Tjc26QmoCMJJFFvusncOLMPRxNZjmpIII4DoVKoMohX3oWqPA6k7evxIO3/M+qZFUhpEjdhgPqv6yGSNKV1N2NmrJQAwjBA8Zmr53MfwqhcWDDIlZqb4VAbGBkBn8eUBDrg9yMdP17Qjtm5kp8gfv4atCk3Zo1DW76c6JIqAjs+SLp9ngQg4vhOfwlqPOCPZ9uMC5m6+HGqZ16J3NaBFPKBnjYHHaQ68N1zD9q73tWtINO91icD3XB1K1ign0YQgaLISXcL7fNZYg9kkOtKKMMx1F8+S/Jf1xZ/nFSmkco25/HeACwKEm9fhe/zT9qEB7AJilRbmxWshkIhTp06xbZt2ygpKbGD1UDKCAEKX/MsgptIEL+rqyasvzNofYVR0uNEsdcjq8YsGiUUCnHo0CGi0agjuzjcGoG7YTAyPSNJ3jai84DD4QUSDod5+eWXqa2tZd68ea4P83AnPZYpw/79+1m8eHEW4YHhm+mx5nbgwAEWL17s2jDW6kE0XGDV73g8HhYtWpTtFFNXR+PnPmd2Ti8pyV8InwkfZjrkEKB50T//ecS73pXdMV1RuvqUdKS2P5X6d3oWJJFASiSye8lYzmNxzLg+nvr/Dh/a9C+RWPp3OFWLktmVfVEUz+kbwcgwMYgmkcoTJiFJAE8DJ4CDBpF5/03kkp+bdR1Tb8yqo8mEXZ/wLx+Fp3QMQ0XIvpR8DvgyjtoeRCrTlXmN0mpU0lfeJa0dyYjhe+0BSn93AeqJtPqHw9/B/3/vQNr8/yCeMXcLigcqkqY1eObrKB4hcOX1qD/9KSIwBUlkLFSkap10SeLA73/P8ePH6cy8xjlgScKEHECo5QjJj5lGSkM3Xd1y1cSoJ56n9PdzKfn7tZT+bjalv51l/neqjijf8WJPPGE+++XliECgbzJIzc3IW7f2qHarGJiEMmNhQAHfuvVFnzO9rqbkDdfD/2aw86COds1yyFy8ykEKVVVlzJgxlJSUcNFFFzF9+nRb+vzyyy+zb98+Wlpa0KqqCl5zqw4o/slPAuD7+tezan+GOkZ70jjR0+sRCASor69n/vz5LF26lAkTJtDZ2cn27dvZsmULBw8epLW1dVj9JqdjoJ+TkUZ6RjM9g4iTJ09y5MgR5s+fT1lZ7kri4Ux6NE1jx44ddkFirj9WWZaH3Rw1TWPnzp14vd6CcxsuL9iOjg62b9/O+eefz5gxY1y3kSSJ1re/nQkf/jDStm143vMeZ3bH40H/4AfNOh8y+vckQIzHJCKyjPfSS9E2bkT7zndQb77Z9EUWAm3DBtS1xa9AW7AImAS285ioo6u3j5xEXfl5SHwaDCO7Zud64cjw2FBTx0jf9tOAX6b0gWuJfvOb6O9bknNcdi1OMOjI2Eh/AdEIfFkzZXPp7nS7QHRA/NFHMRYuzLva7baSb2ZuYrYkrOQPi5H0VnM6vlRGKY4pTXNcCJ3Y2kfwr7oXfijgQzEzg6NgNi9tiqWyTLtSdSu3mc1frZqeDlACgvo3vpFm4MCBA8RiMVt3X1VVlXOlMlMSpjT/KW9dTE/gJvUyT25e30ISOjuDdKgBaQwY47vXR8mCtZLdXfe7Hp3LV0u8fA2+sw8gpddfxb1F9cBxq6sR3/YipvogrkKtTuyyJxD1s7otKxRCIMuyo5Gl5d5lrdh7zjuPut/8htpwGN+sWTlt6n0bNpi1hiknuT517OtnWNdhFCb6IriXJImKigoqKiqYNm0amqbR2trK2bNn2b9/P36/35FdHA4L5aOZnt5hlPQMAqwO8Mlk0i7kz4fhSno6OjrYsWMH06ZNY/z48Xm3HW6Znkgkwvbt25k8eXLBIr/hQnpOnz7NoUOHWLhwYd6XnD2fujqkUAgUBWE9n6qK9p3vYFxxBcZFc/DccbezcMcAjqdi7FRgoq64Cf32lExNUUwCVV5u9u25/fasLI7dB8QwzG3TVpYlTOcoXZLA40HuiDjrYwwD2tu7jpOOyz0QdBILAWa24VmRktSlkEjNIWwGgIHbVxG54grX4MoR1MbjWbbcUknquqT9jgkDmOAhdut6tBtuAMgbTOaVhukgH/gGkt7adS0lzEavPwR8IN4D+EpBMuwC98jl7zYzSa2NlHxuFRxJueBNB2KmWYG22CQp6i+fxbduPcS9EDDH5p80iYnAxIkTHd3YDx8+jMfjwefzIctydkF/miSsP+pibILYFjWJbB1Os4wiJHRK/EX8TbdDiwd29dwUQG1tLShb7C5ymV1oc27E97HHoCzWtQgQKLIHjltdjaTAfbpJmjXgCWB5YVlhJtxkTJZ7l9WqIRaLEQqF2KcodB46RHlzMzU1NVRVVdmSpb5ubDrQMAxjWATdA4X+CO5VVaWurs5uU9DZ2UkoFLIXZsrLy+2FmaEqhRslPb3DKOkZYFhp1vHjxzN58uSiXnLDkfRYWawFCxYU9QczXIgBYK8SzZs3j4qKioLbD/W5CSHYv38/4XCYpUuXFnzZS5KEEgoh/e53qCtXIqXV0whVhXPn8M6caQb3mcxCkC1FiyVQvvEkUhK7o7u6ciX6qlXZYwVQ1dSxBca11yL/x384jql7PLQ9/TQVnZ3Iq1c76wtyoQz4SDLbdlryknjby8jadtTNK1PEJWU8EE+Y+9UBHTHULc+SfIeZnbLrRSLB7BXyzHOfJVtoHPAR+a+/IsbPsj8qFEzGp92K7+Dj2dc3EcV75kX3eS8Gvg4c8xL99hMYky7r6iOTqoWQmqfAAQMWYru9oYRRSrZjYEr5kv+6Fu0tNzibQ6bVzMi+Wqqrq6kxDKRz54jW1nIkEqG1tZXNmzdTUVFhBxuZi0D56mJ6AhGYYkqzNmESTatG6w2pDYpw3Su28WjecQiB79SpPg3U82WNRG0tsUee7CLOgeJ64ECOuppo1HzW4ubffzpZ604dTTG1G36/nwkTJjBhwgQMw6Cjo4OWlhaOHTsGQHV1NbVVVZT0sLHpUMCovM2JgbgeJSUllJSU2Asz7e3thEIhx3NVXV1NWVnZkLk3o/K23mFEk56htmpiBctz586lsrKy6P2GE+kxDIO9e/cSj8eLymJZGA5zFELYeuClS5cW3SDMWs0eikgmk2zfvp3y8nIuvPDCov5mfP/5nyz45CeRVdXO1tjweFDXrDGd1FwgyCY9ZuCU8aGioHz969kBPCBpmp3dkf/nf0xr67TzKUJQdtllGAApjb8rylREXDNlWROyepgiAH3Op6F8NsZ1s0lccQXKy99GOfIw3Bt3WD5LCvg6HkKL34DS/GKXocD+OCgZP1B+v/k8eD0QDyO9v+uiCADZT2zRkw7CY4/JJZi0DQyyGFvqgL/2Ev/QWyhhW9ZXnMbsD+RVCLx2K7FAV8bCJi1lU0jeuBzP4h8gpeR3EuA7dA/a9HdlkSTHmNJkafytHf9j66BNpaTNwHj4EdTLLmPatGmcO3eOUChEY2Ojvcrfb40IO4BvSw7/CvE0sCgIwSJd9wo0Hi0WiQkTumWMkA+FrJ3BnTgXI6/LMguIx00pavrffy/IWnfusSzLtmQJzHdYKBTiWEsLLXfeyQWPP25meXV9SFtUZ2LUyMCJgQ7uZVmmsrLSjs2SySStra2cPHmSjo4OAoGAq9HGYGAgn5NIJJK3/GK4YUSTHjAfjsEOOA3DcKykd7eb7nAgBOC0o54zZ063f8iGcjYkmUyyY8cOSktLWbJkyYiYWzgcZvv27cyYMYOxY8cWt1NTE8E77zRJjRuxSSS6gqIUbDlaPG7Ky/JBSu2QTJpkJj2LBFkEB68X7VOfQnnkEQxFQTEMtI0boRykSCPaxvWoK9d2raYbhjkWTUO/90NQ9f9Qm6LufXZkP/r0j3f9fzkoyUeRxsXhVszMhy99Bw2p+S/OLEANCGsKdlYIIi/8FSkcJmm8TNmZzyIrpgtafOZatCk3FB08O7IObkgCL8qIh+9AvPx90FvtbJsUBt5Eym46CkZXxsJB3IwEnE4iZb2CZNdA3zUT8soKkDWk1YCSgE1Qt24NZ174NfKMGVRVVdkmIPF4nJaWFo4cOUJnZyfl5eW2lKnYRZS816yxEbx+iKW79pUSG7ce/ZK3F7bD7mbj0XywivT7wlq7WHlXd3rgWNtIjY3ol19OZNcuuzat9LLLnAMYpKyKx+Nh7NixjB07FjFnDs0f+ADhnTs5W1JCNBik6sCBYdHDZTTT48RgXw+Px8OYMWMYM2YMQghbCrdv3z7i8bijN1BfvJeGKqLR6KCTvL7EyL1TQwSxWIzt27dTU1PDRRdd1COGrigK8Rwr50MFlmWzmx11MRjKxM6qTZo+fbrdMLY7GIqk58yZMxw8eLCgiUYm3AKr9CUF/frrUZ57zrmT30/yJz9B+vvfUR98MP8JhHk8bc0a1Mcfd37n85mry+mIx9k5axbeX/2K81QVfepU5Ngf8f5qptkDx5tE++t6pP9qQnnkEfD7u/oJffPH8EDU0SxVCMwePUJDW7wRfF0F01LE6qsThWWYdTgZUOJnMmbDQwABAABJREFUnVmAcuAWP2JbEm7QzdqHgIES3IE2aznR8CwO+hcwe5K/2zUrUnMz6ks/hXbJJFTpc4hjSuae9RB75ElEbS2RaxqR//QZSjZ/A04aJuHJVDLKHuRzDVmkRfyzywCiEZSf/w3jY85+PPKpBhCZwYqW3c/n7iSBY3th6VLHlj6fzyFlsiQnjY2NyLJsyuRqaggGgz16n7pKtSIRpEMJxFsKX/9cjUfpAHnn1qL7wVgr+92tgenWvAoQkUJEKV8WaCg2/pQkiZIpUyiZMoUxmPUPra2tNDc3c+DAAXw+n51FHGqF66NGBk4MNulJhyRJDqMNwzDs7PSRI0fs95IlhRtKz1VfYKjch77AKOnpR1hEYPbs2XZBZk8wlAmBZdnc3NzM4sWL8fv9PTrOUCQG0FXcX2xtkhuG0tyEEBw4cIBz584VVb+Ttb9LYJX+eleee840IFi71g6GtI0bEYsW4bnuumy3NJdjAKjr15vNSb/7XcdxwKz3ASAaxRCChTfeiL5pE8Z110G8CfX/bkHSo5iNfUB97S54UnK4OgFwMmzWddh1KsCPFbTbPo8x/0qodzaRFaVTwHJ2e5nsTI+kotVdgW/P57o+awfGGXCLiiRSRd8k7IyKJAVIypUYlXOd17RAjYX6/PP4b70VkgmTuKTXpCh+ohc+hQhVIL670LnKP+8O+PhGeCTWRUDSYSRTErsM+ZbAzBqJrmslbQLfS/chjFK0G2/sGtc9q+DRHMe3oAPVQF3+4CBdcjJ9+nQSiQQtLS0cPXqUcDjsyAIV+yyL2lpiDz2E/8477edOonu9hbJc5v7rRUpvn9tjB7bM7EtPCFBPGrLmI0qFskB9RdYsuM27p9fCgqIoWT1cWlpa7ML1fLVkA41RIwMnhhLpyYQsy47sdCKRIBQKcfz4cTo6OigtLbVJUE9joqGAwVZJ9QdGPOkZDHmbEIJDhw7R0tLSKyJgYaiSnmQyyc6dO/H7/SxZsqRXL6ihNkdLkhiJRHpEDtIxVEiPpmls376d0tJSFi9e3LMf2Lo6Yk88gffWW82aHqv7ugVFQSxaRGLfPjtYoa4OacuWrBXlXGeXABIJlGefJfHSS0jhsH0cgMSCBXiWLUMGlJT8TVq5ksQVVyApjdChmD19LFeuSBxqvA7LaRu7ACuh1Ah06Kj/91kw7kfbuNEkUhZ8dWiLN6L+8Wb4VhzpImzCJBSIn7cWUTaL2OSH8L94NxwBvh+H8yW4M252rk+fZ7QRyTsn6/1UqMZCam7Gv2pVl4FEqm+QmAFUQlL+CPp5/+J6bUVtLfFH1+LTvuQkoAKQPcTnPoyoWJgt3/ICJzHrf8qw3b8kwH/33UTe/W6AVJAcg43mtRE6UOoDjC7CCKCA+CeF6JjsuiXHNcoIer1eL+PHj2f8+PEIIWhvb88qaK+pqbFXW3MFzcbChVBW5tpEs+gC/JTBQjESMdf9XWo43O693g1i0W3ntDxESd66taBcrieNP93uidu8EaLPrbwDgQATJ060C9fTa8n6IovYGwzlIH8woOv6sLkeXq+XcePGMW7cOIQQRCIRQqEQe/fuJZlMOqRwvXFeGyxiPJLI+IgnPQONRCLBjh07CAaDvSYCFoYaIYDu2VEXg6FCDMC8hw0NDVRXVxdd3J8PsiyjZTbsG2BY9Tt9cb+M667j5ZoaFssynve9z5k9CYeRtm1DLFmCSOul4bqiXAixGMrPfoZ+773Q1IS0ZQvG5Mmc2b+feo8HOZFek5Hq+r73VVgVNqVdGvA+kM4DcSyRffw0IwKrb4n0EhAxrZnVFJGy6oNE6RSMSdeRPL8Kj3c5vBQ3SVOqTsd3x+PI/2jB89kfgJxGCI/Gs5t8GlGEkp05zBdAgylHktraTGvvdCRAeswch0f7AYnL7skZjGpveDe+//uS88MkoHnw7boH4SnLkG/FQUhQE4M2ssljWpNUO0hO9UhiaimdX/khcn0r/ldvMeuDBCBUmj+wAV3NLYUtRACorXUWtJ88SXjnTpqP7+aE1ELt1qNM/fLXzfFpmtPJbMqUoptoFkJfWSW73vuVK7sa9RYZ+HeXiOQiSj2RyxWC2z31VFRkz/u22wDM7GwfWXlnwm21Pj2LWFZWZq/Wd7cOtycYNTJwwjCMAbVm7itIkkQwGCQYDDJ58uSsnlOqqtrPVXfJ9UAT49FMzyjyoq2tjV27duVt7NgTDDXS01076mIwVEjPuXPn2LlzJzNnzrS9/HsLSZIGdW6Wa+CCBQv6xIVFlmUSFRWISy5Be+wxRy8dCVDXriVx7bXOBoJ1dWgbN6KuWGHaTXd25sz0WJAA5ZFHzNqCdevA40HE44jVq+0Mj41oFBEM4vnEOqT0r/7d/diiDFhptuCxP1sB0i5Mdy8wHeR2fRul7VGzlkck0RZvxDj/CtBTPzwd5j+SF0jE8Xxhk2m9nY4OumR0KiYBkvxIehhJlVC0EHLbVkRgSs4AWn3mGXwbNpjfJRLOZrAWTqb+XZ4/4JZOhuGHfrvpKApI3wXe1gkzUmYGb9tF5G27TPmWEqT0L5fBTlzrmND1rkA4PUjuAA4YiOkL0WpridRejnyuweQ8FQvpDEvQ1uY+RjcCcMst5rPjQgDU558nePvtVF0qmPShGHj8UB9DWgS8ZJJy3623cm7qVIIAU6cWJQWzsxITgkiBsGvdVU/JQWaQ63rvk0mkZLLfm226EaWeyOXyIRehD37+89nzdgt2+7nnTmYWsaOjg1AoxM6dOzEMg6qqKmpqaigvL++XwHM00+PESLkemT2n4vE4oVDIJtfBYNAmQT6fL++xBrpHTywWG1EmBjBKevoEQgiOHj3KqVOnuPDCCykpKSm8UzcwVEiPZUedSCS6ZUddDIbCCtfx48c5duxYn9/DwSJ0lswyFAr1yDUwF9JJnFi0KLdMKJM0WqtGqcahthtbOJybAKVbYKf6gkz5yle6jtU1KKRjx8wePm5jzvygnuzeODp21sbcKYzS/CCSlMSuD9q6ksQ/7UsRuJuBOFIAuBL4BmbGxA27Ut/fCpSa4xWBKfhO/ieLj9+JfMpnFsRPfig7gE4k8D32mGPVW5CnJsol4HZIiqZMgb9LsDk13yZM44PrrY0VlLO/QR/zdoxK06QgWbYcz9M/cDi4CQC/39HrJb5mDb5HHzXvQ4ZlsPDVoo+50rbClpNlZKfAUkNwIwCJhKOpbXoGzH/77UhqFD5Eqs4qZv57Rerad5jXvO7tb8fwepE1jTMPPoj/1VfxnTrlKgWzsxKXCpMgegOgkNWItK/IQVHZ0AFuttlduVy+GpxchB4hsuft9ns3gO5wkiRRXl5OeXk5U6dORdM0WltbOX36NPv27esX++LRTI8TI4X0ZMLn8znIdTgcpqWlhd27d6NpGlVVVVRXV1NRUZFFcAaa9ITDYUpLSwfsfAOBEU96+vslomkaO3fuxOv1smzZsn75Ix0KpMeyox47dmy37aiHOgzDYM+ePWiaxtKlS/u8oHUwSI+maezYsYNAIMDixYv79LlMv/dFr3Lv3Yu6YoWjf4/w+0n++78jNTaaxgeqCh0dzgA+kUCoqrPvj2FkS5OsurnMDFCuOTSRHW8HPIimZFfWqgzE2SSku3lLHqRII8Z116G1HEf96afh45iE6RqzuJ+XujYX0CWjU6xzeky3L6B892okEQfNnJ//6L3Ev/owvjvvsQPo+F134fv61x0yQkc9jiwjPB6TQLoE3G6SIjtIb1LMXkE3Y9Y/Aehh/DvWgVhNbNET6LWX49n2k6w+RpSUEP1//w/9yiu7zgGmpbgQZlYmA+n9ewJ6HLX240gT15jZsrRguTsEwPpvaqPmfUhHGpGVPDHkCaA0JSEO4+67j5dmzCBeXk51ezvVqmqv4ttZCQeRctp6p2d8elLULzU3U7p7N1JNjZ1pSX74w3g2beraSFGcBKAbgX96c9jeNHctVi5XqBYt17sifP75rqQRGDLucKqqUldXR11dXU774pqaml7VbIzUIL+nMAxj0M0l+huSJFFWVkZZWRlTp051uA0ePHgQj8djk+vS0tJBIT0jqTEpvA5IT3/CqmuZOnUqEyZM6LfzDDbp6a0d9VBGOpmbMmVKv5C5gSY9nZ2dNDQ0MGXKlH59LgHkP/7RlDdZH3i9pstaWpZHfu45U9aWabvu9UJVFcZVV5G49lqzXmXbNofzW8v991P1xS8697NW/NMRjyMWLULbsMEht0uHAFAVkHSXhqA+dO8nUPQngU7zi5PAPZjOaPMwMyJjYwg1CE1NqA9+EekRnA5u6ZkFTOLECuc2QpLRay9HOteAQHKOVfag/9MCIlft6qpdAXyPPOIyoxSCQTq/9z2orHQ6X8WbkXf8D/51n0KKas7aiF277J4rsrcB/8574KgMkzpNV3DdnIB/2yo6l/4IxnpAz7h/ho6xcKFTtmQhRT7T5ViZ/XskYHzTk/Cbp03537aAI1h2BMKWpC89aE4nAMmkeX8yG8wqQCwIb46aluHp9Vu7vSysqCCxcCGhUMhexS8pKWHCiROUqCpSHdlEKkcj0u7U0qjPP8/4VasYq6ooqYyYfvnleH7wAyehVRST0NqSvoeQ1EaIk5fIuDWHTc9O9TWKMXPIlRFLVlaiXXWVK2nsS3e4vkKmfbGu65w7d46Wlha7ZqMnzXVHSY8TAx3gDwVkug3GYjHbFruzsxOfz4cQgkQiMSB1Zp2dnX2uXBpsjJKeHuLEiRM0Njb2aV1LLgwW6ekrO+qhitbWVnbv3t1rS/FCGEjS09TUxL59+5g3b55d4N2PJ0O95Raz5iAFARgLFpjGA6kmhuottzgzNRbicYT1t1NXh6irQ4wPotU8gOEby8HqiZzzerlwwgTk224zA6VIJEv6IgCSSeT/+CHGLZ9CA9S77jIzQsmko95IaLpLbxofWuRzqDd80WnKAEgJEE9hSuFUQE/g3bkU/Z2fhnFK3syCOS+yt0FFbfwOvv0bwHCeDyOJiASRTncFelJzc3Z/onQkk4iFTmtq9cTz+LesgJgO6zFJhZWBsiRS86aYPYp+B9wJzDDgDpwuc7LHvH5BHd6S2paUg/UH/y23y5eFNDmWFG3MssKWAKQkfAy4IwnRrmA5M3ui/OlPOVf+7WD6h04pWnzuw+jfn0zJ2febluEWVgB3JxBTpqCqqqMJYSQS4ZymIRIJVyLV00ak9pwtghCL2QpL/6pVRH/0o+zr6PcT/d73EJWVKCUN+A/dC3/PT2Rcm8O6ZKf6EsWaObhmxDZvBnLXFQ0VspMLiqLYq/HgDFQjkUjRtuqj8jYnRkkg+P1+u2eZEILjx49z9uxZR52ZJYXrj2sViURGMz3DDX39EtF1nT179qDrep/XteSCLMsD7qJh2VEHAoE+c6EbKkivwRoIMjcQpMciqC0tLX1av5MPueouvEuWmEF6IGBmdzL+Bu0nWZbxXnqpbQ2tPHknyj1PmUGmDmM/8V5mvGc1vO1tpgX2tm14li9HyiA9EoBhoN55L3r7a+jrnkITAnX16uwxAyRBbAIqSjAmaRgXbkC9Yq1ZM+MGHbOWxeJ2G5Mo8x42iY1bZqEp7VzN2duAhm//Y0hGmtQPQPKTPP5hSm+8zM5uxNesQV+82JTvpUn3RBkwyQ9NEHvkCTuTYhkP+F+9DUnWuwhMegYqmUQpacD3+38CQwapE2lh6vusAD+OESslHrkV358edxBIz4+eJ3Hf581sixvhAUjE7GyMCEzJtsK2kE4W04Ll9KBXf9flRJf8CHEWxPSFuYPpDNMBuW0rtPhsCSGYVtrxR9dm151YzkuLFpH41rfwr1qF+L6B9JE4huoDFU5P/hJeEaSnb41cBMEi7w4kkyl7bfD9/pqiiIwbuUTIyKca0Kde2cNR50d3zByGA5HpDdIDVcMw6OjoyLJVr66upry83BGfvB7kXN3BKOlxQpIk/H4/1dXVTJs2za4zs4yKrO/6svHuqLztdY5IJML27dupr69n0qRJI3ZVxpLtTZ8+nXHjxg3ouft7tUvXdXbt2oUsy/1Wg5WJ/iY9Vl2Zz+fr8/qdfMjZqNSqt0l950bXJbCDPnXlShKzJ6Lc85TDea16w89g0+9A080Gp9Onm3Urblmj1DGVL30X/eJ3m+YH+WpCksBXDZPQ3NucO1PhBgVoVtG/uBblhw/ABzTTHtvvIT72brT/uAzPlufwen8IRyV4JmbW/RiAz4M2/p9QT/3ceUzZR3z8Knx3PGH2uUmNxffAA6a0KZ3oWTVCPhlkA2l6B57/fQRf+waQvaDHzCxX+p+RDqIOiHuIPfEQ/v3rkKyLbRX83wFsArES8PqAOOgGpS9fDT/0Zps0WORkyhT3TJQHuEmYfX0w5VimFfZtYMQy+jvRZYXtEixnybXqn0DDmeVID6bTnzlXshX0o73thuwxpyGTSOHvIEId58IyLan+G9ZKa2VlpevfnVthf06CsHBh7p45bVuziUwumZ3bfOMRAldeT+yRJ3vd68YNXdK120wJqaYXVYMzEi1x0yHLstNWPZkkFApx8uRJ9u7dazexrKmpGc30ZGCU9GQjXfKXXmcGZuPdUCjEwYMHiUajlJeX2413e9pnMBKJjBoZvF5x+vRpDh06xNy5c/tfNjSI6A876mJhyfj6a7XLqnWZOHEikyZN6pdzuKE/SY81p0mTJjFx4sR+OUdOWDbUN98M8XhBC+qckGWU3/5XlpuaBNARBlLE6O9/L6rXj/ea5aA7r7fAyQEkgM4YCiAeeiinfEwAeFRIphkn6ECdjlhSY35neMCTJD7t8yTnfQKAxCWXkIzfh3S8AfXS3+A1njF79wgd9dT/ZF8rI47vxAZ4DPgWthTNatRqu7XV+mBFHMkH0AkG+HbfYW7rxZbLZYWSCtAK0Scfh/PLYG/CKWOzMi0vAac88KDRJT3zAR9JwFa6ZHsAvgRSTRscasvKROHDlMxdFHAE5lr9ciK1l6M2Pot336MIFGQjAT8GpBIIZBes91au1UW2VjlqXKx98zmOZWYlSoCSKuxaDqvo+MCBA/ZKa01NDYFAIGdhv0UQfLfdhlAUZMOw56xffjnRH/3IvN9pkkVXIpMhs0ufh00uwzEkJWWw0RRztbzON/9u4VIBX8XMbtYClxbe5fUW6Hs8HsaOHcvYsWMdTSx3795NZ2cnZWVleL3enCT69YRR0pONfHVOgUCA+vp66uvrMQyD9vZ2QqGQnWG0LNfLysqKvq6j8rZhiN6+UA3D4LXXXiMajbJ06dIeM+ahjv60oy4W/UkOmpubee2115g7dy6VlZX9co5c6K95WQYTgzEnC8Z115FYsADvxRfnzMAURCSC8rlvuNS+pMHjQQqHTZK1coW5mhyNgmZk2DWD5NpMJo/FczyO8a//ivzLX7qTN01HeBRQU4XwKz1ob3oMdftapLSaHN/e+xB7S9GW3wiA8l8v4r9nFTwaSxGSRNY4ssajkmWGkP69qNbAXwKis+s7l+lKCXMnoWESnm8DYYXA6vvAG4NHM66NArR7gQQEJOh0J0WiAygthcUJWGEQ2PNR0+DgKg1+mzZmAUzFtf5F+GpJzlxLOOmn+vAXQPXBx5Ikb3knibmrEeNnOefiJtfKkeVw7JcWzFtkK9PNrJDjWD5kFh2nO3qJs2d542232Vbr4Czs15Yvp2nBAqJ79lD/xjeavajyjKUQcXPbt3PiU5Q8cwvsjMGp1KAz6mx6M3/HtbaIaTAGqRipGGL6eiM96chsYnngwAFkWbZJtM/nc5Do19t1GiU92ShWAinLMpWVlXZckEwmaW1t5dSpU7z22mtFW66Pkp7XGSxnrzFjxjB79uxBf+n01w/EULGj7g/DhnQzhiVLlhRs/tUf6GvSI4SgsbGRs2fPDtqcHJg9G+3pp1FXrjSDp1jMtCz2+czV/8w+HF4vQpJsgiGBqzlBlnX1lClIYw/C14AjAiIKxskLkH+203yTxbL78VgEQ3L5jLTv5F/+ksSvfoX3He9wZCwkACEQkoL2hc8iLp6BWPQWpEgjma9PSQf/43cTufzdQKp3zPiYKX3rTomVwGmGkI5TOsQ6nY5xssukAD4NBIBWGTokJE2HZLt5CqtZquVm9owC7an5nkhk1/YomK2KZnuJfvkhAol1JuHTUkTuvSD+GfieB/6WhFv8UCk5AvN0SPFmqo98CVkkQDeP4Yn+O54tzxOfdS/alBvs/YrJcmQiZ5bF58xwFHIc6w5KSkooKSkxM66bN5uyxLSFAENRiL/2Gt7UsY2aGqLz5tlmFYXGkou4ue773ZVwiwzXxc3+S5aRRZp0sC/mbxFLKtu6iGk7plRxvFKQmL6eSU8mJEmioqLCNtWx5EoHDhwgFotRUVFhy5VeD7U/r0f3tkLQdb1Hv/cej8dh1pJpuZ7r2YpEIv1q8jQYGPl/OT2ElRkYKjbNVuDc1y8BK1twwQUXUFVV1afH7i76mhxYvWr8fv+gmjH05bysmiRFUYaUwYRx3XUkrrjCXlmXf/5z1DVrTOITiyG8XlsCpd99N8b55+NZudJ0Y3ODz2s6rVlkSNeRf/sL1NK1SA0xeArQQWYnqCr6+96J8p+/zz6eJEFJieNz1xArHkf56U/RP/IRlG9/O3ubRAL1C4+CYZjGC29cAFoGy1LAaFXRDhwwM8IeDzRFXYwMPAhZAUkF3aUxq0RXfUvmVx2YdTcrgEApRCOmdAlghVmgL6UsmTllDc/I5kQNAcTqOFQb5rmiEqipZ8k6x0pA8ZmZnO0KPKiDRyEQXwuS87mTJMAL4maZyNd/i1TpyQrM0yVUUrTRnH/GtEHD99qX8O1fT2zRk2j1JlGJT7kf3+HPmzVLGFlkKr0nDR0UFcwXchzrjexLmjYNKaOXlKRpHNA0wi+/TGVlJaqq2u+FYt3PhK8227ggc98y4IYkksDO1omVwEG/bXqR95wNDUgZ9uducBLLBHxcN8n306TMSMIo39iOcf3inMcYJT1dyMxsZMqVzp07RygUorGxEVmW7SxQMBgckddwNNOTjb4ggpmW6+nP1h/+8AeeeOIJ3vSmN/GOd7yDjo6OHltWf/7zn+fpp5+2a44efPBBrrnmmqztfv3rX3PHHXeg6zo33XQT99xzT6/mVwgjnvR092UghODAgQO0tbUNjVX0FKwsSF+RnqGQAclEX2Z6wuEw27dv7/ceSsWgr0hPNBqloaHBNtIYbGQFLCnbaZqaUNetM6U9qZVu4fejr1iB8s1vomzYgJJIIDQtdx2QkQApjVMkk6ir7oIHVTOgTz0mlnGC8pNf5xpkUbI7CVCefNIM4HJ8bxEn9eMfB0VBXCLBR1LObgrwc5CSCfafPk2stJQ3xuNIcczxpggJQT/x6Y+gd0xGGgPK1v/AK/3QfhNLWmr7jCyPI/P1EnA0SOzZ9XAijr/hXnPcdyfQPvIO1Gd/CaE817YMqNehWUU6ZGW1NER6kP4ScNBP/Lnb8J36Jiy1eiNFQZiX1RWqD6nSg1HZFehmZV0eeghj3hQkw72RrOnGF7PlUd5fPoxHbDKNINQkST7ssGrONDmIK2uKIxB5HMd6K/ty60kTf+IJLnjLWzAMg7a2No4fP057ezuRSIQxqsrMzMa6RTYizZpHHdnZRV8p0T/80OHe5jr/aJSS669P6wvkPm+3LJF42oMwzDIwMO+j78570K56V07yNEp6uiCEyBnky7JMVVWVvTCZSCQIhUIcPXrUdtiyegMNhHPnQGCU9GSjP7Jf6c/WjBkzuOyyy3jhhRd46qmn2LZtGy+99BLnzp3j6quv7nbc8alPfYo1a9bk/F7XdVatWsXvfvc7Jk6cyNKlS3n3u9/NBRdc0Ntp5cSIJz3dQSKRYPv27VRUVLBkyZIh9TLuS0KQTCbZsWMHJSUlQypb0Ffk4MyZMxw8eJD58+dTVlbWByPrHfpiXqFQiD179gyJjByYiwm5AhbXFWRVRfnGN7L69QhV7crEpJ5vCZAuBHZgBm8WFAW2x91TNfE4xhVXIL/4YpZhgQAz0+TW1DRj21yOc866oSRSMon0J2AriLcC7wHeBdJ7kiz59seJfPhJmh55hNq778bYriCv0wh95lb8xkSCl9/jaLgp+QErtm3EJjwCIBg0x6RpTglgSEN0jsO4fCGRXe/u6mXz4ouoHf/jPjeAS1JZIgWTWG6kq39PIIAwDDM7l0wSf+BhfGfuQcKFnEg+hNCBDHLlUlyflXW5804oK0NcqMNNqcyUG2QP8rG/4BGbUsYNJjzxH5A49UnE+FmuJge+9vXZTnMuBCJXs0woLlNUCK49acBepTcMg2AwyIQJE0ypyT33cN5DDyFUFVnX6fz614s6X9Y8OhIQMCD9vkkGxviF+feLx83nMRaz+1Xlmrf737gXhAeSXfVmbmTTMYZR0mPDMIyir4XX62XcuHGMGzcOIQQdHR2EQqEB698yEBglPdkYCMnfhAkTuOmmm7jpppu44447uOKKK2hqauLmm2+mqamJN7/5zVx99dW85S1v6bWz28svv8x5553H9OnTAXj/+9/PL37xi1HSMxCwGlXOnDnTTscNJfQV6RlMO+pCkGW5V3MUQrB//346OjqGlOlEb37UrZ5Cp0+fHlINYi3S4wbXFeRw2Aym048BCFkm+eSTeD72MWfwvIVsWVikE34gmxKuzPEA8t/+ZgevDvh8JDduxHPjjc5eN+SQullQFFPu9uyz+baCa1POadYC68eSlK67HV7eQ/Td74YjR4jU1REKhZh59dWO4nazNwuws2tMpO5x/JFH0BcsQEwIov7tv/CtWw9xr7mvYRD46Ecdq/E2wcjlcFcGrCBFIFLXId00QQiiGzciKioQCxciqY2pRpguVt4iDnIp6DpCF+YcZEjKH3bKzlyCY9OVrwPpL2DsVuFRAb4uwmvDSCKdOZttcGGAcmQL2vhZ7iYHlV7ij34S37oNro1M0+FGTFybrRYI3nMhX08aK+C3+rqwejWRD32Izt27aSotpVmSUF591V7BLy0tzfkuyWrkGv9TTtMDt/3U73wH3/r1XXbzOeZtS/4sMp4Ot3d3EdmqUdJjoqdBviRJlJeXU15eztSpU+3+LWfOnLH7t1jPUL6i9aGGfJmv1ysGmgh2dnYyf/58Fi1axOrVq4nFYvzv//4vv/3tb/n85z/P2972Nh588MGc+3/zm9/k+9//PkuWLGHDhg1ZC7YnTpxwZI8mTpzIP/7xj36bD7wOSE+hF6oQgiNHjnD27FkuuuiiIftS6AvSM5h21MVAUZQeZ0QSiQQ7duygvLyciy66aET8kOq6zu7duwFYunTpkPoByFtjVleHtn496u23O5pZCjeZmc+HdOZMdrBlAItAvNK1v2l4kOf58PnQV65EWb/eGTxHIqYzmxAOomNlgdyeFMtcQfnRj0ypT6a5gT3XHMSpPI76zDMk162D2lpKgGBzM3KePkMA+P1E//3fzYaUgLrnO/i2PgYBFb4miKs34fvIk+ZqfGpM1mq8G8FwGDlMISuFJXRgaqlpYW0YBD75SZsk6O+6PMtAwHE8I2L+h45pVdwIHu0HJC67J39PmvTjdXpJdK5BnA+yfBZv47OACpJObNET6PL8LucxCzLoU5eY++cwOdDefyPaO28sqiYnk5h0p9FmMZBeew1lyxb0JUsQs0xnOqm5Gd/27UQrK2HatK5tx4yhdMwYSjHN7+LxOC0tLRw5coTkyZPUhsME5syh8vzzs4vZy4AZQAC0WnfTg1zwPf44Uqa8LmPemZK/5Ic/jOf73zczsLpO7MknAVwzZ/LWra73oTvZjZGOvsp6pfdvcStar6ystIvWR40ChhcG2tzBslG34Pf7ueqqq7jqqqsAeOtb38q8efOy9vvyl7/Mrbfeymc+8xkkSeIzn/kMd911F88888yAjT0XRjzpyYdkMsnOnTvx+/1DLqjMRG9Ij2EY7Nmzh2QyOWh21MWgp3Nsb29n586dnHfeeYwZM6YfRjbwsBz1xo8fPyQb4ebL9ACIRYtMaVY43PWZm8RM0xBjx2bvD7C9QCYmE8kk+ic/iaiuRr33Xge5UX78Y9djGV4vsmFArvoir9eW+mSNz++HaMqO2rEPSFHwPfYY2o035icAioIuSeaxkkn2rlmDPn06E194gaovrIHHYma9v24SJV/ya1AdgJNpx0hvEuqS5Yp/9rP4fv1ZuEHPHmupj87PP0TJfWuQTiSgPZ1I7XLaJOsJEpM+iPfEc6B33Vd0oBMzW1TuzAw4JFSKAmGncYO8oBNf6RdhL+BT4IcSnFLhNPAIiOWzSMor8MRTNT0yJOUVtq11XitnHz1yYcspeysDuW1rUSTCvvxr1uDZtMn+/+SKFegXX4z/9tsJKArjNY34k7kbhvp8PiZMmMDk//s//LffbspBEwn2rFlD81VX2cXsled+RaDhE45rYJlAFEJOsuzz2RkyN5mi57vfJbO/VVbG6cUXKZ07N2dt1Ki8rQv9sYqfWbSu6zrnzp2jpaWFw4cPo6pqUZnEUQwNDDTpKWRZ/cc//rGo49x888388z//c9bn9fX1dh8hgOPHj1NfX9/9gXYDQzfK70O4/SG3t7ezefNmxo8fz5w5c4Y04YGeE4JoNMrmzZspLS1l4cKFQ5bwQM9qX06ePMnOnTtZsGDBiCE8ra2tbN26lfPPP5/JkycPyR+iQvdKTJmSJXeRFAXtoYcQPh+irAwRCKBt3Ih4y1tMcpEORQZv4ayrAEQwaB+LujrEm99s9pJJP3euA0gSkb/9DX3tWlfXZ5JJmlavzv5OVU2CEYCskpcE5ueqagaU1lhTwbQIBBAej3lMXUfWNPSrriK6Zw/jP/UpgtEoFatXI42NudS76DA+44SxGKTOE3viIcQsD6LWi/B6iT/6KGJcqUl4fGkXQoBIQNL/T5TE1sJdCdMK/JLU94qC8pvfoPsuJ/K2XXRe+gsiV+0mOfu+VHooDQpdbnMuGRFt+XIiu3bR+YMfmNfNQhlIK0xpoFQCkqIjvV9DOhJBSjXSlE69RvKt1xNZ9ltik75F5I2bif/LY87j1y/vGuPbdjlMDixIzc3IW7ea8qx4M3LbVqR4c9Z2WWP+xS+I7NoFlwpKfz+Xkr9fS+nv56Ke+GnOfe1zvvYank2b7CylBHg2bcJ/221I0ShKOIwcS82z2X0sUnMz8h/+YG4TjSJ3dCDH41zw+OPMHzcOn8/HycMN+F5dhdQWRXqtHaktin/bKsf80uefiVxkOfLXv9oExSZG6UgkkOJxpEgEKW0eorYWY7FpZGGNW2pvR4pGs+Y6Snq6MBByLkVRqK6u5vzzz2fp0qXMmTMHVVU5cuQIL7/8Mnv27OHMmTMki2gCPYqBx0CTnnA43OO66FOnulL0P/vZz1wzQkuXLmX//v0cPnyYRCLBj3/8Y9797nf3eLzFYOhGwP0EIQTHjx/n+PHjLFy4sNeFWAOFnpAey3Z7qBS/F0J3SI/VNDYWiw3p7FV3cezYMU6cODGkpZZQRKantpZTDzzA2E9/2pR0JZOm1fN115H40Ie6MhOp+jnt299GXbHCzAbE4+YKcqezlkSASYYkGQIB0DS09esRixaZxyoHKbQFMT6YLZdzGyNw+JOfpG7mTIxrr0V56ino6HB8f/LGGznxL/9CRX093rVru/oQWZmhHNbSNGH2FgoGHdIebflyIhMnUnr11Q4i5vnZz0jceis+j4cJv/gFspvUKIWOFTdTdsd3upzpEglKPvYxeJMMNwNrDJOIbALfnXfC+TLS2oyDpFzlPJ98wTQqsFxJ7RqfMP61d8HqTxF7wpmJcGRWkjH4rgApAIHctTOithapstI0rWg3ewVRR3a9TqoJKh2YRGPLm0DxgkgQn/toVuNS+/guVs4WHLKsRVFYIYHH78iIuB4zJXuT4s34/+E0S8jVeDPd5lrZsqXri7LUvJqAZEaQn6NuRm5owG/Zt2ZmGz0efKdOMX7xYuoDJ5F/Ipk27op5DY0VOqem/p2SSW+h6te/JvCJT5h/W8kksUceQbvxRsc83TJblhQPCssU3eahfuc7ruN2ZAJHSY+NwZD6WfVkEyZMQAhBe3s7LS0tHD9+HMBuYFleXj56n4YABvoZicfjPXb2XbduHdu2bUOSJKZOncrGjRsBc6H6pptu4oUXXkBVVb75zW/y9re/HV3XufHGG5k7d25fTiELIyNSLBJWjxNJkli2bNmw0rN2h/QIITh06BAtLS1Dxo66GBQ7x3g8TkNDA7W1tUOiaWxfwJIg6rrO0qVLh/yzmY+gGoZh/p295S1U7duHcuyYg+DY1tbp+1h9frZtw/O+95k1KymYUhsPXKjBVgMUA+JRtMe/inHttWaAeOLnqP+3DiQPiCT649ejrPpuVlNSx5NSVkbnrFkIIczxZRAlw+sl8oEPsGDBAli4kMR732uO773v7TpOWt8c27p6E4gO0K58M6VvepMZSOq6Le1R/vpX1+tW+s53mjK3jg7z+I2Y7nVpC+xC8tB8tpRSXUdOJrvGUQbcaJhvdOutvgLYJRCn9GxTCBmz2eiuBEwCylPHNzDrlDqAsOnC5b/tFiKXXw6YK/76FDP7Y/fFeQPF1c5kBs5NuDdBbUrN50OxVG1VStq34w4EAm3qjRQLhyxLjcLHMB+CVFPVXOTFcQw3swTZk9V4M7PmJX7//eYXl+BsBPtsEv437QRp2THPM8/gu/tuMyMWieTOUCaTiHFBU27XloSnYo6Mo7wpgeffqjm9YwfjVq1yuCb677yTuCSRvOEG+7NcTnMWsohRynnQcT/jcdPggFTN0oYNrv2u0jOBo6SnC4PtVmY1R62oqABM+X8oFOLkyZPs3buX0tJSW045XGKKkYiB/nvp6TP5gx/8wPXzCRMm8MILL9j/f80117j27+kvvC5IjyRJtmvZpEmTzI7ZwwzFEgLLjrq0tHRI2VEXA1mWC6bV29ra2LVrF7NmzaK2B3r9oQirfmfs2LFMmTJlWAQBuTI9brVIbjU7rqirg6qq7Dqa0hL4QALpWWGSgCSAhnrnHbB6Nfi8EA8j3Qy8wQxMlZrn0L76EOrdn7d7jpBMOiV3mkYitcJJXR3axo2oK1ciVBWRSND62GNMWbLEOb7Ufg68BNIuulbyOwBZRv3DH8ygLxVw+letIvnii3hy/BhI1hgtdABPWU1CPaBIJMZ+lmkf/3K2Q1u+rMkh7F5BgOkyZ4B0J/Bd4O/AtcCVQJkH2jOOLSXwbPoG3q895V6bUUTtjBRvRlIbiT3xEP5VZl8h0dkJzxhIHzfM++pT4LuymTWqj4NXArqeAwnw77ybyPh35ycpadkWR72K2zVyIS+ZEIEp0JaAM5jHKKcoa27fAw+QvPF9eN70vG25LQFipYLY68GIqkiaRjyVHVOfeQbfnXc6nhnHOMDMlAlB8ksfpnTbZSYZ2x9PEZG0++YNUKf5qfODbNlRp11H79q1hN7yFoLTptnvm3xOc+BSr/OnP5kkCMw5SxKll11G7IknMKZNy/o7FkB87VqnacQo6bEx1K6Fx+Nh7NixjB07FiEEkUiEUCjE7t270TTNtsWurKzs8zijL5uUjyQM5PORT8kxnPG6ID2nTp3i4MGDzJs3j/Ly8sEeTo+gKArxAk0WrYL+oWhHXQwKEbtjx45x/PhxLrzwwh53CR5MuP2oWSRu9uzZ1NTUDNLIug830nPu3Dl27tzZq7m4ymgSCXhWc/bsAdB0JPQuZ7WngXmYQankQVz3ZhL/egCpsRHp1VdRV69GWM+X14u2cSN6dbU9D+O66zi9cCHH//pXpr/1rZRZrlpNTbZNr/zjH7sPvANnfx1c6ogkCc8PftA9c4aXMOVmE4A2FV/rF8EtwMiXNZEkxEvCzBw9mKo9t5zPPwa8CuKnwM8hee9VeNpfcB5HA+9XnkCKJ3rUt8bRPDSQJP6nh9E7F9BaUUEoFGIGHUhjMPvIvCmVNZoQpPSVN5nmBekoQFKymored19XwO92jTLIixuU/3oR7jAQllPdCg+xVU8UtObG40H/t7fiCb0AbRHz/HVAlZf4tz5L87kawuefT/2iRSZpuvvugs9GfM0atOvfRem2y7rkdjUgMteKdLoyKm4LSR4Pra++yu6mJoLBoL2CX6ixZTox0pYvJzJ/PqVvepM57vTePn/5S/Z5/X60tOwSDL1AfzAx2JmefJAkiWAwSDAYZPLkyei6TmtrK83NzRw4cACfz2c/Q4FAoNf3NKcz6CgGFCPx7/N1QXqEEEOqb0tPUIgQnDhxgqNHjw5ZO+pikEsypes6e/bswTCMYSdLtOBm8Xz8+HGOHTs2LElc5r06efIkjY2NvZ9LWsalS0ajIbmU6GS9imXMwLIcEElE6RRoB1pbUdeudWRHhCxjXHEF0smTNulpbGzkdFsbiz70IVu+IT/3HOott5g7pQLagj8BHo+5yh2JOD+PdrpvXwgdIL2WxOq4mbn+JlLb8EyqpidhOGR2qAqyppnmChpd/YTAzgZJHeZ3nod/Cx9R4QeaXR/CtQr8V8ashcjZtyY900IZ2c1DD91D5G27MDoVSIZgfCVGinhIaiNinumOFp/7KL4ddzivt9BzkhTXRqj33w9+f6ofkh++q2fV9BTKGvlvvx0pltbf6TsK+urLnZcjh821PnUJ/HfCzLRZ1/NjYXy//CITQhonv/QlWLSoizQVWNjyPfoo1BkwJiW3a8d85j/iQ/wQ8Pqy+hLFP/tZfPfd52zaaxhMectbmFRTQzgczmpsWVNTQ3l5ecEgXLL6b6WP2+NBCofd64QyZXMjMKjqDYbLtVAUhdraWlttEY1GCYVCHDhwgFgsRkVFhW2L3ZN626FMAAcLA515Gal28q8L0lNfX98njT0HE7lIj1ULomkaS5cuHdYF/W5zjEajNDQ0MGHChCFp3Vws0kmPYRjs3bvXthAfriROCIEQgn379tHZ2dlnz59d39PYCK2teD74QUi229/nyqQIHRgfBEVHW7wR+ed/NAmLLGcXVHu9ZgbI68U4c4bDL7xAdOxYlr75zV0/tk1NqLfcYgbRRcDxk5Sxj0gNWOqj3y3h9XbNS1VBlom/bz3aNe9C3vsX5H8cQr9lOtI68D90K0Q1aCF3NshCUiN+6Wp8y56EZgVqdWL19+N//n7nfrGYXb+Rjqy6lifXQIl7PUzJ2R1MPLgW6YgPNFMehdJFRpJTb0Ag8O+828wSCT0vScnZCDV174UQRDb8DabWFN2/xj2D480mfGUQf3JNVwNZy+baUwPflhz1NtK3AcO07p7wmc/Q+a//6up2aBp3mD1w7D5VsRi+dY+ZfZG2Y2Y3FUCPE3/wAfTFb3TU5KjPP4/vgQfA5zP7ZKWeG9uKGigrK6OsrIwpU6bYjS1Pnz7Nvn37KCkpyVvHka+nkbZ4cd46IeueDNd3+ii6EAgEqK+vp76+HsMwOHfuHKFQiMbGRmRZtp+hYDBY1P0eJT3ZGOi/lWg0OmyMvrqD4Rshv86QjxCMHz9+yFobdweZ2YOWlhb27t07bNzn8sGam2XCUFdXx5w5c4btPZMkiUQiwSuvvEJ5eTmLFi3q27lYZgd797qvfqeCQQsC0NY/hHjnm+0Mj/eWmbkJSyowq3r2Waq//GWqPR5kXbcd5iBHwJsH9uyTyaxsjNnM030/AVkNUF2Pm4bE+9+P9/vfN79L1Rj57r4b7V3vQl/4L+gLU7KyV1bCHak0mQb8AcQVIGQPkpFE2oQty7OxGyL37raJgbSz0XTLS78OgQBSOOyYp2tdy7r18PWM4xtJhBKk+sA6ZBEHLd51ITIMBrSpNxIZ/+6iSEpBhzGfDykcxvDNKrrPTjGNSm35XokHvg7x8k+ijbkWKRxGbmgArx/SMkXpz4GoVFBe+SlG5fnEH34Y3z33OFzWjClTKPnQh5xZQ4+XuHITvqcfR0odVgJ8n/kykV277L46ckODaRGdXleTSJiZGRdIzc14GxsZM2UKdbNn56zjqKmpoaKiwlz4yOX8ltanKa+5xSjpGXGQZZmqqir7NzuRSBAKhTh69CjhcJhgMGj3BsolpxwlPdkYDLvqUdIzikFDJukZbnbUxcCaoxCCI0eOcPbsWRYvXozf7y+88xCHLMucO3eO/fv3jwgTBk3T2Lt3LzNnzuy3+jH56adR16wBnM5rEiAUBeHx2P1ytA0bMG66yd5Xatzi3nAx9SxpGzcSiUSY9sADKImETazUm28msWABzJ5dnE1vH8DweLrdMM37wx9mk6FYDPWZZ0iuW2faLL+6Cgmt68J5QLwV+LxKct0n8H72G9DinJ8EeJ94gvittyKPNXutkKPkJbMfjytJjHuJl38SX/sGR+NMSQ+bTntkZOAspNXuWARFipq9iHIRlkKNUEkkoK3N7iVTDAoF9VK8OVu+1/oovo89ZmZ8EgnIVZR9CcgrIvhb15nE8z88xB5aj7FwoZ0ZkZqbs/dPJjFq3gy+pyGRxlhTdtDKiy+a2TaXDKdlkpBZk5VVC5Uyqkiv49A0jba2Ns6ePcv+/fsJBALmCv4//zN6gYxOzus7SnpGPLxeL+PGjWPcuHEIIQiHw7S0tDjklNXV1TaRhlHS44bBaEw6SnpGMWhIJwSHDh0iFAoNKzvqYiDLMpqmsX37djweD0uXLh0xL754PM6+ffu46KKLhl39TiaamppoampixowZ/Ut4PvGJ3PUzXi/aF76AmDkTsWhRl7NaCjkJiyyDYdDe3s7hP/6RRVbdkIV4HO+yZej33ot+001d9UVgBvOpoLCnYZppv+2FeMIkYJLEntWrmVFXh99y7ioACRAugbQE+NavNxtDzgBkJcutTDIARcP7qW+aDUrT9rXHqKq89pvfYEyezNholLIJE1DXrMG3fr3thJcZ+EtR03zALSuizbkRrezGLovrDpD3NsC5BOQqP0wzGHAYIRhJ4lX3Iw5Xoy9Z4uglA13F9cqWLUihkCntsnor6TolH/1otvsczjqkzKDdci2TDzQg6kBMXNi1n5uddTSBVAY0pWR1Hg/C7+9yMxMC6nywImy7ugHwsST+u+8h8o/djkyJG+kyFi7MdhFMJhHBYFe2LR9SBAlA/stf8N96K1Iiv1GFqqp2HYcQgs7OTkKhkC3VrayspEaWqexGwDpKel5fkCTJllNOnTrVllNaRNrv91NTU4PH4xmWsu/+xGCQnuFaH54PrwvSMxJeqsr/Z++94yO56/v/52xfrbq0ulPX9Sqdyt0RE+BnAsQEiAnBBqcnFFc6uBBI4tDd+H4pLpjYX0qqIRC+JAQSJ3wDxBAXTvVUTnfq0ul2V3X77sz8/hjNaJv6Vmlfj4cftle7M/OZnZ35vD7v1/v10usJhUJcuHABm81GR0fHriEEKtQy+IkTJ6itrc304SQFkiQxNDREIBCgtbU1pwmPWoFzOBzU1NSkLjzV4cDwkY+sTwDcbgz3368ElEZI0jREGiJErvp7FTOB0g9/mOM/+1lcCKgAEAyi/8u/RP/AA4SffJLg0JDm3qZ/5hn0n/lM9L4igyeXV1Rav/VbGH74Q/D7o8eh1+H7+39AqqtT5GGNjVwbGaGpphgMeghvrvdwzXMTCGD9/d+HAhH+lxj/RtXsIRhccxs6SaIlHMb8trcpk9JAAMlsRgYW3/1uDO96G/rQNLrn/wOddQzL+Ec1QhJ6/A8w3vHNxFInc2V0ReFMCPndRPQZ6cFg06pBALpr/4Gl8y4Eya8Qi+fA/NWPKZ4OMoRuvZXAww9rxx7XU/S5zyE1NGD9nd9RZF4rpCxyUr9WlSMS+sCPsbjeA/NGGFgNNZWtjSDFED0D0X1SViver38dSku16pj+0o8wOz6EQAQ5EYH9+rh+obUydBKRIcHtTlzhJOZSCAQwfP/7mL70pTXd3dYyqgDlmWqz2bDZbNTX1yOKIgsLC5qbl8ViiXLzWgt50rO3YTAYsNvt2O12ZFnG5/PhcrkYHx/H5/MhCIJmiLDXSZAoimmd8+3WSo+wgSPErjDqFkWR8CYS2rMZc3NzvPTSS7S0tLBvs7knOQSHw8Hg4CB6vZ7rrrsu04eTFASDQbq6uigvL8fr9dLQ0KAFv+Ua1GBfvV7PiRMnGBkZwWazpaTSI7z4IsbXv16ZwEVABrDZ4kIbZauV4M9/rpGIqKqPw4Huhz/E8KEPISyvSoHk4mJCf/M3GH7rt9CtY3IiW60Eh4ZWtzkwgKm1dXX/scGTT4J8wYynvx9efBHb29+OEHGPlfV6PJcuRU0mJ//fn3G86zH4TAhhjQX6uGBV1pjMRv79VUa4TQbCIKM44D0B/CLx9tRt+j/1KSyf/nR8taAIpNfp4C0Cgl9UJvdPgXAUaALsIJda8bT/BGHaHVc1EZxObKdOKdstAr4ARFQ6ZJ0F77m/Qy45g975Y6W6I+hAXPm+l4D3E2UKIAOeF15APnYsevvq361WvH/7txT80R8hLEWYYRQX4/3e95AbGxN+Ru2NAaWSZXv2lCJh047Viue1fQqRm/o2ls67VohfEB4XEX4SWnN76jYL/v2E0tOkIgDyvRY8/3NReU+CylNsRSrR/8eNx2IhcOedmBMQnDWvnQTHvBWoVSCXy0UwGKS0tFTLdImcvF67dg2v10tTU9O29rOb8MILL3Du3LlMH0ZWwOVyMT8/T3l5OXNzc8zPz2MwGLReIJvNtufIsiotPXr0aFr29+///u8899xzPByxqJRjSHiB7K5SwS7F1NQUAwMDWK3WXUd4ZFlmeHiY0dFR2tvbd82NbGlpiRdeeIGmpiYOHTqkubblIvx+Py+++CKlpaWcOnUKnU63ZjhpMrCWk1X4s58l/L//NxQVxX3G9LKXYXzDGzAdPYrumWdW/2C3I73+9QmlQKBM7tZFhAwIgOPHEe+4AxmQi0C+FWXiXqD8W74VAg/9OabPfQ7b296mSJkiIYoYvvzl1c2PPs3x5S8gVIYQ4oIzV/9zzV/Feit/nVa8+76F92XfJei7A/nDQBdgBGGtRdPCQqg0wWGdQkxU/ArwBRB+W0KnExEKQDCB8A7gW8BnUAjJzyUEqxupoyNuwiyMja2ej4RBoSYwlQKrNteCGEFwE+XsAPoXX1zdfmwsgdGofD6RGYH1MqZn3xO/zZjvXFjsUshX1LEatR6jcO1NeF7bh/e67+F57UX8f/wVZKsVubgY2WpNbNVsrmS64VNIshHZC3IQ5K8Z8T/wGPof/xjbqVMUvPnN2E6dwvDtbwNKFSv2dbmyMupcq3K4qP0/9hjh97wH9HrNBU5zg4uBDMhmc8Jj3goKCgqoq6vjzJkztLe3a5PXX/7yl3R1dTExMYHX681XevJICNXptLy8nMOHD3Pu3DlOnjyJwWBgdHSU559/nv7+fmZnZzcMNN8tyPf0JAd5eVsWI9KO+vz58zz//POZPqSkIhQK0dPTo8n1BEHIWWIQiZmZGUZGRmhtbdVuGmtlEGU71MDREydOUF5err2e0vHEStMijQocDnjf+6Lf7/NFpdgbbruN4KtfrVVn3G43C7fcwoG//VsEs5JjEv7KV5BbWxFiyFBcBSTGqQtA/F//C/G229D/3/+NXvxa9LFIQEkA45NPJpxUCoD5S19SJqFFYP753QgLQAlKxs5XUchOAHgj8O+s2euPwaD0BcVUxDQEAsgldchVxwjfUIqJb8KMWyEc3wH531ePSUN7AEvxn8H7Alrlij7gdhSyFLuPlZwfLXjzbwIMvnYa29HqOJtjubBwtbF+naDQqD6ZJcAL8j5gnxVEX9wxiIcOKdtfw2lNOnMmTgom31mO7fK7lOOPNc2L+M6VfqK7QIr5EmJCTSMNF9aSo8ViufyNXCn5VWrkWeRrIH9N6RXSKjWR/TXNzfEZRDF9N2rVR7z+ejx9fZokU3C7Ebq6VnuK1oPJhOdnP4vrldoJ9Ho9FRUVWmBxZKaL2+3GbDZjs9n2tIQp3Rks2Y5ERgZms5mamhpqamqQZZmlpSXm5uaYnJwEiMqXytU533rI9/QkB3uC9OQidpsddSyWl5fp6enh4MGDKWuGTzfUzBqPx8P58+ejMmtykfSsFziaykoPRGf1REnWYsNLAwEl3yVyMqfTIXR2Ir/udSx/9auUfeQjlJlMCIKA+MEPIr7rXdr2Jv/yL6n7i79AWLGMlt74RnT/8i9aw374K1+JM0kAoKICqfm16Me+plR5tH2D7Ih/exQMBoSxMYxDD4ApBOUoBKACRfI1DTy28l4/K84Fyv/KAAUFIEkEHnhAsTiOgGKUsBoWaXvlK/E/9hjiDc1gCykGB0vAf8WYFwBUmuFWGUEOrI7pVuBx1n5S6IEG4OMoBMIAh8UuxkMtXLx4EVEUtd6O0uVlhaT5/Ypb2ZMrlTKLDQQpOoNH9MNzrObQFEDwc3cg/LED45PfXN2/0YjtzW/W+nCiyE0wQOj9vwMhVzQRsV7GdvldyviLgXeD/FXAZAWR1QwbzZktwvIZQLBsGGq6kVUzKPcLyVSBWH1akQcCupdeSpALZFSqWQleV/tuEvUlIcurrwWDcdXTqDGtbM//xBNJJTyJEJnpMj09zfLyMgsLC4yMjGA0GrXrpaCgYNc999ZC3q0sGhtN8AVBoKSkhJKSEg4cOEAoFGJ+fp7p6WkGBgaw2Wzr5kvlIvKkJznIk54sxG60o47E1atXuXLlCi0tLbvmRxUKhejq6qK0tJS2tra4h3UukR5ZlhkcHMTn860ZOJqWqpya1RODKEJUWIgptgfM48F4881cve8+qj77WfSBgEYC9A8+qJCeFSy94Q1cveEGKiP7gRyOeLIVAd0//IMSemo0gl+P/McinAd0EHK+BbH1FeuPSxTRL3wXo+2HCAKgRlWUA6PAT1GIyQ9XXlcJjw4C9/0plJYiWSzI9fX4P/tZLB/96Gq2y4c+hOWhh6IqX5bb3g1f0kOxDnkJ6NSDToyumNhs+J/4Cyzip7SsHOVYgdfED0GWV/72TRD+EK03RwCKfF/ggPU6mk6eIaQvZW5ujunpaZa+9z1ORJLTX4A8bCLwxF8gNr8auTpisr0oKYRHzaHxgel9j+Hpu0jwpj/A9sY3IoRCSmUnFNKqHiq5Mf+fD2Go+CeMtqcx/vfThHS3Evjth5ErKzF1/Z/owbwcOA0h/asJvvbLq5WTRM5sehu+s3+NWJXgpKyB9Zzh4s7rGtUq8ezZNfOCEuUjWe68U9m336+9JptMioscKMTTagVZJnjnnYRf+UrkM2d2JGnbLmw2G3V1dSuH5Wdubo4rV67g8/koKSnRGtlzOXh7I+RlftHYKgk0Go1UVVVRVVWVMF+qtLRUy5fK1WpiJowMqqqq0ra/dGH33kUikCs3k91sRw3KjezSpUt4PB7OnTuHMVZ/n6NQq1aHDh1as+cqV0hPJHlbL3BUp9PFheUmBQEHgmdMCRg1RxsSRBGRCEIU/spXMNx6q+aUpibX7/v0pxEsluhwU50O3Q9/qPT52O1K9ae8HPn48dX3rEG21OMw3H571ART/oaZQNlvYX7qnzAG/wNj6IeIv/Zr6P/zPxNvo0SHef6L0XbFIsifQCE7YeA6EDpRXMpUGMD84APRDm86HeFf+zUMP/kJmM1YHnkkQf+SiDAighv4Ksg6MV4yJ0mIza+GX/559Ot6kP8O+HR0ZUgQQDbo4awAYsz+5AAFL/4+yBL+U59lf+kZ9pUWYouR/MkA8yLmP/kkQvgvtGqN4BuDOTPoY7Zr0K/27Vit0SQg0m0s5MJw4J+izq8x8CTBmXcjVx8jtO9NmHq/sSrJK1b+CZ7746hJv2xtVCpOUecphFRyhs1iPWe4RBPdNXOBjh1bMy8oYXUo0cTOYsH39a8jl5Zqsret5uokG7HnwGKxaBImSZJYXFxkbm6OsbExrcejoqJi1zWy5ys90ZAkadskVxCEqHypSGfBy5cvYzKZKC8vp7y8PKeqiZIkrRnmmgp4vd5dsygdiT1BenIBoVCI7u5uCgsL17WjztUVoUgns0SVkFzFZqtWuUB63G433d3dm5IcqplKyYRu4h8wvHS7Elophwh3fAWp/m3RlZUVyVmkRbX0trcRKivDeMstUcn1gtkcncED4PFgeO97QRAIf+UrCB0d638vMWQrYQCnwYj5f/8TQiCA0owD+p//HM8Pf4h+ZATx0CF009NYb71VyUIp8sc38QdAKAXmlP+VX9IjyzHVmCAIsR+UJAzPPqtZbUO85aYA8P+AXwKhGFmbBUWu9eijyNXHCBR/BPO1TylOb2pPz5yNUMGNGH1/F2O/bYXWYNwOBQBR+R4sPR8AfRGIQXjZynFEQCeKsOKqZ7rjDmZOnqSssYaCSjH+HIXF1f6qNaoeAPrRF+M/Kymvh6uPof/pMrxfQNbLyvveDdKvn0Te/3riECvh3IKkM2EFJqIPRz83h/naNQSDYVMW1Wu9nrA6lGhBYqW/SfvcpkeSOsiyvOazTqfTUVZWpqkdAoEAc3NzjI6O4vV6KSoq0ty8cr0KlCc90Ujm+Virp0ytJhYXF1NRUZH11cS8vC05yN5veA9haWmJnp4eDh8+vK47mzpxzrXyrNoMf/ToUexrraDnGFTXuaWlpU1VrbKd9DgcDoaGhmhpaaEogTtaLJIubws4MLx0O4Log5XsEsNLtxHUtcRVVmKNCgAloDT2eESR8P33Y/joR7XJuloFUrdj+Pd/V2ywEyAh2Xr1q+MnmIGA0gMUUVGSDQZEvZ7Qr/0ahslJpOJipZ8lGEzYxC/rQVB7gYqARhOcCsB3JeW9oZWDT2BUFLt8kHA54X8SvGYB/sSC99a/Qzr0GoSAE7GmA/7YDEUBLXcIq0Tw1IcwvvRdkCMqH7KIv/lBLD13Rx1YVEUIQFRIjfyHwEsr20wAwWRCvHyZLl8N9rqPcOLWzyE8GVbGL5vwP/rYuhk16t/EprMwFbNxHUjlVoSpQYWIBCNsxJ8y4fvwP0cfy+Aghl9+G4pN0Tk8BqsSsmqOt4uOG08igqxXqlX6H/+Yo3fdhWwwoBPFuGygtXqCEr2+VnUIWPMcZQu2sohnNpuprq6muroaSZJYXl7WMl10Op1WBSosLMy5RbVcXcxMFVJJAiN7yqSVoGqXy8XY2Jh2HZWXl1NUVJRV30m6SY/b7c6TnlxFNl24sZicnGRiYiLK6Wst6PX6tF/4O8XU1BTj4+ObGp+KbH8AqFW5oqKiTdtsp0wOtkOogaNOp5Nz585tunyebCMDwTOmVHgiwxoFI7rBF9Zu4I4gPbOSxPKHPsTJz38+iqTIBw8qNsyJHM6MRiwDAxiHhxEqKpDr61ezfiAx2fr5zwnfcw+GBx5QNF4+n2IbHVFhAhDCYcx9fZh/8zeRBUHpP1HP10oTf2S+j/Dkyuta7o8PbEbk63WK1Mscho9JJGQ924UEtIJcdwbD6NNYeu9VeljeLiH/lQlMFrCuTKCNFQRKPoJ56WHFWlpaDej0VN+IsNiFEFrEeuF2WPQrMr1aohiQoDq9rQS4YjBESfGEcJjq665jf2UlwWAzo01vwv+aX+Cf8MHRNkoOH6Y8HMZgMKzrkKaXusGkVMoARS44pqPA9j64FAB9zGTKZIkK4jR/5CMYn3wyYZaQ6tq2mUDThBUYtxvdf/+3koMU0d8U68S2FQhOJ1JZGb7HH4eSkqhqzmZc5DKJ7d7rdTqd1sh+8OBBLdh6fHwct9sdVQXKBRl1vtITjXSdD51OR2lpKaWlpcBqQPrk5CTLy8sUFhZqZDqd0rJEyFtWJwd7gvRkI0RRpL+/H1EU12wWj4VKenIBkiQxMDBAMBjc9Pgg+6tZW5GARSIVcrCdQg0cNRgM60oqEyHZlSvZ1ghyzARRDiEdO7eulEmWZUZGRpibm+PMffcRvP326N4fh2NN1yr8fupuuw0h8u8WC8gy4nvfG0+2RBHT+fPaewiHowwDZFAIligSfOABzB/+MEIopM37ZfUfoxHhFyGEPlbtnpdBLgBuW8m/WRm/XGrB+7q/Ri45g2noc8pkfKVxSSgCFmNOGSufLYrYNqv/LS+jTOJl4FYj/lc9hmHm/2Lu+cBKVSYA14HcbMF34OtIh89ouTEYjWCGwIPvJ/ymP9Hcy2RzJfJKY3/on5/D+LEnFQvuB4juW9JHHE9hIUvv/kOKHvsrxTkvLEZVIkwmE/b6k1B/UrOnjVzVV+UqtphFB81xLSL0SDaAcFSC8BJUKJk4UYi4noTBwVW78UiHOasNUBzmWGZD+2hQKjCBz30O8/vfH1VptHzyk0plMBKRPUlbgOFb38Jy++2rvxGDAf/DDxN+xzu0Y8hGsqMiWQtcJpOJ/fv3s3//fmRZ1qpAqp2xOnHNttV7Fdm+0JduZGpxN/Y6crvduFwuent7kSSJsrIyysvLKSkpSTtJTbeRgSoh3W3YM6Qn1Ra7W8F27ahzhfT4/X66urrYt28fJ06c2NLNPJurWdeuXWN4eJjm5uYt3wyyTd7m9/vp7OyktraW+vr6LX8+6b8ns51wx1cwvHRbVE8PtcejLaojbKQjSVt7e7vyQIg0IVjpxwk/9BCGu+9WiEqEaxXhsNJTEomV1Xf9ww/HB38Gg+v2zlBURPDhhxFvuAFdd3ccWYv6FZjNsByIlnq9FohdlFYDO5fB+I1vILwG6ACaQC4GfgRChIOzAHAdSuaPCLKarRNCIR1fM+D7w88jWPyIHa+Gygpsz56Il8SVGJCPl0I4foJvvvNhwi+/EZ11DNnaqJEfwenE+OffRAiikJtIwuDzrFazAMJBio4/hfRFI3pHCP+rHyDcfFPsUSjbjbCnVVf1XS4Xo6OjeDweSkpKNE2+KZHjWuTgioHbLchPymAyx8m+TI8/Hr3zXwB9EHj4PYR/+zZk8xrGAUYjuuEuZENp1DkRW1riK42qhXQkEuRBbQTB6cRy111KFVFFOIzlAx8gIAiE/uRPtrS9TCAVk31BECguLqa4uFizM1YJkLp6r1aBMr16ryJf6YlGNpwPQRAoKiqiqKiIpqYmwuEw8/PzXLt2jUuXLmGxWDQybd0o5DoJSPdicL6nJ4+kQO2dOHXqlFZS3SxygfTMz89z8eJFjh8/rjUObgXZRg5AeTBfvnyZhYUFzp49u60HZTaNa2Fhgb6+vrjA0a0gFeOR6t9GsOrVce5tiTJ7AoEAnZ2d2sJB3PHF9uM89BBya6vmWsX8vGJ8sEb1TQDkmPFtODULhxFvuAHsdnT/7/+t/T6TidCHP4zxs58FtRJ0BgTVm0ECFoB5oCqArC/E9MVPQWAlu+anwLtBeDlwA/AsMLPy2SLlb5EW0sAqmXqXjPVDHwG/CcJ/TvA9vwvVekVVaAfUZ5waFNqboC/lOhnbi68AgzlK5hbXw/ILYLwQ//95CKYCWLo+CsVGCAXhnSK6wiA6AlAIlvGP4jl647rZN6unzxTV2xGpybewzMvE4Prf1XUCnjt+inA12r1McDox/s3fxH92GcLtN61WthLJ1oJ+rCO/A/MmWAgSKLub8Nk/Ud4be88WRQIPPojp3nujenrWqsis1TskjI0ldGkTAPM99xD+zd/M6ioPpKfCYTQa11y9l2U5K0Its2GSn03IxvNhMBiw2+3Y7XZkWdYMEYaGhggEApSWlmr26qkgJ3kjg+QgT3rSBHXiPD8/v6XeiUhkM+mRZZnx8XFmZmZob2/f9spHto0xHA7T09NDQUHBajVhG0hLrs0moPZY7eQ7ghRWTs125EirahURFRzV+GNNYp3AVtpw990Ef/7z1Z6dRJPRrcJoRNbrtSDT4AMPoHvxJwiOEYyPfnntiXc4zMxv/ibzNhstD/w5wisC8DaI0kDZQP5zYDmI7Se/At8SEWRWW56+CpxGITmHWSU9duKdy2SUxv5iwCAiFIvgVCoNpoefjugjUv6R9eA//YAyyW8keoJfBPy+Yg1OeCUHqPMuPJXXJyYD8yLikRuQr6vEc/2NykS9dIGCoT+CcMR7dUbNIGAriNXkBwIBrpo/zb6xP0VCjw4Rd9XbKXZ8a6UCpJA0ufYYcm30toSxsahgV/XUhX/rt6ICO+ONAxQSJxT64Sd++CqY9Z/EzEP4H30soclA+KabuNzSQunCAiXr5OMYnn4ay733Kp+NMTxISKhUbFMul26kW9YVu3qfKNRSlU6mswqUl7dFIxtJTyQEQaCgoICCggLq6uqQJImFhQXm5uYYGRnBYDAk3V493aQnEAjsutgU2EOkJ5PytmAwSE9Pz4Z21Bsh2wiBClVmpNPpOHfu3I5+mNlUEfF4PHR1dXHgwAGqq6t3tK1Mj0uSJIaGhtYNHN0KMjUe1SJ8PWOMhK5ZgOllL1MmtSsSufCTT6J/xzsQIpvpN3EMMoDFQvCJJxCvvx7d+DjChQuYPvTB6AydNT7r+IM/YMLv51xRD8IjgdUqzFWgAIWcPAfCAooM72/FeNmb2h9TBAxHvJ7AGY4g8DCwCMIfs9pXAwg2FMIT+WwTjIjVNyrHGzvBrw2ASUe04YQOwTeGVNmxpqtaZLUCcyH0R2QnAYhBhOACQsC5ZeKjHYbTiXVsDEvjW/AdfQt4RlkIl+JYArf8ZgpxUlB1ktKyIxREfEaroiQibWYzgc9/Pm5fkUYKGombC2mBqsp15Fd6ffr68PT1xVtQl5URaGhYk5gYn34a8wc+EB00G9E7JFdW4n/sMSy33rraX6adT3HLcrlMINOT/UShlrE9HGoVKJWT8Gyf5KcbuXY+Il3fINpe3ePxUFxcrP19u8Ya6e7pEQRhVxLxPUN6MoXN2lFvBtlIerxeL93d3dvuDYlFpsmBClWG2NzcTHFx8Y63l8lxbTZwdCtI9yKCGtyrVkrXe3AknLz6fFGTR8NttxEcGmLmX/6Fuje9CV3E+1VSE/7jP8bw9a/H9QKF7rlHaRRfqTxJgPXXX4ewAeFRUfmNb1B8oglT2V8jRC4mlwF3A28G/lo9kBXEezwoJgE/YrXKw4pRwTBwCqUyFERxilOJzl+tHLCKEuIrQ5iiqi5RTmn2ELae34zehujB2PkYoaI7Ea+/Pm6CH+V0FvTDu2T4VQOyDBImdMiAhPWlP4qSy20Fa7mplQAlVcDhw/j9flwuF8PDw/j9fg784hc0fepTipHCymfWs8KOlZmp/wgBJwyEEhPOlYqL1NGhkT/dSy+tGick+C0KTidCVxfme+6JJ+Erltex2T2GL38Z85e/rFQdN5DLZRMyTXoiERlq2djYqPVwXL16laGhIaxWq1YFSvYKeDadh2xAtvb1bhaR9uqqEYvqCgdoZLqoqGjLBkLpQLb0v6cCedKTQmzFjnozyDbS43Q6GRwc3FZ/0lrI9BhVNzCXy7VtGWIiZIr0qG5zhw4d2jHpjkQ6xyOKIj09PZjN5s1JDO32aPMDVa4UIVtSJ6N6rxd5pfqjwWbD/zd/g/ia1xC89170ExNINhs6jwepoSEqHwiHA/2PfpQwtFJ9JTa3RggGMf/pX8JngcivREQhId+AOJeElTu1rEOpJAjAn6GYFhgUSRqSDt/978R68JsIkl+RtD1MVGWH2K/MSfxEXRCRrdFVArmyEn3gx1g63wOyvOoSBwjPgfGr38Jo/gGEpSgJVqKATp4EToWgGARk0OkQpACEFcmdJpfbZMVHcDqx3HcXQrUfHD7wJXZTs1gsWj6HfO0ahW96Ezq/XzOvMN95J66XXkKMIG0AupdeQt/VhfmjH01oUS2bK/G3Popl6U4Q/dFEJcKgIJaYlf75n8NvvxbdwqohhPYenS7e7CBme5HfTej++wm/5z1ZbU+dCNk82Y/t4fB4PMzNzXHx4kXC4bA2cU2Gk1euVTZSjd10PiKNWFRjDVVSubS0hM1m06pAFosl04cbhWz9be4Ee4b0pPPLU+2oJUni/PnzSVuxyDQhUKESA6fTydmzZ5O66pXJPJtwOExvby9ms3lHMsREyATpcTgcXLp0aVtucxshXZWe7brMRZofCD/9KYaPfjT6DSuTR9FiicpLASAYRKytxXDhAlJDA1JHh7JNAIcD3UsvITU0oP/xjzHdeaeSNxNIMEFdD0YTzIeiSc+KZE2WlJ1F3rFkHfAxE3xKREBcrfz8HPgMEDDgeetzyNZl5Of/QSE9xaxtaa3+fxB4FLgLhXRZTfhbHwNAt/CSNhnXrKClaMkgS8BXURzbgkpWUSThSNhsr0rzikES9OgFPRBBSAU9+l/+SOkF2sTk3dD/FDzoV/J49CikqltA19WF+JrXJPyMfmJCqfBEfPeywcDUf/83c4cOUVpaSv33v0/lffcpx+92K+dtDYvqcO1NeH77egzBpzHf87Dy/cbI+2LJX/39fw7l9yOUKoYQgYOfw/ye+5T3JIAM+B94YM1zku321Ikgy3JOTG4jq0ANDQ2Ew2EWFhainLzUKtB2Jq67aZKfDGQzGd4pEkkq5+bmGBgYIBQKUVpaqpHpTFW70i2lSyf2DOlJF1S5V01NDfX19Un94er1egKBwMZvTCHUxn6LxcLZs2eT/sPQ6/UZqYh4vV66urpoaGigtrZ24w9sEekkPZHVqu26zW0EnU6XctKzuLhIb2/v9l3m7HZkwPi610VP9IHwQw+B3Y5w7dpK2SQCkoTtla9cNSh4/HHEm29G/8wzCslRq0eiGNUPFEso0IFsNCMk+s2KEuGqP8IQ+DqIIKiT9WXAbCb44Q9jeuQR5RjCIUJ/8R7kwv2YuC9mOyD7Ifjbn0fXeAoCDgRphREVA+8G+asgmwsRgiuvRx6P2QzdEnzUCGVh/Pc8CMjYnj0V1fgv2Q6AW69I6ewr24Z1JV1yZSX6rq74YFg1pBQQZCm+quV1Y/nQ3TD/oYShn5EQAk7Miw8qMkH1Mr8VeL8X6y1vw//4VxJ+PpEEUieKHH7tawmXlbF0+TIV99yT+LuLGaO2TXMloT+6h/Ab3xFXcUnUZyboQgizQKGyD/PwPVBmjGqXkkH5jmSZwEMPEc4BG+qtIFcntwaDgcrKSiorK5FlGa/Xm3DiWlpauqlnZK6eh1RiL5yPWDItiiILCws4nU4uX76MyWTSqkDplJx5vd5dGUwKedKTVOzEjnozyHSlR5VKNTU1UVNTk5J9ZKLSo8r0Tp8+TUlJSUr2kS7SI4oivb29GI3GpFerIpFqN7qZmRlGR0dpa2ujoKBg4w+sgYSmBkVFyK2tABinpuLlbaKoBJauVAFMd9yBr7kZ0513Rq3Uxz2CLBbkQGBV6qYzxFliywAmE8GPfxz5zCsJmf4A43/+HYbP/zUEzWBdJVm+W29dNUm4915lu6GY7YkQrHg34oF3Ki+Y7QTbH8d04Q4l7+iVIQK/+1nCgRbEggKKX/3q6GMOBBSiNhuCWbDcex98EaVStFLVsXTeRWDpY/Bet0JwROAOkM+DkMgtbqWKJjidmD/60TjCyR8YoLwAWQoyWvVxampqsHTepTA/r1vJ85lWiJLlzjvjZGqR0C12xYfaiivHdSWE5a7En493YFutyuiBiuVlhBgnt6jPB4OI9fUJjS8SVVwS9plFkD+WgDkd2BJUDPV6kCTkXRgUKElSzk9uBUHAZrNhs9mor6+PmrgODw9vKs8lX+nJA5Q5nloxBDRb7MuXL+P1ehkYGNAyyXZqRrQe3G53nvTkOlJ5Y02GHfVmkEnSMzs7qwVzJqOxfy2ks9IjyzKjo6M4HI6ky/RikQ7Ss9PA0a0gVfI2WZYZHh5maWkpKS5zCSeb4bDWFxGuq4uq1iSEXo/h299WZGwRiLujyDKYTKvVgXAYTCZko1Gp2Hg82ltNH/uYYowABB9/HN+Lf4ZufHy1Z8jhUP4/GMT6oQ8pJGwNmD7wNYK2X0W8+WYAxLqb8dmvR+cdRypoALMdHaADAo8+ivmuu5SxBIPIghAt79sngBT7veow/6+/VCRs6lC/C5wwQXFQyQX66kpfkWzWyEPCEM/CQnxv/yby8VIWQ2XMz4Woqj2Gp/J69L/8kVLhmY6oDPn9GJ5+mtA99yQce8IrUJXPARj0a1o3Rxk0RPTvrOXkJq8cvxwOM/LxjzM6Oop1dnZTze2JbK7ld4YU+/DnlPOH3geiUblezOZVSZ3XCyTuUwKl2iX4ooNicwW7scIRO3FVq0CReS5qFUiVL+WKzC+P9MJqtVJbW8v+/fu5cOEC+/fv1zLJIh3jioqKkvo78ng8edKTR2KodtRFRUWcPXs2pTfwTJCe2EloqrMLMlERSYVMLxapHpcaOHry5EnKyspSth8VqRiPalhgsVhob29Pzm8p1tRgxa6aleZkS309A3ffzbEHH1RMBhKt7LvdGL/4xSjSEglt4i2GwRDjKmexEPjmN9H9139h/PznlUms+hteIQOmO+7A19+v9Q5pMjrVNW4dCAChkLKN669fNVkw25GWQPezbpBlpDNnFEnarx/G1/0zdDMeJJsN6yteEb3BGV+8Ti/sAVfMjieA960QnpejZAa5zHje+jPkWiXTJnF1Q0Q6fAa5tBJpcRHFp1uRholHboC5D8aNz/zww4pbXhHa5J5lpYon769D8fMOKcetutUtr2zAH4hq/F/Lgc3wrW8phESvV9zPHkuQr/PZzyKdOYPc2Mg+YN/oKJ7iYpzhMBcvXkQUxXUtjmNJlmP072i89AmErwYQNJvrELLFgv/P/xzLJz8Jy8urG0ggqTNMfUsxl4iQIm7V+S6T2I2kJxaReS6iKLK4uIjL5eLKlSuafCkQCOzKIMg8kgNRFDEYDFGZZMFgUHOEW15eprCwUKso7nSetluDSSFPenYEtefgyJEjVFVVpXx/6SY9oVCI7u5uiouLkzcJ3QDp6Fvy+Xx0dnZSX19PXV1dSvelIpWkR3UJ3Gng6FaQ7J4etUpVV1eX9O8k0tRAbmzUCE84HMZisdB033143/QmbK98ZbwUi5XJ6ArhWcuRDYCwiBxrWx0KIdXVYX700bUzgIxGpaqzUuHRZHRrINExYDCg/9GPEG+4Aex2hTjdeusq6XiFHm7Xg9ECcohg2+NQdzPBxx/HdMcdK2YMywi3rGxYBDkA6EH4CnGkR7XDlr8KtBZCqYj/+kc1wgPrS8iithVBRAJ33435k5+MHpvJhOHFpzGLjyiT+5/64K8EMFkgFCL0yT/BWPMN8EpwTxBhPnLjq8RjLVtrwenEcvvtCBEEzXLrrfi+9S08P/mJFmirHnfkdmyhEMWPPkrDTTdFWRwPDg4mDLqMlL7Nz7+eAnsDVcI7lZMZ+R0fORInjYx1bosyl4iQIm7F+S7T2AukJxJ6vT4qz0WVLzmdTq5du8bi4qImX8ply+Y8kotEFt4mk4n9+/ezf/9+ZFnG7XbHZUyVl5dvy10wL2/LIw7JtqPeDNJJepaXl+np6Um61fFGSHVFxOVyMTAwkLK+q7WQinFJksTg4CCBQCCpLoGbQTJ7etJSpbLbkVeqIKIoar0E6oTLGAggWCxRNsGy2YwEGCJIeCwpipuuWUzIsqCFoAYffxydx7NCKtYg8z6fImsDdOPjCYNV4yAI0TbZy8uYPvxh+MAHCD7wAKZ77lmdxBcB7xBBEDVbaNMvbycwXY54/fX4+vvR9/4I08yHocCt9Jd8HIQSFNvriMOOG7Megqa3E3rtxxJOtGOrG7GEp+yHP8T2mc9EVVOwWKIrXKEg5vmHEQr9sOBTKjlBwK+Mxfhn38Tzy5+iH34RS/gjQERFzmJR+rogzj1NlYsJP/lJQgmk9fd+D2RZIUcrVbhELmzqdgyVlXEWx06nU5uEqKuwxcXFmjxUKqmJd/7z+5Hq6jYkjIJvbKXCE3Gt6IxRGUvZjr1GemKhypeCwSAFBQWYTCZcLhcjIyMYDAaNNBcUFOyZ87SbM2K2i43c1ARBoKioiKKiIpqamrQFmEh3wY36yiKRl7ftAiTrhpEqO+rNIF2kZ3p6mtHRUVpaWtJe4kzVGGVZZnx8nKtXr9LR0ZF2P/xkP7CCwSDd3d2UlZVx/PjxtD8Qk9XTMz09zdjYWFqqVLIsI0lSHOEBFNIRM/EVBCHOmCwKJhQf68gFeRl8//Iv6EymqP6cuFX76B1FH8cGhEcwg9ykR3bpIWCC5WWFiKy4pJnuvlvJeVGRyGzA7cf8vt+BYUkxTrjxBvi3DyjvcwBe4iVtiY5FBJP/bwjxsTXfs5aVss7lUsJBA4FVAvHRjxJ44AHM992nTfYDD34Yc+kXIexf2y3uqhvx1A0Q/lD034JBWFhAt7AQTyYNegxPP435wQfjxwXRvTTNzQhuNyTaToTsLLJqVVhZSWFhIU1NTVo2x9TUFAMDAxQWFhIKhZCWlpTersjtWa0IbveGhFG2NoIUQ9akUFzGUjZjr5MeFZIkYTAYKCsr0xZ+/H4/c3NzXLlyBZ/PR3FxcVqa2DONvKlDPCRJ2tJcMzJjChL3lZWXl69ZUfR6vUmPucgW7N5fTgqQSjvqzSDVpEetHPj9fs6fP5+RG2uqekX6+vrQ6XScO3cu52+oqove4cOH0yKrTISdkh5Zlrl06RJutzsphgWb2Z8kSdqKWdxv125flXqtTLaDjz8OKD03UVkt6jYxIP+hjPDXK7/JIICA9Y1vJPjQQ7BSHcBuJ/jQQ5je977EEjeLRZG3AYYvfjG+6kBEhaWDlUydMBToCPvfhuHOv46vIkXeJxIQBUEPjHrAt9JTdH0/wbYV17dqPYTdiY/VbEAWwwhGFIL0bpBLjYjLlxF1Jdq53cxvTD8xgRxbATMaEVta8EQGhBaB+dlHlL+v4xYnV1YS+uQfYPzTJ5XxhoBwiII/+iOF/MTeV/xuzA89sGrlvRZkGdsrXqFU74LB+NBQnw+5sXFN+ZwyrOhsDrfbzcDAAJdFkf2SFMfjVBnbetk7Wihq511RPT0QnbGUzciTHgWJzoPFYqGmpoaamhokSWJpaUlrYldlchUVFdhstl11DvOkJx6J5G1bQWRfmSRJLCwsMDc3p1UUi4qKtEB2nU63o0rP29/+dgYHBwFFxVFaWkpnZ2fc+5qamigqKkKv12MwGHjxxRe3Pb6tIE96NolU21FvBqkkPYFAgK6uLiorKzNSOVCRbMtqtVekpqaGhhUJUS7j2rVrmoteJldidnJ9qFlPBQUFtLW1pfxak2UZURQ1h6S19ifefDO+66+PdlED7TWhs1Oxjl6Z1A7fdx+e1/0KFa/vpfb37kVPWJvAm973PoKA+E7FSlp85zsJAqYPflCxxI7ccSiEcOEC1l//dfD71+79KUIhPGZVYhbEYP1rMAWiJGgEAgQ//WlM99+vEKhl4OmVnh6dEQIehEdZbfZf6SkSO1Zd33R3fw/jZx5JcCw6eMAMnoCW1yNIYXSFB5FXFizUCVwsAYo1ERDr6+Od8yIIjFxZqTmT+U99FkvfR6HcCLdG9/RoAaABJ8babyJ8ARgFPg+ERFhaUq4DowHZCIIBxdb6RpD/JYFDGzESPvU7Wflu46i+IIDLtabsLa6HaUWKUlxczP6jR/F96UsUvO99SHo9QjjMlY9+FJ0kURYKYTQaY/cWhXDtTXgqr9cMHvTOH8dlLGWzsUGe9CjYaKKv0+mimtgDgQBzc3OMjo7i8Xi0KlB5eXnOV4G2WtXYC9gp6YlEpOsbKNfS8PAwDz74IMPDw5w4cYL9+/dz8uTJbW3/H/7hH7T//vCHP7xuDMiPf/xjKtMcqJzbv44tYLs3VtW9bGFhIS3uZeshVYGQak/FsWPH0n4BxiKZltXz8/NcvHgxbY5mqUQ6AkfTAdVEorGxMWVZT5GIJTwbwm5XDAUSvdbRge/GGwkPD9PrdlN54gTH6urQha3oLFYIrTptCYDhQx9i5ld+hbKjRzEajYg33gh33x1lPy0DwY9/HNN990VbR0eOARBf+Ur0zudBDMQQEQPUAf0RrMcEwr45At/+tjLhLy7W3Nt0PT/B/Bt/At6IhYWIniLMdiSzHenWBoyffzSqt0YGgn/+F/Cm2tUcoBVTBL1tP3rQSI8qI5QkiXA4jPk738H2/vdrVtmBBx9EfutbGf2zP+Pgpz+dsG8l1pkscOpziKUtyL/WCB8iikAJASf6az9SGE0xYFNODZGcxmKG95rA4l3Nx/lu/PmOfVJs+OSwWNC/+OK6sre1IAgC8i234H3taxHGxpAaGigwm3E6nYyPj2sTlMrKyjVX9GVzJbK5EmF5EMuFOxDkYM4YG+RJj4KtVjfMZjPV1dVUV1cjSRLLy8u4XK6oa6aiooLCwsKcO78b9a/sRaTynJjNZk6dOsU//uM/Eg6Hee6553j88cd57rnn+M53vsNrXvMabrjhBl72spdtiVDLsswzzzzDf/7nf6bkuLeLPUN6toN02lFnChMTE0xOTu44BDJZSEalR5ZlJiYmmJ6ezkj/TrKh2mubTKaUBo6mGioJTVe1NNKwIFnnbNlioU8QONrRoa2USQ0NWv9HJHThMJZvfIMLb30rOp2OY9/+NtZYGZrNBpWV65oXCID+hRcIPPYI5uCHiHL6EkSYjK6WCCEw3vEIWJ+EUFiR6amObnfcAf6Y31fkfc3hQNfVBYJA8MEHlcqWJClVDosF06c+RbD2cXw39is5QF4buqseMCvBOIaVKpl+hThKkoQ0O4vt/e+PcqQzv//9WDweZm64Ac/v/m5c30oiZzJz3314XtunTODNxJMjQQ/iSsZPQhlcEBr1oLYpLhHHaLZ1hw+FEM+ejZclxritxSJyAStSxlYMFBcXc/DgQYLBIC6XK2pFv7KyMq6vwzD1LSy/vJW4QWe5sUGe9CjYyXnQ6XSUlJRQUlKiXTNzc3OMj4/jdrspKirSqkAbVQ6zAXl5WzySWelZDwaDgVe96lX813/9F+9+97v5//6//4//+I//4Otf/zp33nknR44c4fWvfz033HDDhlmAP/3pT9m3bx9HjhxJ+HdBEPj1X/91BEHgtttu49Zbb03FkOKQJz1rIN121OmGJElcvHgxI4YM62GnlZ7IcZ07dy5rxrVd+Hw+urq6UmLlnE5MTU0xPj6eccOCncDhcHD58mWam5vj9c46XXQfDcoEet9f/RXFH/kIwWCQ4iefjJ9UezwwNpawjycKfj/m996rWEm/2wivsCpVlpqPY/J8PPq9skJ81MqT6Y478DU3K1bYiapJKz1Fwo9/jOm221Z7VoxGgvffj+n++5XjXvms2gMk/Pgy1jvvXCVsgqA4r630Q4k334xOp8MwPZ0w2LXs/vsZa28ndOgQuvLyqIlOlDPZEopcTQ/C8S4oORPV76ORo9XhQ5kN3u1B+KryOUTgXRC47gHMffcp23YFwKID9wZOeeo2CwshHCb0G7+B8Qc/UMYkikp16tixTdlzR21zExNdk8kUt6LvdDq1vo6Kigoqi2SqLtyFEMfyADGY1cYGedKjIJkT/VgrY7UKNDk5CaBVgZIdaJks5ElPPNIt+VNzekpLS3nrW9/KW9/6VmRZZmBggB/96Ee8+tWvTriY/OlPf5o3v/nNAPzd3/0dv/M7v7PmPn72s59RW1vLtWvXeN3rXsfx48d51atelbIxqdgzpGcrP+5sq34kG36/n66uLvbv309DQ0NW3fh20reUzePaDnaDPE+WZYaGhvB6vWk1LEgm4VGd/5xOJx0dHXGrpbrxccWBKxFxMZnQjY9jAQSjMc5wQABMn/kMC299K6Xf/z4EgwmrDVFZQX9lIfCuv0bQj2D627tB2kDyajQq8qu1zn0opASV3nEHQmSTfiiE6WMJXNmMRnTd3at5QpEVqpXPR4alSg0N8c3/gKTX07gy6RVFkfBKf49er0c21yvOZM8BT7BSwPBR8MBNCsFcOd+BBz8MVTG2zfpCAofuwsyjcNqtGDnYAYMZecyEp/UnCDY3ckshtk+v/ZCVQTEv0Onwf/azCE4n5ocfxvjDHyrfoyxHueRt5La2U0Su6IOixXe5XMxe/hkVcvzDXAYCR+/O2ioP5EmPCvV+lWwIgkBxcTHFxcUcOHCAUCgUF2ipVoGyRTKdJz3xEEUxrVW6REYGgiBw4sQJTpw4wQc+8IF1Px8Oh/nOd77DSy+9tOZ7amtrAaiqquItb3kLzz//fFpIT/7KioCaCD8/P8/58+ezlvDspK/H5XLx0ksvcfToURobG7PugbNd97aFhQVeeuklDh8+nJXjgq05nk1OTjI4OEh7e3vOEp5wOMyFCxfQ6XS0tramhfAkyuDZCSRJor+/H7fbTVtbW8IHTyK7aw2hEFJDw7rvEYCS732Pkd///bi/ySRonDeawCFj6r0PYTmeJMW9PxQCpxOWl+PeJ1ssq1lCa6wkxp1FdcK/wUNYNz6u/MeKe13scekB64kTmEwm7R+10hs2lOGp+RTyk2iKLQEQRBEhFEJYWkLw+TDf8zAsxBAqWSRccxPIoqIVswP/AdzpxvK792BrfxW6Z0eQa4/h/+xnkc1m5MJCZKMRWa9fPed6PYG778bT14d4442YH3kEwe9H8PmUYwkGEfx+LHfdheB0KruurETq6NgU4dlpf6bZbKampoaDzb+GPsGlLgsmlqrevu5+hIAT3cJLCAHnjo5lu8iTHgWb7jncIYxGI/v27ePkyZOcP3+ehoYG/H4/vb29vPjii1y5coXFxcWMZuWkS8qVS0j3OXG73TuKK3n22Wc5fvz4muoUj8fD8srzyOPx8G//9m+cPn162/vbCvYU6Vnv5ur1enn++ecpKSmhubk5a3902yUFsiwzOjrK8PAwHR0dWTuR3s74Jicn6e/vp729nYqKihQd2c6xmbGpk2yn08m5c+dSLgVLFbxeLy+88ALV1dUcOXIkrQ5tySI8oVCIzs5OCgoKOHny5NqTkhW7a9lqVf5BIROy1ar106w18VchhMMcePrp+NdJQDpCIbAL8JysVEESfEYGZSJvtSphpZ/6VFywaugjH8E3MIB4880KKdtEhVUGwr//+4oxwnqSPJ8PKXKlsKgIBEEjFDIg/vbLNYc8nU6HXq/HZDJhsViUVWexDYwbXP9GE4Gyu5F1VmRDMbLOir/1UeSiY8q/f25Efh/I31Ykf8LyMoLPh+WuuzA+/TSWj34UTCalavQXf6GYD7BKsMyPKFbZwtjY2iRvxbBgO0hKJdJcib/tMWSMq+dXMDLT+BkujS/wwgsvMDQ0hMvliqqiG6a+he3ZUxT8/M3Ynj2FYerbOz6W7SBPejJT3VBdBJuammhvb+fMmTMUFhYyPT3N888/T29vLzMzMwQTVGlTiXylJx7pNnfYaU7P3//938dJ26anp3nDG94AwOzsLK94xSs4c+YM58+f541vfCOvf/3rd3TMm8WekbetB9WO+vTp0+va62UDVPnXVkhZOBymr68Po9GY9Tk1W3GokySJgYEBQqFQVvUlrQWV9Kx1nMFgkK6uLsrLyzNqG75TqLK8dP2eUmFYoGZyHTx4cFM9fVF21zYbOo9H+zcOB9jt69pWb+ablgFZp8Pzvveh81rhCX+0O1kkrFaCjzyCePasIm2LveYKC5Fe9SotHwi7neATT2C69dZVKZpen5AIhe68E+x2wvfdjOGT31CWzoIxY7BYlHMA4HBguuN2hJjftf6Z/4C/GIDa43H70Ol0cPAgiBssgAQChA/cSLjpTzTbZlXOJZqvh6f0CInImV6P+d57lWDUFZg/+cl4YqM6sDU2rlvNW8+wIB3QrKsXuxAAqeQMxeZKWkDL5XC5XFy5cgWTyURVscCRi3chSP6ccXrbzciGildslpTH48HlctHb24skSZSVlVFRUUFxcXFK5xB50hOPTPX0bBdf+9rX4l6rqanhBz/4AQAHDx6kq6tr29vfCfY06VHtqBcXFzNuR71ZbLXnxev10tXVRX19fU43wsdCzRWy2+2cOHEi4w+MzWC9Ss/y8jI9PT0ZDRxNBiYnJ5mcnEyLa16qDAvm5+cZGBjg1KlTFBcXb/6DEXbXwjPPrDb5B4OE7rmH8DvegfjOd+K78UYMX/oSxi98AcLhTTuGCQCShPWLX0T3yCPIgj5x8zpAOAxOJ9ZXvELp5XG7o/8eCGB+29u0KodqPBBcXMR0991ak34c6REEdJOTCP/8jxg++Q0l5FREIT6xl/b8PDgchIeHEWU5/mGjA/3Qi4imitVspGI0VzjDN/8JRDG6OqbXK4YJgqCQEJ0O26tehecLXyD8trepm1UOdWwM9Gs84oJBpV8nJhh1LQc2ubJy1agAlD4miwUElL6ibSyKJnuiK5srkateE/d6bC6Hz+fDM/7/EGV99HeiM2S109tuRrZN9AVBoLCwkMLCQhobGwmHw8zPz3P16lWGhoawWq1UVFRQUVGB2WxO6r6z7VxkA9Itb9sp6clm7CnSE9lTEQwG6e7upqSkhI6OjpyYNMPWSE8uVbC2AtVZLxtyhbYCQRASkh41cLSlpSXnbjTqxE2WZQYHB/H7/WlxzUsV4ZmentZMTLZN2hyOuCZ/4yc+gfHBBwk+8QTizTcT/sQnCP/u72J9+cvjzA3WgwAYVra5Vj1UBhAETB/7WJykTXUgQ5IUJ7dIR7bmZiUvKBjUqj1x+5BlzL/926uVKtUt2wCyHrDYwB8EScL8B38AwSADt9/OsUQHKgGDTqw3n1ghh354t6xYTz+uVLGijt9oJHj33ZgefnjV8GHlXNje+16Wy8oQW1oQKyuVinF9fcLqjAwE3vEOzF//evQfRBH/Aw8okrcEDmxRRgWFhRhGvod5/mHMpV/E/OwjWR8EqsJqtVLQdA79sBT1BUvhIL0jbkqqpqioqMh5q/9cQqqMDJIFg8GA3W7HbrcjyzJerxeXy8XFixcJh8NaFaikpGTHhCVPeuKRbtITDodzogiwHezJK2txcZEXXniBhoaGtPQbJBObIT1qBWt0dJRz587tKsIzNTXFxYsXaWtryynCA/GVHlmWuXz5MmNjY5w9ezbnCI86nlAoxC9/+UsMBgNnzpxJC+FJtmGBLMtcunQJh8Ox4yqVbnw8TiYlAILfr+TkOJRMGyoqCN17r9JIbzJF9bvIoLy+zn603h2LBRnFDU1W95XIBc5mI/jIIwSefDKhjEsL2Nxgn0KMNA9Qls/uNhH4P4+DXq/sf2kJwe/n+BNPEHroYWS9bnVsBgh//A8xffxTijHA0hKCP4jwRAjhcb/SfxO7j1AI02c+k3hsgQBFv/d7lLa2Yvnud8HhQB4ZwfOnf5rwHJq/9jVCf/AHSg9WcTGy1Yr/0UcJv+MdePr68H7ve3j6+gjfFE1iVKMCasAsPoRQ6EcILyFIPiydd23JECCTkibZXKn0PEX0QgXbH6fp2DlNNvzCCy8wPDzM/Px80gKj81gbuTLRFwQBm81GQ0MDbW1ttLW1UVpayrVr13jxxRfp7u5mamoK/xphyxshT3rikTd3SB72VKVHDa3MZTvqjUhPKBSip6cHm82W00GWsZAkicHBQQKBQFqsj1OBSNKjOgWazeac/Z4EQcDj8dDb28vBgwfZv39/yveZCsMCNfzVZrPR0tKy422u6+ZmNK5m4tx5p+KCFggoUiuzmdB73oP0qlch1dWh83gQOjuVgFCDAZaXExOZz39e690xffjD8VI2FR4Puv/+bwzPPKNVeDSEQoiHDsW9rhIrrFZYcS1LBDkEoYPvBWOJIpmL3I5OpwSeGoxg1ilyugf+EvnsKzE88k/Rltc61ixhrfetCKBVp6x33IFVr181KPj938f893+vSQnVzCHjN7+J5yc/QXC7oyymI4NCE8Ew9S0sF+4EOaZCl+VBoLHQ+oAieqEKgIKCAurr6xFFkfn5ea5du8alS5dSKmnKI3dhMBiorKyksrISWZbx+Xy4XC6t37a0tJSKigpKS0s39ZwTRTF/fcUgnUYGsixn1L0v1ci9meMOcOnSJXw+X040va+F9UiP2+3Wmq/TMQFNJSJXQdUG/4qKipxu8FdJj8/no7OzM+f7rERRpLu7m+bm5rRUE2VZJhwOJ9WwwO/3093dTV1dHTU1NUnZpurmZrr9dvD7oyfraiaOKn9TsSJxMz72GL73vlfpDwLo6MB3440KUfrpT+Mka0gS4g03KEYJFRUQk5+gVn5Y+bfhG9+Il7xZLIT/8A+x/uZvag5rUfsoKCD0e7+H8atfTThcGUDWYfzgkxB6FDm2EuT1YnjqqajXTB/7FL6f/SxBDw3xvUFbRSikmBesEC/zt78NghBv7a3XIy0uIp89u+nrSQg4lTDUWMIDIIW2FASaDc3rsrlyTZKm1+ujJrOxkiY15DLVje155A4EQaCgoCCKOC8sLOB0OhkeHsZsNmvEeS1n0nylJx7pNjKA3euquKdIz4EDB9DpdDn9Za5Feq5evcqVK1dobm7ekdVgNkDtfdHr9SwtLdHT08PRo0exrzSJ5yp0Oh2Li4tMTEzkdOAoKAG+Xq+X8+fPb63ZfxtIVf/O0tISfX19HD9+POnfhermZnj6aYwPPhhlGKDzeBQZWSTpUbFSCVINEXA4Vpv8f+/3CM3NYfziF5XKjyyvWmLDKtm64w5l+4GAYkQQDsfvZwWS1YrzwQepuueeaBIWCa8X4ze+oZgHRKwAqlk2iCKCJGk5QLJOF0e2Eo7T44k5Xh8EQ3GkbMfftmp+EEOwhBWTAlEUteqhXq9fl1QLvjHQRYehygA6syIXy5Eqz1ahSppUWVNkY/vg4CA2m02bzO7WXoA8tg69Xq9dF6AYK83NzTE0NEQgEIiqAqmT+jzpiUc6F0jC4XBOKmk2i907sgQwmUxbcj7LRsSSHjXx3uPxcO7cubSm9qYKakDh7Owso6OjtLa2xqUD5yI8Hg9zc3OcPXs2Z5uEVZlhMBikrKws5ddbqgjPtWvXGBkZ4cyZM6mTudrthO+9l/A73rFKXOx2padngzBTAP0zzygSOKNRqVrIsjKBX3EtI8G5iLLODgaxvva1UX+P/YQgy3iMRkSdLuphIEe8XwAIBFZDO1ekJ6H3vAfjl76EEHNPFTbT/+HzKePs6NCOl9FRzH/4h3HHu2Pik+CeLwPBhx7CWFODXpKi+sRAefDrdDrtH+1z1kaQYr47wYznVT9DLkpo17AmcllCEtvYHmtvHFkFyuVFxjySC7UKVFdXhyiKLC4ualbqRqORiooKAoFA/pqJQTrPh8fj2RXzrbWwp0jPbvghRZIe1YGutLSUtra2XTE+UCoiQ0NDBINBzp8/n/OrDmpjcCAQ4MiRIzlLeEKhEF1dXZSVlXH8+HE6OztTOnFLBeFRQ3rn5+dpb29PzyJBhJW1+v9ahUOWFUKzIvXQKjcOB6Y77lAc1iIrMCpZWpHDme64A9/11wNEESvJbkf3H/+x5iGp5CX0xBPsv/569DGyOMlgAIMBfURvjqB9Vibww6fAL8MThtVcn4j3bXhVRH6X6vGOjiZ+60bbioXRiBzR0xN8/HFAOVcqaQw+9BDiO98JrDaQR640qwRIrQKp7xOM5fhbH8XSeddKxSekhaFuB7vhnh1rbxwKhZifn2d6epqBgQEKCwupqKigvLw8XwXKQ4Ner4+zUp+bm2NhYYGlpSWNOJeVleVsO0Iuwu1250lPHtkDvV5PIBDQbJt3g+wrEsFgkKWlJWw2G62trTk/KYjsR9q3b1+mD2fb8Hg8dHV1cejQIW0ca1lwJwOpMCyQJIn+/n50Oh2tra0ZlVAkDDNVK0GA4amn4s0GEsFoxPDUUxgfflizWVYzd1iPkJrN+J57Do4fR//MMyBJq0TFaCT00EOY7rsv8Wd1Qcw/eydUBtcORzUakWPlZJH/o9dHyfj0zzyjkJJNILIKFfVaYSGIojJ+9dxGnFNfgtcSDi+iuiMlqAL5qt5M8NWvRO+fAFvTtiVta0lWhIAzLmg1lxAbcul2u3G5XPT09ABok9lcl2HnkVxYrVZqa2tZWFjQJJQul4uRkREMBoMmkysoKMj5eUE2Yzdn9ECe9OQc9Hq9pqXeLbIvFWpAp81mo66uLudvbLGBo1euXMlJ61fViae5uTmqf0en06Wk0qMaFqj7SAbUqmhVVRUNK/KxjCOiAhR1VTgcGB96aHMVDr8f4wMPKNk6KxUh0+234Xt5M0KCyonahxP8ylegogLdP/4jpttuUz6vvkeng4MHCT34IKZ77okzYxBEFMJTDLwbeFJxb4s6XoMB37/+K/rLlxENBqwrVRUNPh+Seu9Sc40iSF4iYhN5/JGSOhkIfvrTyK98ZRShkWKJTWzFbROIrAJFOi9KQgWiUVmhJhRCr9cn5Vo1TH0LS+d7oqpIuZD9sxYEQaCoqIiioiKampoIhULMzc0xOTnJ8vIygUCA2dlZysvLd4U0O4+dQ5IkDAYDRUVFWq+l3+9nbm6OK1eu4PP5KC4u1qpAua4E2QjplsHmSc8uQq5PoiVJYnJyEq/Xy3XXXberfuyqEUNLSwujo6M5SQ4iMTs7y+XLl6MCR2NzenIB4+PjzMzMcPbs2Tgb0WSTnlT173g8Ho185kK2k258PM72ec3qxkrQaBSEANavXQefT9DDA8iiiO4HP8D0rnclNjkIBDD/zu+AJBG+8UYM//iPyCrJ0IPwbhTCA/By4DTIPzbC9wQwW1YlZdddh3jddeheegkslujKlcWiVLcAXXe30qMUCbMZQoEoNigDGAQIx/ToFBUphKejI34sSYRKaiKrQJESuCgZ3CYdBiOvcc0ZTvJpRgmWzrvwVF6fkxWfRDAajezbt499+/YhSRLPP/88Xq+XyclJBEHQqkCFhYU5/7zeLHK5tysVSGRkYLFYqKmpoaamBkmSWFpawuVyMTY2psnkKioqsNlsu+66SXdGT17elkdWwO/309XVRVFRERaLZdcQHjUUcnl5WTNiyEVyoEKWZa5cucL8/HycsUQujUuVgomiyLlz5xJO4JIpb0sV4XG5XFy6dInTp0/nzOpVwpwflXAGoq2SE5kGCEFADCKvcQoFwPDMM2tWkgQAjwcSvE/WAadjPlAM/JaeF97+FQwOHwUnT1J29CiFK/ItqaEh3nRh5XX9M88o1t4x41Kyi4zIgZDylAqvHFc4wQQxvGr+kE6oMjiDwaDJ4FQSBMpkRSU/6u8nSroWM9lN5AyXa9k/W4EgCOj1eg4cOMCBAwcIBoPMzc0xPj6O2+3WVvPLy8t3zfMuEdR7Xh4KNnJv0+l0lJaWUlpaCkAgEGBubo7R0VE8Hk9UFWg3VA/TTXryRgZ5ZBzz8/NcvHiR48ePYzQaGRsby/QhJQWhUIju7m6Ki4tpb2/XbvwbBbBmK8LhML29vVgsFtrb2+Nu3DqdLifGFdmHdODAgTUfyIIgJGWVMlWEZ3JykpmZGdrb23OrgTrWejq2GV+1oxaExH0/AlABrHOprRkyGvO32PdJBhOCE4QSHch+0FlBgFDb45yuu4lgMIjT6WRkZASPx0NJSQmVlZXse/RRLHfdFT+eW29VMnUi9q/tN6C8LgsGsJnA440+GLPSthR66J51e3TSgUgZnNFo1MiP2g8kiiLG6W9j6/2AJl2rLPkIcFLbRkJnuC1m/+QSYnuaTCYT+/fvZ//+/ciyrK3mj4+Po9PptJ6O3baaL8ty3qI5Alu1rDabzVRXV1NdXZ3wusn16mE6g0lBsRXPlQXC7SBPerIYsixr8qL29nasVisejycnJs4bYb0g1VyqiKhQA0cbGhqora1N+B6dTkdoLaviLIH6vah9SOshGd9TKgwLVBv3YDBIe3t7Tjr/RBkdJGrGt9mwvuIViT9sBPzAbUbkv9IrTCI2JHWbEETwv+4H6J2XERsPoSszIRU0gFk5PpPJFCVDWVxcVEhQUxPWf/gH9vv9FDU3Y21oUNzl1jM7UGGxxjnEYQQ+AByyEL7pHfGfCTjQecejji2diDVDwO/A1vuBKOna0YUHWQj8LpiU+4VsrkzsDLcLqzywfvaIIAiUlJRQUlLCwYMHo1bzvV7vrurpyOfSRGMn5yP2uomtHhYVFWnVw1ypAqU7mDTf07OLkEssXxRF+vr60Ol0nDt3Trvoc7UKEgm132WtINVcG+Pc3Bz9/f2cOnVKK7knQraTOafTyeDgIC0tLZtyVtpppScVhgXhcJienh6Ki4s5evRoTv3m45Co8T7iteDjjyvSsBhCI4vAfgvBtz+B+GGFJAk//SmmT3xCCTX1eOJ7g0ALGl0LMiDecgvWV/9mdMXm5sS9NDqdjrKyMq0Z2efz4XQ6mXI6CUxP0zAxwaHNnIdwmOBDD2G6917QA0Ef3G6BNoFg2+NxpEY/+QymC3eCYAQ5RLDtccS6mzezp5RAp9OhC0wmCDXVo/dNELJUacGoUvVvI1Zen9PubZvFVgIXI1fzVTKtkqBcd/ZKZ/BkriBZ5yO2eri8vIzL5WJychKIdhLM1u8gEz09qo34bsSeIj25Ap/PR1dXF7W1tdTX10f9LdcIQSRkWWZ4eJjFxcV1g1SznRxEYmJigqmpKTo6OjbM38nWccmyzNjYGNeuXePcuXObloLtZDyqBXAy5Ww+n4/u7m4aGxvjqoe7EWo1yPD00xgffHAllyZI6KF7lOqHOcLFrKMD3+/9nkKAOjtXCIQegkFC73wn0qlTmD/wgYRuaaLJpHCNj3wE0yOPIAQCq05xak7QJuRlVquV+vp66uvrEUWRhdJSZKMxSt4Wu28sFs2C23fjjUqVa78NXYEncRUn4MB04U4E0QesHOOFO/DZr89IxUeFVNCgWNxFQC/IGEoPIxuN0cGoQjG6wjMKWcrEwaYJ253sR5LpQ4cO4ff7tYBLn89HSUlJTuW75Cs96YEgCBQXF1NcXMyBAwfinASzNU8qEz09WeNwmgLkSU+WQV1tX6tqkKukJxwO093djc1mo6OjY92HXS6MUQ0cDYfDUZW49ZDKXJvtQpIkLl68iCzLnD17dksP3+1UelLVv7O4uMjFixc5efIkJSUlSdlmTsBuJ3zvvYTf8Y6NM2jUKlFHxyqBaGhA/+MfY7799oRObpLBgPdv/gbz9LRiXx1rOGA0RuXtbBZ6vZ6K48cJffWrmG6/XbHJDgZBlpEsFnThMO73vx/9HXcgqDLLtSy+I6DzjisVHiLMAASjInXLIOnBbCfQ+himX96BhB69IBFsfxydVcm82lQw6iYd4XIFyeplsVgs1NbWUltbiyRJLCwsaPkuJpNJW80vKChIwlEnH3kjg8wg0kkwMk+qt7cXSZK066a4uDij30++pye52FOkJ5tvLLIsMzIygtPpTGgPrCJV2SiphBpseeDAAaqrqzd8f7b3vqiN/pWVlTQ1NW36usq2Sk8wGKSzsxO73b6lcajY6nhSRXiuXr3K2NgYra2tWK3WpGwz57DVDBr1/WpGTiyZQam0BB5+GMO5c5hOnEj4HkI7c06L610CpCtXcBYWck2WWb58mWKHg8rKSioqKjbs30hUUUEOKa9nEJIk0bd0CvOhf+ZIjZGgrTFh5Sm2FyjyH7UyslsIUCpkXWrjuirP8fl8uFwuhoeH8fv9lJWVUV5eTmlpadZUgfJGBplHojyp+fl5pqenGRgYwGazaVWgteZmqUImKj150rOLkCzHqWRC7UOwWCxbXm3Pdly7do1Lly7FBVuuB71ej38zafQZwPLyMt3d3Rw9ehT7NoIOs+XaUw0Ljhw5suVxqNjKbylVhgVXrlxhaWmJjo6OnG9ozgR04+NKf44vstcEZIOB4COPwLvepeTsJHgPZrPS07NT57QYwqaz26kCqlC+Y9UMQc3kqKyspLKyMnH/htlOsO1xTBfuiOrpSSSDS5fRgSiKdHd3U1ZWRmPjSeQtLJJEEqDIRQNJkgiHw5rtcy4+M9JR4bBardTV1VFXV6dIKleqQJcvX8ZisWir+ZlcLMnL27IPRqORqqoqqqqUfjuPx4PL5aKvrw9JkigrK9OqQKn+7jJhZJC3rM4jZVCrIE1NTdTU1GT6cJIGdUI6Nze3pT4RyL6KiArVgOHMmTPbWgnJlnE5HA4uXboUFZy6HWx2PKkwLBBFkYsXL2IymWhtbc3qKm42I1EmkGwy4XvuOYQTJ9Z8D2Yzvueeg+PHU3p8giBEZXL4/X6cTieXLl3C7/dTWlqK3W6nrKxMu7bEupvx2a9fk9Sk0+hArQrX1tbu6P4eaYkNOw9GjcoLypBZQrob+PV6vWZ4AIqMx+VyMTg4SCgU0iayJSUlaSUheSODVWTLomAkBEGgsLCQwsJCGhsbCYfDzM/Pc/XqVYaGhrBardp1lYoqkCiKaV3Qy1d68kgZtlMFyQWolSur1UpHR8eWHyDZ1tMjyzKXL1/e0IBhI2Sa9MiyzOjoqCah3Gmz5mYqPakwLAgEAnR3d1NdXU1dXV1StrlnEZEJJAoCOkki9MQTGuGJfU+Ua1uKCU8iWCwWbeVekiTm5+c1EmSxWLQqkMViT9zDk0ajA6/Xq1VT1Yl2srCdYFQVhqlvYel8T5Qtdrj2pqQe32aQ6cl+QUEBBQUFmrHG/Py89kxO9UQ2EvlKzypy4VwYDAbsdjt2ux1ZljXyfPHiRcLhcNLJsyiKaZXUeTyeTbm35ir2HOnJBnlbrIvZdiafmX5grAWv10tXVxeNjY3bXtnMNDmIRCSBiwxQ3Q4yOS5Jkujr60MQhG0R0URYbzyp6t9xu9309vamZCK5V7Hw+tcz+M1vcsJqxXbqVEK52lq5QZlEZGAlKA9rp9NJX18f4XCYiooKKisrKSkp0a6/dBkdLC0t0dfXx6lTp1K+oLWZYFTVElsXdGHpfE9UXpCl8y48ldenveKTTc+wSNlkoolsZFN7sifluTDRTxdy7VwIgoDNZsNms9HQ0BBHni0Wi9YLtF0JZbqNDPKVnjySilAoRHd3N0VFRRu6mK0FdbKZLY2YKlTnudOnT+/IQStbSI9K4NYLHN0KMjUu1bBg3759NDQ0JG2isZYbXeSqsyq3SQacTifDw8M0Nzfvas1xOqGd01e9auNzulWzhDRDnXyoEpS5uTmmpqbo7++nsLCQyspK7EU1mFNsdKCe0zNnzmTEMSxRL5BWcfWMIOuMCunRPmBUpG57mPREInYiGylnGhwcTHpTe7aeh0wg10hPLGLJs2qkoUooS0tLqaiooLS0dNPjzPf0JBd50pNGLC8v09PTw6FDh9i3b9+2t6PKv7KF9KiyKYfDsa7z3GaRDfK2zQaObgWZID2q8cKxY8eorEzupCZR1TTSsCBZhEeWZSYmJrh27Rrt7e1ZlaGQy5icnGRmZmZXnlODwRDViLy8vIzT6eTCpAt78Uc4vvCQMtknnNjoYJuYnp5mamoqa85pXC+QfAhBiiF9UoiwqQ7SPOHMlcl+rJwptql9p9bGuT7RTyZ207kQBCFOQrmwsKAtipjNZq1SvV4VKN1zPUmSti3hzwXkSU+aMD09zejo6I6bxyE7SIEKURTp6enBZDIlzXku05We8fFxZmZmNhU4uhWke1zXrl3TVpxTUa6OHU8qHNokSWJwcBBJkmhvb981D8RMQpXX+nw+2tvbs2bxJFWIDCU8ePAgweAZRmfeiGe2j/lQCbblJiodDsrLy7d9LtSFn4WFhaw+pzrrPoLtEe52Ughvy5cQjeUgioTDYfR6fVossXOF9EQiUVP73NycZm28nYDLXDwPqUK6pVzpRKyRhloFGhoaIhAIRFWBIu8f6SQ9mW79SAf2HOlJ981FnbT5/X7Onz+fFBeObCE9qvyrvr4+qQ3lmRqfJEn09/cjiiJnz55N+o0mXaRHzXxyuVxJMSxYC5EW3JGGBcl6aIVCIXp6eigvL6exsTE/MUgCRFGkr68Pq9VKc3PznjynJpOJ/Y3N0NishVk6nU4uX76MyWTCbrdTWVm5aQ2+LMsMDg4iiiJnzpzJ+klbrLudYLZjIToYNdIUQb0PJntcu2GyH1tRjAy4lGVZqwIVFRWtOdbdVN3YKbJRtp8qxNqpLy4u4nK5uHLlCkajUSNI4XA4X4FNIvYc6UknVJepiooKjh8/nrQLKRtIj8vlYmBgIKnyLxU6nS7t49tpUOdmkA7So05qDQZD0gwL1oLa05MKhzav10tPTw8HDhygqqoqKdvc6wgGg3R3d7Nv3z7q6+szfThZgURhlk6nk/7+foLBIOXl5VRWVq6pwRdFkd7eXgoLCzl27FjuTBbM8e52awWjRjrCJTMYdbdNrhIFXM7NzTE5Ocny8jJFRUVaFShSPpQnPavYq+dCr9fH3Yfm5uYYHh5mYWGB0dFRzZp/r5DCVCFPelKEhYUF+vr6UtJLkUnSI8syY2NjzM7OJqV/JxH0en1aZWBLS0v09PRsK3B0K0j1Az4QCNDZ2Ul1dTUNDalPoBcEISWEZ35+XiPUu8nKPZNQ7ZMPHTqU0ms812G1Wqmvr9c0+HNzc8zOzmoN7GqTsslkIhQK0dXVxf79+3eddfpGwajrWWJvFruN9MTCaDSyb98+9u3bp/WVuVwuJicnAbSV/L1U3dgIe5X0xMJqtVJbW0ttbS0vvvgiVVVVzM/PMzIygsFg0K6dhAHNO0AwGMyKXsRUYs+RnnTcZCcmJpicnKStrS0l7j2ZIj1qFUGv13Pu3LmU3ZzSaSt+9epVrly5Qmtra047lqjE7fjx42mxcpZlGYvFwsLCAr/85S8VZyy7fcfncHp6WvvtJLOfai9jYWFBM+XIk8jNQ6/XRzWwu91unE4nXV1dSJJEIBBImrNjNiNRMKq62LGdYFQVu530RCKyr+zAgQMEg0Hm5uYYHx9nfn6egoICDAYD5eXlaQ2izDbkSU88ImWSoCxuqjI4n89HcXExFRUVlJWV7fja2e3ObbAHSU8qIUkSFy9eRJIkzp8/n7LVm0yQHp/PpyWLp1oak44HodrMvbS0tKPA0WzA7Owsly9fThtxUw0LzGYz586dIxgM4nA4uHTpEn6/n7KyMux2+5ZsOdUAWI/HQ0dHR37lM0mYnZ1ldHQ0TyJ3iEjpkt1up7u7m9raWpaXl/nFL35BcXExdrt9T0xa17PEhvWDUSOxl0hPLEwmE/v372f//v2Mjo5qpHp8fDwqf8pms+2pc5QnPYkReQ2YzWZqamqoqalBkiSWlpZwuVyMjY3t+Npxu9150pPH5uD3+zWpQzKzUBIh3aRHtW8+efIkZWVladtvqqAGjhYUFOw4cDSTkGWZK1euMD8/nzbiFjm5UR9OZrM5qiFzfn4+ShKkNoavdXxqX0RBQQEtLS05+31kE1QZ6tzcHO3t7TlN6rMJ8/PzDA4OcubMGW1yIMsyi4uLOJ1OTX6iyuB2+wRirSqQSoTC4bD299gq0F4mPZGQZZnCwkLsdjsHDx4kEAgwNzfH6OgoHo+HkpKSpK3kZzuyKYojF6DT6SgtLdX6qmOvncgq0GaeAV6vd1cHk8IeJD2puMmqpODEiRNaI1oqkS7So+ajpMK+OVNQHecaGxupqanJ9OFsGypRMJlMabFyjtT0r9e/ExvO5na7cTgcXLhwAZ1Op8ngVC1yIBCgq6uLurq6nP4+sgmRNt+tra35ldMkYXZ2lrGxMVpbW6PuhYIgaBOPw4cP4/f7cTqdUZXPyspKysrKdv13sVEVSHWiUp0f86QnvrphNpuprq6muro6aiV/dHQ0pf0c2YB8pWdniLx2ZFnWrh21gqjK5AoLCxNeO3l5Wx7rIrKpP52kQK/XEwgEUroPVaony3JK7JszAdVx7vTp05SUlGT6cLYNv99PZ2dnWqSGsHnCE4tISZC6gqkGs/l8Pmw2G4uLi5w4cSItfUh7AWoVs6SkhAMHDuy6SVGmMD4+jsPhoL29fcPVdovFElf5VOWfVqtVWxRIhQlMNmGjXqBgMBhnirAbIAScCL4xZGsjsnljE6P1yF/kSv6hQ4fi+jkiq0C74RktSdKur2ZtFdvtbxYEgZKSEkpKSlbyyVb7yNxut+YmqBJpyMvb8lgH6kq7wWBIaVN/IqS60pNOqd56SNZK4G6qWC0uLtLb25u2quJ2CU8imM1mzZHm6tWrDA8PU1JSwtDQUJwzVh5bh9/vp7u7m/r6eqqrqzN9OLsCau+f3++nra1ty/f52Mqn1+vF6XTS29uLKIqaJXZJScmuJ6iRVSA10PPkyZNRvUDJtMTOBAxT38LS+R7QKcGv/tZHCdfetO5ntlLdiO3nULNdRkZGorJdUmGglA7kKz3RSOb5iOwjU90EZ2dnecc73oEoirzyla+ktrZ2S6TnW9/6Fvfffz/9/f08//zznD17VvvbZz/7WZ566in0ej1f/OIXueGGG+I+PzIywi233ILL5aKjo4NvfvObKX/+7znSk4wHS6pCOTeLVJKe+fl5Ll68mLZJ9VpQHdx2+n1lW8VqJ+NSneZS5QoYC1Waoh5vsgio2mvyspe9DKPRGOeMBUS5we32yWAy4Ha76enp4dixYxn93e4mqPcOk8nE6dOnd3wdCoKAzWbDZrPR2NhIOBzG5XIxNTVFf38/RUVFVFZWUlFRsat7sBwOh3Yfs1gsUaQnXcGoqYAQcGLpfA+C5APJB4Cl8y48ldevW/HZ7vNAp9NRVlam9dlGZrv4/X5KS0upqKigtLQ048+9zSJPeqKhLgQkG5Fugj/96U+ZnZ3ln//5n/nud7/L4OAgExMT/MZv/AY33HDDuhEHp0+f5jvf+Q633XZb1OsXL17k7//+7+nr62N6eprXvva1DA0NxV2H9957Lx/84Ae55ZZbuP3223nqqae44447kj7eSOw50rNTOBwOhoaGMiqRShXpmZiYYGpqivb29k2nkacKapDnTn7war9IVVUVjY2NWTF5Vse1lYeQ6my2uLjI+fPn01L+l2VZa0JO1k1XkiT6+/sRBCGq1yRSBqfauTqdTi5fvozX69Xc4PZCT8R24HK5uHTpEi0tLbtempAuhMNhLVi6sbExJfswGAxROS5LS0s4nU5Nfx9phpAN965kYGZmhsnJyShzDfU3HRuMGmmHnQtVIME3tlLh8a2+qDMqUrd1SE+yJvqR2S6SJLGwsIDL5eLy5cuYzWatCpTpZ/t6yBsZRCNdGU779u3jne98J3q9nnA4zPXXX8+//uu/ctNNNxEMBnnd617Hb/zGb8S5Ep84cSLh9r73ve9xyy23YDabOXDgAIcPH+b555/nuuuu094jyzL/+Z//yd/+7d8C8Ed/9Efcf//9edKTLVCdsubm5jh37lxGJTjJJj3qZFQURc6dO5cVNx11jNud4Ku5NakIh90Jtkp6RFGkp6cHi8WSFqe5ZMrZIhEKheju7sZut1NfX7/udk0mU5SEQ+2JGBoaoqCgQHODy8vglFyjqakp2tradn2PSLqgLpY0NDSwf//+tOwzUn8f2buhEv/S0lIqKyspLy/PivvzdjA+Po7T6aS9vX3dMagyOIPBEBWMGkmAdhqMmgrI1kaQQtEvSiHl9XWQiuqG2rSuVn29Xi9zc3MMDQ0RDAajqkDZdA7zlZ5opJsEut1uqqqqaGtro62tjT/90z9lYWGBZ599lqeeeorHHnuMb37zmxtuZ2pqil/5lV/R/r+uro6pqamo97hcLkpLS7U5XqL3pAJ50rMJhEIhenp6sNlsdHR0ZPxHmUzSEwgE6OzsZN++fVlTDQHlpr3dMWZz4KhKejYD1bBAbYpONVJFeDweDz09PRw6dGjdUnkiROYOyLKMx+PB4XDseRmcugjjdrs3nETmsXmo1+rRo0czKhOM7d1YWFjQqp9ms1mrAmXzqr0KtVLt9Xq37CYYaYZgNBrjLLHV/05kiZ1uyOZK/K2PYum8K6qnZyMzg3S42BUUFFBQUKCZa6jX0/DwMBaLRbvHZrrfNU96opFu0pPIsvqmm27i6tWr2v+fPn0agE9/+tO8+c1vTtuxJQt7jvRsJ6ypu7ubAwcOZE1zcLJIz8LCAn19fRw/fjzr3LP0ev2myYGKyMDRdMnAtorNkh7VsCBd2UipIjzq6uKpU6coKira0bYEQaCwsJDCwsIoGdyVK1fweDzZL4MLONB5x5EKGsC8NfIXCbXXxGg05nONkojFxUUuXrzI6dOnd3ytJhOJVu2dTif9/f2EQqEoM4Rsu+5lWWZgYABBEGhubt7xtZqsYNRUIVx7E57K67fk3pbuib5er49aSPL5fJqzaSgUoqysjIqKioxcT3nSE410kx6PxxNHep599tktb6e2tpaJiQnt/ycnJ6mtrY16T0VFBQsLC4TDYQwGQ8L3pALZNytMA9Rm8o2gVgyam5uz6iGYDNIzOTnJxMRE2prit4qtVERgVYNvs9myOnBUEIQNxzU9Pc3Y2FhOGxaAco3NzMykTHqVSAbndDqzUgann3wG04U7QTCCHCLY9jhi3c1b3k6kTLChoSEFR7o34XA4uHz5Mq2trVlfPSkoKKChoYGGhgZEUcTlcjEzM8PAwIAWBlxRUZHx616SJHp7e7HZbBw8eDDp9+SNglEzVQWSzZWbIjva+zOYVyQIglYFqq+vJxwOs7CwwLVr1zSLdZUgpUM+myc90UiVkcFaSFZOz4033sjv/u7v8qEPfYjp6WkuXbrE+fPno94jCAKvfvWr+fa3v80tt9zC17/+9bRUjvYk6dkIsiwzNDSE2+1OW9L9VrAT0qMGFwYCAc6dO5eV1RDY2hhzKXB0PTKnVqqWl5fT9t2kwrBAlmUtmDFd0qtEMjin00l3dzeyLFNRUYHdbl8zlC2lCDgwXbgTQfQBSpOz6cId+OzXb6ni4/P5tKpzVVVVao51D2Jqakqzs8+2e/1G0Ov1VFVVUVVVldAFMVPXvboIVVlZmTZyvpVg1GyaWGfTRN9gMMRZrLtcLi5evIgoilrQbnFxcUqup3Q17ucK0n0+ElV61sN3v/td3vve9+JwOHjjG99Ia2srP/rRjzh16hRve9vbOHnyJAaDgUcffVQbxxve8Ab+6q/+ipqaGh544AFuueUWPv7xj9PW1sY73/nOVA1NQ3bOeDOIYDBId3c3paWlWVsxUNOst4pgMEhXVxcVFRUcP348K8emYrOVnlwLHF1rXGqoZEFBAW1tbTlrWBAOh+nt7aWoqCgpcpbtIFIG19TURDAY1LIsPB4PpaWlmgwuLYTMO65UeIhwdRKMitRtk6RHlV6dPHkyJ67zXIAsy4yMjLC0tERbW1vOT7YSuSC6XC5GR0dxu92UlJRoZgipXFBRnzN1dXUZk4RvFIyqLqhlgyNcNpGeSERarDc0NBAOh5mfn2d6elqrKqoLTcmqKqa7spHtyIaenvXwlre8hbe85S0J//axj32Mj33sY3Gv/+AHP9D+++DBgzz//PNbP9AdYE+SnrXkbWofxZEjR3bdSqo6tqNHj265mTwT2KjSI8sy4+PjXL16lbNnz+aMc1Ui0uPz+ejs7KShoSEtmtZUEZ5sDcc0mUxUV1dTXV2tNYU7HA6tiVeVwaXqGpIKGkCOcXWSQ8rrm8C1a9cYGRnJCelVxrHJvilZljX79DNnzmT1AtB2EXvdLy0t4XA4tCBLdUU/mRJaNdj60KFDWeeaGWuJrf6jyssyRYAyKW/bCgwGA3a7HbvdrlUVXS4Xvb29SJKk9ZYVFRVtezzZSgAzhUy4t2VTK0cqsCdJTyJMTU0xPj6elY5fO4XaI5JLY1uv0hMZOHru3LmcuknGjks1k8h1wwK1EnHixAlKS0uTss1UILIpXJVvOBwOenp6kCRJc4NLqhzIbCfY9jimC3dE9fRsRto2Pj6Ow+GIyjXJIzE22zel2sAXFxdz4MCBnJhw7hQ6nY7S0lLtt+n3+3E6nZrUWZUt7cQExOPx0N3dnRP3gFgZXCQJCofDCIKAXq9Py7MlV0hPJCKrik1NTYRCIebm5picnGR5eZnCwkKtCrSV+1ae9ERDFMW03ve3Km/LRex50iNJEgMDAwSDwazucdkOJEliaGgIn8+Xc2Nbq9KTjYGjW0Ek6VGJdrrCYFNlWDA7O8vo6GjOVSIi5Rvqg9vpdGpyoGRmo4h1N+OzX79p9za1rzAUCtHW1pafCGyETfZNqdKrmpqatFRVsxUWi0WzwhdFMS4LS60Cbbb6ubS0RF9fX9Y5322ERDK4SAlcumRwufYci4XRaIwK2l1eXsblctHd3Q2gVYE2s5iU6+cimRBFMa024l6vNyuNrZKJ3JkFpwCqHMdut3PixImc+7Gtt0KkPtzLy8s5duxYzo0tUaUnWwNHtwI1f2hwcBCv15vzhgUjIyMsLi7S0dGRU6Q6EYxGY5wMTs1GsVgsWhVo2zI4s31TPTyiKNLb20thYSFHjx7Nud9uJrCZvimfz6dJr3JB4psu6PX6qOZ11QSkt7cXURSpqKhYt3ldtaXPtUWPRMjVYNRsgiAIFBcXa5XUYDDI3Nwc4+PjuN1uiouLqaiooKysLF+93gDpNjKQZTnnn+MbYXePbg0IgsD8/DwXL17MyoyazUAlBYl+EMvLy3R3d+d0b5JOp9Mm6QAzMzNaX0OuSPTWwpUrV6isrKS1tTUtE1q1gTeZ1Z3IrJgzZ87suglAbDaKOhFUZXCqK9ZO9OuJEAgE6O7upra2NuudCLMJG/VNLS8va7lXeSOItRFrAqLKliYmJlheXqaoqAi73U55eTlGo5HZ2VnNXj9X+io3i1wKRs1mmEwm9u/fz/79+5FlmaWlJVwuF+Pj41Gum7n+XE8F0tnTI8vytgyycg17kvRMTk4yNjaWNllRKqDKv2J/ECo5OHPmTE5rM/V6PcFgULM/drvdWRs4uln4fD4mJyex2+0cPXo05ftLVf+O6nC4b98+6uvrk7LNbIcqg2tsbCQUCuFyuRgbG9NcsdSJ4E4eUB6Ph56eHo4cOZKTCzEZxTp9U2ol4syZM7teupFsxMqWlpaWcDqdjI2NEQqFkGWZ06dPZzwTKB3I9mDUXIAgCJSUlFBSUsLBgwejHAY9Hg/BYBCHw0FZWVlOP+uThXQbGcDulxfuyauqrKyMqqqqnLYoje15UXsAPB5PzpMDUMYXCoW4cOEChYWFabFxTiXUymJ1dXXaAkfVlUlVj54MuN1uzeFwr07MjUajtnIpSRKLi4tauKXZbNbc4LaixVYn5qdPn87pxYpMIlHf1NWrVxkfH9+VlYh0I3LCqtfrcblcVFVVMTY2htfrjTJDyOVn62aQTEvsvbC6vhZiHQb/53/+h6WlJcbGxtDr9VoVqKCgIKef/9tFOi28c9FQYzvI7ZnxNlFYWBglncpFRJIeNaW9uLg458mBimAwyNTUFCdOnMgq++PtYHJyksnJSTo6OnA6ndsOlt0sIg0Lkkl4nE4nw8PD+Yl5BHQ6HWVlZZrznuoG19fXt6l+CFCqsxMTE/mJeTIQ0Tc1NjaGy+Wivb095xeBsgVq5T0YDGoGG/X19UiSxPz8vHaP2C75z1VsJhg1L4NbH4IgYDAYOHToEKBIfV0uF1euXMHn81FSUqL1Au12Uq0inT09wWBwTzx/8k+CHIVKepaXl+np6eHQoUPs27cv04eVFKg3uvLy8pwmPLIsMzg4iN/v59y5c5oFaigU2vjDO9hnJOFJFiYmJpidnaW9vX1PSFm2i4KCAhobG6NkcGo/hBoOWVFRgV6vjzKCyE/Mk4dI57vW1tb8JDNJkCSJ/v5+jEYjp06diiLxkb0ZoJB/p9PJxYsXCYVClJeXY7fbKSkp2RWLcuthvSpQpCmC+vfdfj42i9hKg9lspqamhpqaGq2iroZMG43GqCrQbkU65W1ut3tXn0sV+adsjkKv1+NwOLh27RotLS27YuVdlmXGxsaYnZ3lxIkTzM7OZvqQto1wOExXVxfFxcVR4Yfr5Q/tFJGGBcma6Km25+FwmPb29vwEcguIlMHJsqy5wV25cgWTyUQ4HMZqte5KI4hMQZIk+vr6sFgscRPzPLYPNduopKSEpqamDc9rQUEBDQ0NNDQ0EA6HmZubY3p6mv7+fgoLCzW3uL3g3rVWMKooioTDYW2haq9XgdbL6ImtqPv9flwuF8PDw/j9fkpLS6moqKC0tHRXVYHSTXp2wzxyI+xJ0pPrD0JZlllcXESSJM6dO7crHhzqZEUQBM6dO4fH40kZOUg1vF4vXV1dNDU1xVWqUkF6UmVYEAqF6OnpoaysLCdtz7MJgiBoD+2mpiY6OzsxmUwEg0FeeOEFzQ1uPRlcHutDlfna7XYaGhoyfTi7Bup53bdvH3V1dVv+vMFgoKqqiqqqKi3Dxel00tnZCaARoKQGAmcpIglQMBikv7+f+vr6qB7MvWqGsJUJvsVioba2ltraWi1eQFWImEwmrQqUq0ZVKtLZZ+P1evOkJ4/sgzoRFQSBAwcO7ArCEwgE6OzsZP/+/TQ0NGhp2KnufUkF5ubm6O/v5/Tp0wmtcZNNelJFeHw+H93d3TQ1Ne0a2WQ2wO/3a4RYPa/hcDhKBldcXIzdbtdkcHlsDDVzrbGxMX+9JhHBYJDOzs6kndfIDJdI966RkRE8Hk/SnBCzHYnO607MEHYD1qv0rIfYeAGfz4fL5WJoaIhgMBhVBcrFc5gu0uPxePaEbXie9OQQ3G433d3dHDx4EL/fn7OVkEgsLi7S29sbl5eUShlYqjAxMcHU1BQdHR1rNu8mc1yR/TvJJDwLCwv09/fnM02SDDW1/sSJE5SWlmqvGwyGKFtg1Q1O1a6rDeG5vmqZKqhW38eOHdPkL3nsHGqY69GjR7UJZbIR6961uLioSUCNRqNWBdpNvQbqIt+hQ4eiQrb3uiX2dklPLKxWK3V1ddTV1SGKoiYrHh4exmKxaFWgvWCwsRW43e486dmtyMUS+rVr17h06RItLS0UFRUxMTGRk5WQSMzMzDA6OkpbW1vcQy2XKj2SJDE4OEgwGNQMC9aCTqdLikVpqghPpJNY/qGQPKiW1htlxQiCQGlpqUaKfD4fDoeD/v5+QqGQ5ga3FxrCNwOVoDc3N+8JaUa64Ha76enpSevCR2zfhs/nw+l0Mjg4SCAQoLy8nMrKypxdsYfNE8m1zBDU+77qPrubHOGSRXoiEWl7DYqEy+VyMTAwQCgUoqysjIqKCkpKSnbFOdwJ8j09eWQFZFnmypUrzM/Pc+7cOc05S6/XEwgEMnx020Nk4Oi5c+cSulblSqUnFArR1dVFWVkZx48f33AimoxxpcKwQJZlLl++jNvtzjuJJRk7cb6zWq1RDeEul4upqSn6+/spLi7W3OD24vd17do1RkZG8gQ9yVCJZEtLS0ZXfq1WK/X19dTX1yOKInNzc8zOzjI4OEhBQYEmAc0Vm12Px0N3d/e2iORmLLHV9+Tq5D0d9swFBQUUFBRo19T8/Ly2oGy1WjWClA3XVLrzm/I9PXlkHOFwmJ6eHqxWa5xzVi5VQiIR6Wq2XqZQLpAej8dDV1cXBw8eZP/+/Zv6zE7Glar+HVEU6evr05zE8hWE5EAl936/n7a2th0/0BPJ4JxOJ6Ojo5oUyG637wkZ3OTkpEYkd0NfY7ZAlQFlG5HU6/XY7XbsdjuyLOPxeHA6nfT09CBJ0qbysDKJ5eVlent7OX36NEVFRTvaVjKDUbMJ6QziBOX8qfJJWZa1KtDFixcRRVGrAhUXF2fkHKrP+XQh39Ozi5GNN8VYqKtCjY2N1NTUxP09F0nPVkhCtn9Haom8ubmZ4uLiTX9uu6QnVYQnEAjQ3d1NTU0NtbW1SdlmHtFEsrm5OenXc6QM7vDhw5oUSJXB7dZcFLUi6fF4aG1t3dXN7unG1atXmZiYyPosLkEQKCwspLCwkKamprg8rGyrgC4uLnLx4sWUVc420wuUCwQoFfK2zUIQBGw2GzabTauqz8/Pc/XqVQYHB7HZbFoVKF2/jXQGk4Iib0s019xtyPwdIY84OBwOhoaG1nQAg9wjPao2e6skIRsxPj7OzMwMZ8+e3XIZfDukJ1X9O8vLy/T19aW0UXkvIhgM0tXVRXV19bYsfreDSCmQmouiyuCKioo0KVA2TAK3CzUcU6/X09LSsqvIXKYxMTGBw+Ggra0t566R2DwstQI6NjYWtZpfUFCQ9mtmfn6ewcFBWltb01KB3WowajaRoEySnlgYDIa4yqLL5aK3txdJkigvL9eqQKm6ptKZ0QNKv1le3raLIQhC2jWTG0FNaHe5XFH9O4mQK6RHDRy9du3atkhCNkGSJAYGBgiHw5w9e3ZbN6Stkp5UER61sb65uXlPlLTTBdVJ7PDhw1HOTOlEbC7K0tISDoeD0dHRqId5LsngRFGku7ubsrIyGhsb84QnSVB7Rt1uN62trVkz6dwuYo1A/H4/TqdTk5mWlpZit9spKytL+VidTieXL1+mra0tY8+99YJRIbuqQNlEeiIRWVlsbGwkFAoxPz/P9PQ0AwMDFBYWalWgZEpt00168kYGeaQV4XCY3t5ezGYzHR0dG/74c4H0RAaOnj17NitvaJtFKBSis7OTiooKDhw4sO1J11ZIjyzLhMPhpBsWjI+P43Q66ejoyPdDJBFqA3gydPvJgiAIlJSUUFJSwuHDh/H7/TgcDgYGBggEAlovRGlpadYSCbVyVltbuyfkF+mCLMsMDg4iSdKurZxZLBbNvliSJObn5zUSZLFYtCpQsvuXZmdnGRsbo62tLWukgolkcLEEKJOW2NlKemJhNBqjFpXcbjcul4vu7m4ArQpUVFS0o99UuklPvqcnj7TB6/XS1dVFQ0PDpvsqsp30qFkE1dXV1NfX5/QDVc1HOnTo0I4D+gRB2JD0pKp/R61UAbS1teXEAyZXcPXqVW2Sk00N4LGwWCxRjlgul4uZmRkGBgYoKirSeiGyhQx7vV66u7s5cuRIVI5XHjuDuiBltVo5dOhQTt+fNwudThdlX6yaIfT19REOh5NmBz89Pc309DRtbW1Z8zuKRaQMzmg0xlliZ8IMQRTFnJNWCoJAUVERRUVFUf1lk5OTLC8vU1RUREVFBeXl5Vu+FtJt7ODxePKVnt2MbJG3qQ3x6/XvJEI2k561Ake3g0yv/qi9SGo+0k6xUaUnVYQnFArR3d1NZWUlDQ0Ne2KSkw6o8s25uTk6Ojpy6qGt1+vjZHBOp5Px8XGtF8Jut2csGFINcz116lTO9wFmE1SpYHl5OY2NjZk+nIxBbVxvbGyM64MrLCzUqkBbmaxG9kblkslGNgSjZvpZnwzE9pctLy9rJEgQBK0KVFhYuOEzON1GBl6vN2sUCqlE7jyhdxlkWWZ0dBSHw7GtXpdsJT3T09PaivdOJ0t6vT5jN0JVBnb16tWk9iKtd6NLFeFR+0wOHTqE3W5PyjbzWA2llWU55/shImVwhw4d0nohIoMhVTe4dIzT5XJx6dKlDcNc89gaVJluXioYjdg+uOXlZZxOJ52dnQAaAVpvsjo6OsrCwkLO3ws2CkZV/zvZwai7gfREQhAEiouLKS4u5sCBAwSDQVwuF2NjY3g8HoqLi7UqUKLFskz09OTlbXmkBKIo0tvbi9Fo3Havi06ny4pKlQpZlhkaGsLr9a4ZOLpVZCqrR3WJkiSJc+fOpeVGnCrDgrm5OYaGhjh16tSeWMVJF9QMrdLSUpqamnZd5SyyF0INhlRlcIWFhZobXCrkO9PT00xNTWW9dXKuwe/3a5EB+cWPtRE5WT148CDBYBCn08nIyAgej4fS0lIqKyspLy9Hr9drNuo+n4+WlpZdNXGH9AWj7jbSEwuTyUR1dTXV1dVIkqQRa7WyrhIgm82GIAgZ6enZC3OEPUt6MjVJ8fl8WlNufX19Ro4h2VClU8XFxbS2tibt3GaimqU2TVdWVqZtMqsaFkByLUSnpqY0bXkuu+ZlG/x+P93d3TQ0NGw6lDaXERsMuby8jMPhiJLBVVZW7niVUK1+Lyws0N7enlPyoGyH2ht17NgxysrKMn04OQWTyURNTQ01NTVIksTCwoLmzGYymZAkCZPJlJI8rmxDKoNRdzvpiYROp9Mq66D0QLtcLkZGRvD5fJSUlCDLclorL36/P6ccPbeLPUt6MoG5uTn6+/s5efLkrnnwbCVwdKvQ6XRpJT2qYcHhw4epqqpK+f5SJWeTZVmzaM1PHpMLNVn9+PHj6f8NBxzovONIBQ1gzsxKfeQqeKQMTr3eysvLNTe4rUxgVCcxURQ5c+bMnpn8pANqb1Q2uQrmKnQ6HeXl5ZSXlyPLMj09PYRCIcLhMP/zP/+z7es/V7GWJbZaEVKJz2YIULp7WLIJZrM5ilgvLi4yMjLC3NwcTqdTM5ixWq0pI9aqZHG3I0960oDI/pCOjo6sdnfaClIdOKr29KQDDoeDS5cu0dzcnJaJQaoIjyiK9PT0UFhYuCdWHtMJtc8kVcnq60E/+QymC3eCYAQ5RLDtccS6m9N6DImQSAY3OzvL4ODgppvBVblvYWEhx44dy1+zSYQajpnvjUouVPc7m83GwYMHNTlS5PVvs9m0638vyDRjCRCwZjBqIgKUbreybIVOp6OsrIyFhQUKCwspKirC5XIxPDysZU1VVFRQWlqaNJIiy3JWtUukEnnSk2KIosjFixcRBCEl/SHqako6EWnCsFGI6k6Qjp6e2PDUdD2cUkF4VNlVfX091dXVSdlmHgpUqWBG+kwCDkwX7kQQfYAPANOFO/DZr89YxScRYmVwbrcbh8PBhQsX0Ol0mhtcJGEMhUJ0dXWxf/9+6urqMnj0uw/Xrl1jZGQkL29NMtSFJbWfT0Wi69/pdNLV1QWgWWLvNL8lF6DOc2KrQKoELpEMbi/J2zYDtafHYrFQW1tLbW2tJq90uVxcuXIFk8mk2bAnQ5q2269L2MOkJx1frt/vp7Ozk5qampRk1ag3inSWJEVRpK+vD71en/LA0VTL2yRJ4uLFiwBpC0+VZRmdTsfw8DB2uz1pFbLFxUUuXrzIiRMntDTyPHYOtUnZ4/FkTCqo844rFZ4VwgOAYFSkbllEeiIRmV9x8OBBAoFAlAyurKyMkpISRkdHOXjwYFrkpHsJalZMe3t71mbF5CJEUaSrq4uqqqp1SXrk9X/gwAFCoRBOp5OxsTHcbjfFxcXY7fY1nbt2G9QqkMFgWDMYNRvdaDOJREYGkfJKUHrEXS4XQ0NDBAIBysrKtCrQVuXFe4HwwB4mPanG/Py8NglVL9BkQ230T9dETHX/qa6upqGhIeX7S6W8LRgM0tnZSVVVFY2NjWk1LGhpaWF+fp7x8XHcbjdlZWXY7XbKysq2RbxmZ2cZHR2ltbV1TzQipguqhMVsNmc0sV4qaAA5FP2iHFJezxGYzWZttVIURaanpxkYGMBgMHD16lVEUaSiomJPyIBSDTU3KteyYrIdalWytrZ2y5V0o9GoOXfJsszi4qLmCGcwGJJmBpILSBSM6nQ6CQaD6PV6QqFQSiyxcw2bkftZrdYoebFqsjE8PIzFYtEc4TaaF/j9/l3TdrER8qQnBRgfH9dW2VI5CU2nu5kaOJpKEheLVI1veXmZnp4ejhw5kjbrVlXbLAgCZrNZCzCTJIn5+XkcDgdDQ0PYbDbsdvumQvFUmeH8/Hx+RTfJUB0Jq6qqMu+yaLYTbHsc04U7onp6sknathUsLS0xNTXFuXPnKCgo0GRwXV1dCIIQNQHcK6uPyYAsywwPDxMIBPJmEEmGukjW1NS046qkIAiUlpZSWlrK4cOH48xAysrKqKys3PYiWK5hfn5ek2EajcaMBKNmI7aq4lFtr9VAeK/Xi8vlYnBwkFAopFWBEmWt7ZWMHtjDpCcVD1NVLqXmu6R6lS1dpCeZgaNbQSp6eq5du8bw8DAtLS0UFhYmdduJsJFhgU6n025UsX0Qqh2w3W6PO+/qtWYwGHI+DC/boNr7ZlOYq1h3Mz779Rl3b9spZmdnGRsbo7W1VVtZTCSDu3z5Ml6vV3PD2isTwO1ClmX6+/vR6XScOnUqTxaTCFXhcPjwYW1CmUzEmoEsLCxoxjpWq1VbBNiNfVlOp5MrV67Q1tYWVeVdLxgVtmeJnWvYqYqnoKCAgoIC6uvrEUWR+fl5rl27pl1X/f39vOxlL6OxsRG3272l+dC3vvUt7r//fvr7+3n++ec5e/YsAP/+7//OfffdRzAYxGQy8dBDD/Frv/ZrcZ+///77+epXv6o9Xz/zmc/whje8Ydtj3Qr2LOlJNtQb4759+9Iml0o16UlF4OhWkMyeHrUq4nQ602ZYsFWHttg+CHUFcHBwkGAwSEVFBXa7HYvFQk9PD1VVVWmRGe4lqL1Rp06dSokj4Y5gtmdtD89mMD4+jsPh0FZ0EyFSBpeoCrqX3LA2C0mS6Onp0fpH8oQneVBz9dKVbxS5Wi/LMl6vF6fTSW9vL6IoUl5ervWC5vr37HA4GBkZobW1dc3f82aCUXerDC6ZrQuReWrqdfWv//qvPPHEE7jdblpbWwmHw4TD4U3N806fPs13vvMdbrvttqjXKysr+f73v09NTQ29vb3ccMMNTE1NJdzGBz/4QT7ykY8kZXxbQZ70JAELCwv09fVx/PjxlKwErYVUkh5V3lNSUpLUwNGtIFnjkySJ3t5e9Ho9HR0daTMsUFeotuvQFrkCGA6HtfAyl8tFWVkZVqs17anNuxmq21W+Nyq5UGVXfr+ftra2Tf/+YqugHo9Hk8EBUW5wuT4B3C7C4bDWWJ9xGeYug8fjobu7m5MnT2ohkumEIAjYbDZsNhuNjY3aM2BycpKlpSWKioq0/JZckzZfu3ZNU49s9tjXC0aNtcSOfH+uIlUW3up1dc8993DPPfcwPz/PU089xbe//W06Ojo4fvw4b3jDG3j961/Pvn37Em7jxIkTCV9va2vT/vvUqVP4fD4CgUBWVSn3LOlJ1kNycnKSiYmJtEu/IHWkRw3pTEXg6Fag0+kIhUIbv3EdBAIBOjs702a+AKuGBZC8G6/BYMBgMOD3+zl//jzhcBiHw8Hly5exWCyaVWp+BXx7GBsbw+l05nujkgxVhmkymTh9+vS277uCIFBYWEhhYSEHDhwgGAxGyeB2agaSiwgGg3R1dVFfX5/R+/RuhBpC3NzcnBYZ9GZgMBjYt28f+/btQ5ZllpaWcDqdjI+Pa5bwudALNzs7y/j4OK2trTu6164VjBq52JjrMrh0HHdZWRltbW0sLS3x+c9/nr6+Pn7wgx9wyy234Pf7ed3rXscb3vAGfuVXfmVL2/3Hf/xH2tvb1yQ8X/7yl/nGN77B2bNneeSRR9IW9i1sEEi0a9OKZFkmGAxu+/OSJDEwMEAoFOL06dMZWW2/fPkyNpstqQ88VUrS0tKS8fTu2dlZlpeXOXz48LY+v7S0RE9PD8eOHaOysjLJR5cYkYYFyXzwTExMMDs7S0tLSxSxUUvVDocDh8MBoBGgvdKYuBPIsszg4CCiKHLixImcfThmI8LhMN3d3VRUVNDY2Jiy/UTK4Obn5ykoKNDMQHbrIoAquzpy5Eha1QV7AQsLCwwMDNDS0pIzga6BQACXy4XD4cDr9VJaWkplZSXl5eVZpQSYmZlhamqK1tbWlMnlIy2x1f8Gcs4M4YUXXuDcuXNp2df3v/99+vv7+eQnP6m99trXvpbp6Wncbjd+v18z8Pj0pz/Nm9/8ZgCuv/56Hn74Ya2nR0VfXx833ngj//Zv/8ahQ4fi9jc7O0tlZSWCIPBnf/ZnzMzM8PTTTyd7WAknYHu20rMTBAIBurq6sNvtnDhxImOrKsms9KQrcHQr2ElPz+zsLJcvX6a1tTUtk/+t9u9sZbtDQ0OEQqGEFrSREoimpiaCwaDWBOv3+zUNeGlpaVav/mUCashgUVERx44dy5+fJEK9RzY0NKS8CrEZGVxlZSWFhYW74jt2u9309PTkM7lSgLm5OYaGhjhz5kxOSVzNZjM1NTXU1NRoAZZqJdRsNmu/gUyOaXp6mpmZmZQSHlhfBrfXzBA2i0RGBs8+++y2tjU5Oclb3vIWvvGNbyQkPECUbO7d7343b3rTm7a1r+1gz5Ke7T78VOvmdFYP1kKySI8aOGowGNIW0rkZbCenR5Zlrly5wvz8POfOnUuLVCkyaE29kSYD4XCYnp4eSkpKOHr06Ka2azKZovJQ5ubmmJmZYWBggKKiIux2OxUVFXsiEG89qJPyuro6ampqMn04uwoej4eenh6OHj2aNnt7FWvJ4EZGRvB4PJodcHl5edbc57YC1Wgjm2RXuwUOh0NzEsumHoStIjbAUjVD6O/vJxQKaY6IiayLU4WpqSlmZ2dpbW1Ne+UpkRlCbDBqNlaBNlBhJRUejycpi8MLCwu88Y1v5HOf+xy/+qu/uub7ZmZmtKyr7373u5w+fXrH+94s9vbMZ4uYmppifHw8I/07iaDX6wkEAjvahuo6V1NTk3WNsFu1rBZFkd7eXoxGI+3t7Wk3LEgm4fH5fHR3d9PU1LRmM+FG0Ov1mtRN1YCrzfpms1mTAO2VUDIVbreb3t7ejEzKdzvUSfnp06czLo8FZREgcgV8fn5eC+/LNTtgl8vFpUuX8kYbKYDaZ7Ibe/oKCgpoaGigoaEBURRxuVzaQpiaC5fKYODJyUmuXbvGmTNnMi61SxSMGmmJrfbiZtoRTu1JShc8Hs+W5n/f/e53ee9734vD4eCNb3wjra2t/OhHP+LLX/4yw8PDfOITn+ATn/gEAP/2b/9GVVUV73rXu7j99ts5e/Ys99xzD52dnQiCQFNTE1/5yldSNbQ47NmeHlAaQTfDpiVJYnBwEL/fT3Nzc9asku+050V1nUtn4OhWsLy8zMjICC0tLRu+N1OGBTt1aEuEhYUF+vv7U+oaFNkHJEmS5oS1WyRAa0GVr5w+fTq/Up5kqMYauSANiuyFczqdWf8biMw3ygbp8W7C9PQ009PTKZddZRvUXDin04nT6QTQYhGS9RuYmJjA6XTS0tKSccKzERJZYquLmemuAomiSGdnJx0dHWnZ32c/+1na29u56aab0rK/NCHf07MdqA455eXlHD9+PKsehjuRt6lVq/b29qydoGy20qNKDtNpGR5pWJDMm+HMzAwTExMpX80tKCigsbGRxsZGQqFQnASoqqqK0tLSrCr37xTT09NMTk7mvHwlGzE1NcX09DTt7e05MSmP7YWL/Q2UlpZqbnCZnqxNTk4yOztLe3v7npqUpwPqpDxRv+RuR2QunCoFdblcjI6O4na7KSkp0aSg27nuxsbGmJ+f58yZMznxHMmmXqB0R1FsNZw0l5G/g64D1f3r6NGjWZPMHontkB5JkhgaGsLn82UkcHQr2Mz4rl69ypUrV3aFYcGVK1dYXl5O++TGaDRSXV1NdXW1JgGanZ1lcHCQwsJCTf6Qq7KPyHPb0dGx5yY3qYQsy4yMjLC0tER7e3vOntvY38DCwgIOh4Ph4WHNEj7dMjjVXGZpaSkjvRC7Hep1myuT8lTDZDJF/QaWlpa0AFGj0ahJQTcj7R8dHWVxcZGWlpacPbcbBaNG9vAme4zpJj0ejydPevYCBEFYU942MzOjBRVmq/XvVklPKBSiq6uL0tLSjAWObgXrVXpkWeby5cssLi6m3bAg2YRHNZKwWCycOXMmo99LrBPW8vIyDoeD8fHxqB6hbK0OxkKSJPr7+9Hr9Rk/t7sNsiwzMDAAsKvObWwjuOoG19PTo8ngKisrKSoqStmYVdfGcDhMc3Nzzk4csxHqs0OVq+fPbTx0Oh2lpaWaO6Df78fpdDI4OEggENAMQRLlYo2MjLC8vLyrzm26g1HzpCd12NOkJxHUh43H4+H8+fM5XwlRoQaOHjp0aNuN8enGWuNTrYbNZjPt7e1pmWylqn8nEAjQ3d1NdXU1dXV1SdlmsiAIAsXFxRQXF3Po0CH8fj8Oh0NzAaqoqKCqqiqlk7+dIBQK0d3dTWVlZUpzYvYi1N9gcXExBw4cyMrvP1lIJIMbGxvTJEB2uz2peShqoKvZbObkyZO7+tymG2oulyzLnDp1Kn9uNwmLxUJdXR11dXWIoqjlYg0NDVFQUEBlZSUVFRVMTU3h9Xo5ffr0riE8ibBeMCrsvAokSVJaz5/X680K45l0IHtn9BlAMBiku7ub0tJS2trasv6GuFnSo+a2NDc359SFnajS4/f76ezs1G7A6UCqCM/y8jJ9fX054yJmsVior6+nvr6ecDgcNflTeyCyxQpYdb87cOCAFqqWR3Kg3ierq6upra3N9OGkFYlkcJF5KDt1RBRFke7ubsrKymhqakruwe9xyLLMxYsXMZlMHD58OOuf79kKvV6vVTvVXCyn08kLL7yAKIrU1dWxvLxMcXHxnjjHqbDEzkSlJ1sVTcnGniY9kT/I5eVlenp6OHz4cM5MkjYiPare3uVycfbs2ZxoMI5E7A1TNSxIp9tcpIY3mZN51emqubk5J282BoOB/fv3s3///qgeiEuXLlFQUKBN/jJxzS0tLdHX15dS97u9Cp/PR1dXF4cOHcrKPsd0IpEMzul00tfXhyiKmhPWZiuhqvx4L5LJVEOSJHp7e7X8pr0wGU8HVEOQmZkZKioqOHLkCHNzc0xMTLC8vKxlw5WXl+dsT+hWsJYl9lbNEDJhZJBLC+I7wZ4mPSrUZviWlpac0jWuR3oiM2s6OjqyYvV9J5iZmWF0dDRtGUmpNCyYmJjA4XDkjNPVRoic/Kkrfw6Hg87OTnQ6ndYHlI7vLdI2ORuytHYTlpeX6e3tzZPJNaDK4FRHRJfLtWkZnBqW29TUlDOLbrkCtXpWXl6el7kmGWo7gCRJnDhxAkEQ2LdvH/v27dOy4VRFgE6n0ypENpttTxDPzZghJKoCpZv0BAKBPZPXt6dJj/qDXV5eTlszfDKh0+kSGjGoErDa2tqsCxzdKmRZ5tKlSywtLaXNbS5VhEfNe5Ikiba2tpwnookgCAKFhYXaimogEMDhcGgNsOrqd0lJSdIfehMTE8zOztLR0ZFzv+Vsh5pv1NLSkpOVyXTDaDRGVUIXFxc1Qh4rg/N6vXR3d+eMzDWXEA6H6erqYt++fVnXM5nrUPujgIRxHoIgUFJSQklJCYcOHSIQCGhSUK/XG2WGsBecCdcyQ1CJkPrfer2ecDic1nOi5hHtBexp0jM6OgqQtmb4dEANHD158iRlZWWZPpwdIRwO4/P5kCQprYYFqSA8oVCInp4ebbVxt1xvG8FsNkc1wLpcLqampujv76e4uFizw97JDV4lxoFAgPb29j1z804Xrl69yvj4eD7faJvQ6XSUlZVp92Ov16vJ4ILBIIFAgKNHj+b8/TrbEAqF6OzspL6+nv3792f6cHYVZFmmv78fg8HAkSNHNvU8M5vN1NbWUltbq0UjOJ1OhoeHk9IPl2tYrwrk9XqxWCyEw+GUB6Ou5WC8W7GnSc+BAwe2He6ZjZicnGRiYiKrA0c3C7VaZTAYOHbsWFr2mSrDAnUl9+DBg3tauqLX66mqqqKqqgpZlrXV7ytXrmw7C0WVcdpsNk6fPr1nyGS6MDY2hsvlygdjJhEFBQU0NDRQXFxMf38/Bw4cYG5ujrGxMS0QcqcLAXsdwWCQzs7OvFwwBUiGIURkNAKsLgRcvHiRUChEeXl5yhQB2YjIKtDMzAxut1szMklXMOpeOM+wx0nPbvmSVdlUIBDg/PnzOf+wjKxW9ff3ayQklZBlmXA4DCTXsGB+fp6BgQFOnTpFcXFx0rab6xAEQcuBOHLkSFQWiizLVFZWYrfb19V+B4NBurq6qKmpyTd+Jxmq9DcUCtHa2pqvniUZKtlva2vTVrZjFwJMJpPWD7dXVr+TAXXB7MiRI9qkOo/kQJZl+vr6sFqtHDx4MGnPZXUhoKGhgXA4zNzcHNPT0/T391NYWKj1Au122fLMzAxTU1O0tbVFyeAi/4HkBqOm2x470xA2KG3t6rqXJEmEQqFMH8aO8N///d+YTCbKy8uTehPKFKanpxkbG6O1tRWr1covfvELzp07lzIilyo5GyhjmZycpKWlJT9p2QKCwSBOpxOHw4HP56OsrAy73U5paal2c/Z4PPT09OQnNimAJElaWG7e2jf5mJmZYXJyktbW1nUncerqt9Pp1HKx7Hb7nrEC3g7Uqvrx48e1YM08kgP1vmCz2Th48GBa9qkGZDudTlwuF4BGgAoLC3fV7+Dq1avafWGtqnok6VElcSr0ev22yIvb7eatb30rzz333PYOPHuR8OLY05WeXIfb7cbr9XLw4EGqq6szfTg7gtqX4Xa7owwLVIe6VJCeVDq0DQ8P4/V66ejoyPnKW7phMpmoqamhpqYGURSZm5vj6tWrDA4OUlRUhNVqZXZ2Nudyp3IBauO33W6noaEh04ez6zA+Po7T6aS9vX3D+0Ls6rfL5dKsgNV+uPLy8rzscAVut5uenp58VT0FUC2/i4qKOHDgQNr2GxmQffDgQYLBIC6Xi5GRETweT0rCgTOB2dlZJiYmaGtrW/f3rJKa2GBUVQK3HRmc2+3eU+Y0+btljuLatWsMDw9TWFiY8yvd4XCY7u5uCgsL40JhEwWUJgOpIjyRPSYtLS27aiUqE9Dr9ZrER82dmpycxGg0cunSpbz8J4nw+/10d3fT2NjIvn37Mn04uwqyLGuuVduRCxoMhigrYFUGNzIygtFo1Prhcr2Xc7tQs7mam5tzKnYiFyBJEj09PZSUlGQ8MNdkMkWFAy8uLuJ0Orly5QpGo1GrAuVSXMG1a9c0o5itLmCoJgcGg2Hbwaher3dP/WbypCfHEBs42tvbm9NmDD6fj87OThobG6mpqYn7+0YBrNtBqgwL1EljXV1dwrHksX3Isszo6CiLi4u8/OUvx2Aw4PP5cDgcUWGQVVVVu072kA6ocsFjx47lXcSSDFmWGRgYAKC5uXnH12ZkPxwo91Cn00l/f78mg6usrNwzTeALCwsMDAzks7lSAEmStIyjbKv8xroiqr8Dtb+5vLycysrKKFl0tuHatWuanH+nFdu1glETWWJHVoHylZ49hFx7IKhVBJPJpAWOpoIUpAvz8/NcvHiRU6dOram/TnalJ1WGBepK4/Hjx/OTxiRDkiRt0njmzBnte7NarZr8JxQK4XQ6NdmD2gdUVlaWtQ+8bMHCwgL9/f35VfIUQJUFqX0QqXjmWK1W6uvrqa+v15rAI23hVTe43SiDU/OjWltb89XeJEMNda2srMyJvL/I34Eqi56dnWVwcJCCggItHiFbbPcdDocWuJ4Kg4bNBqMuLy/nSU8e2QfVkUbNPFGRq6RnamqK8fHxDe21dTpdUsaXSsOCa9euMTIykl9pTAFU6eNG+UZGozFK9jA/P4/D4WBoaAibzabJf3a7+89WoV67kS5ieSQHoijS1dVFZWVl2lbJDQZDlC380tKSNrlS5T92u31XyOBUeV97ezsmkynTh7OroF67VVVVORnqGiuL9ng8OJ1Oenp6kCRJq4ZmyhREvXZTRXhisV4w6rPPPsvw8HDKjyFbsKfd22RZJhgMZvowNoRaEUkUONrf38++fftyJslbtcL1er00NzdvuPqYjPGl0rBgbGyMubk5mpub8xPqJMPv99PV1UVjY+O2wwVlWcbtduNwOHA6nVEPw90w8dsJJicnuXr1KmfOnMlfu0mGaqdeV1eXNSYzqvzH4XAQDAY1N7hclMFdvXqViYmJDR3w8tg6RFGks7OT6urqXSnTDoVCuFwunE6nZgqSzmqo2oPU2tqacbL+f//v/+ULX/gC3/ve93ZjnlXCm9qeJj0AgUAg04ewLiYnJ5mcnOTMmTMJJ2lDQ0OalCfboTpDFRcXb9oKd3BwUAsq2w5SRXgkSaK/vx+dTsexY8fyEqokY3l5md7eXk6cOJFU61m/34/D4cDhcOxZG2C1qd7j8XD69Omcdj3KRqhk/dChQ1RWVmb6cBJClcE5HA6WlpYoKirS5D/ZLoObmprSyHq2H2uuIRwO09nZSW1tbdaQ9VRCNQVxOp3Mzc2h1+ujzBCS/UxwuVwMDw/T1taWccLzr//6rzz00EP84Ac/yJlF8y0iT3oSIRgMssE5yAgiA0ebm5vXnJhcvnwZm8227ZXwdMHr9dLV1UVTU9OWbqbDw8MUFRVty00qVYYFwWCQnp4e7HY79fX1e2aynC44nU6Gh4dpbm5OqdZYtQF2OBwsLy/vGvvT9aCSdb1ez7Fjx/LXbpKhGkLkUk5MpAxOnfipctBsk+uOj4/jcrloaWnZtb/RTCEcDnPhwgXq6+uzfj6RKvj9fi0by+/3U1pamrTe0GwiPM8++yyf+tSn+MEPfpC1CzNJQJ70JEI2kh5VGrGZwFFVq53NifRzc3P09/dz+vRpSkpKtvTZkZERzGbzlsvsqTIsUCc1hw4dyonqWq5BlVy1tLSk9cGg2p+qEz+r1apN/DL9gEoW1MbksrKydfuj8tgeVDOT06dP53R+lFoNdTqdBAKBKDe4TFW0VddSt9vN6dOn85X1JCMUCtHZ2UlDQ0Pern4Fam+o0+lkfn4ei8WiVYG22v84NzfHpUuXsoLw/Nd//Rcf//jH+cEPfrDbv+s86UmEbCM9y8vL9PT0cPjw4U1pLCcmJpBlOevsJFWo8rztuuuMjY2h1+u31EypOpQk27DA5XJx6dIlTp06ldOTmmyEGujq8/k4depURldx1cZXdeIHaH1Auepyoy6k1NbW7kqdfqahuoitJUPOVaguWA6Hg8XFRYqKirT+h3T10qj3hmAwyIkTJ/KEJ8kIhUJcuHCBpqam3djXkTSoZghOpxNRFDVL7I164tR7Q1tbW8ad4372s59x77338i//8i974TmQJz2JkE2kRw0cbWlp2bR17PT0NIFAIK0pyZuBmk2xkTxvI2yF1KXSoW1ycpKZmRlaWloyfuPabRBFkb6+PiwWC0eOHMm6CkQgENAawP1+P+Xl5VRVVeVMA7jX6+X/b+++A9sq7/3xv72yvIdkO45XvOIlOYEAYZSVMLLkDOavzAYu3NvCLYVLA12UQqGl0Pa2hQ4KLau3lpxlh0AadoGWYUneceI9NWxLtqx5zvn9wfccnOAkHjo6R9Ln9VebYT0RGs/7GZ+P0WhEUVFR0DcyliOTyYTu7m6o1eqQ/mzgOA4TExMwm82wWq3C/QeFQiHaMTiO49De3g4AdBxTBB6PB3q9Hvn5+XRyYQ5OvhMXFxcn7AJNXwwYGxtDe3u7LALPv/71L3z729/GgQMHgqIEuR9Q6JmJ1+v1ax+Y+eA4Dp2dnRgdHYVarZ7T9ufIyAgmJiZQWFgo4gjnxuv1wmg0IjExEQUFBQv6opptqJveiTgyMtKvFdqOHj0Kj8eDsrIyOkfuZx6PB0ajEenp6UHxQXzyyndCQoJwAVyOrw3+yFV5eTkSEhKkHk7IGRgYwNDQUFhWwOPvP5jNZqEZJF8Nzh+7MSzLoqWlBYsXL5514Rsyex6PBw0NDbIuuBEM+MUAi8UCq9UKAEhLS8PixYvR09Mji3YAn332Gb75zW9i3759yMvLk3QsAUShZyZShx6GYdDY2IjFixfPqwoY/0YrKSkRaYRz43A4YDAYsHLlSr9chhweHobD4UBBQcEp/4xYBQt8Ph8aGxuRkJAgWmPBcMbvQATr/Si+8g+/8r148WLhGJzUq3rAl8cxVSqV7C6kh4Lu7m6Mj48vaCc7VJy8GHCqle/ZYllW+OyV2ymGUOB2u6HX61FYWEi7v37m8XjQ29uLvr4+LF68GMnJyUhLS5OsQI7BYMB//Md/oLa2VlaL4wFAoWcmUoYep9MJvV6P7OzseTcAGxsbw9DQEMrKyvw8urmzWq1oa2tDZWWl31aVzWYzxsbGUFxcPOPvTw88/jzr7XQ6YTQaF9Qjhpza+Pg4WltbQ2oHYmpqSiiHzbIs0tLSoFQqERsbG/DAPDg4iIGBgTnvHJMz4zgOHR0dwu4v3TE50cnH4CIjI4VjcLO5E8cX3EhNTZXtXdVgxjc6Ly4uDtVSxZKy2WxobW0V+vCMj48LJbEXLVokFMgJxN2/5uZmfOMb30BNTY1sFsYDiELPTHw+HxiGCfjjnq7h6FzY7Xb09PSgsrLSj6Obu97eXuGYhz+3cq1WK0wmE0pLS7/ye2IVLLDZbGhpafF7jxjyhZGREeEOhNTb/mLxeDzCa3dqako4+pOUlCTqJJnjOGEHgsr6+h9f8js6OhrFxcW0+zsLbrdbKAricrmEvnIzvRf4Xm4ZGRmyrkgarPjAU1JSsqB5B5kZH3hOVdDk5AbBfDEEMb4X2tracOutt+K1115DeXm5X392kKDQMxMpQk9fXx8GBgbmXdFsOofDgY6ODlRVVflncHPEsiza2trg9XpFaXQ4Pj6OgYGBE960YhYsGB4eRk9PD1QqVUhVYZIDjuPQ09OD0dFRVFZWhs0dCJZlhaM/4+PjiIuLE1b7/Nlckb/0zTAMVbkSAcMwaGpqQkJCAvLy8ijwzAPDMBgbGxPeC7GxscJ7AYBw8oF21/3P6XTCYDAEVQ+pYMLfn6yqqprV3IE/EmqxWIT3An8kdKG78x0dHbjpppvw0ksvQa1WL+hnBTEKPTMJZOgRIyC4XC40NzfjrLPO8sMI58br9cJgMCA5OVm0Oy8n72SJFXj4PhA2mw2VlZXU6dvPWJbF0aNHw35CPv3oj8ViQUxMjF+OO/AV8JYtW7bg4iHkq/gdiPT09HkfRSYn4jgOk5OTwpFQh8OB9PR05OXlYdmyZfQa9iP+/mRpaemce+WRM5uYmEBTUxPUavW87k/y7wW+JDYAoT9WfHz8nN4L3d3duP766/HCCy9IMi+UkRmftLCf2QXqg5Xvk5GamorS0lK/PW5UVJQkx/P4ggUFBQWiNriKjIwU/n1iFSxgGAYtLS1YtGgRqqqq6MvWz/iCEImJicjPzw/r5zciIgIJCQlISEhAQUGBcNyhtbUVXq9XuPswly86fvEhIyODJuQi4Mv65ubmhnozv4CKiIhAfHw8YmJiYDKZUF5eDp/PJ/TrOt0xODJ7DocDRqMxpO5PyslCAw/w5XshPj4e+fn58Hq9sFgs6OnpweTkpFAlNCUl5bQLsn19fbjhhhvwxz/+MdwDzymFfegJhLk2HJ0LKUKPxWJBe3u7XwsWnEpUVBRYlhUt8PBhNCMjIyhKJgcbt9sNg8GA7OxsZGZmSj0c2Vm6dCmys7ORnZ0Nr9cLq9UqfNElJSUJX3SnmvS5XC4YDAbk5+dTY0ER8EeCqMeROPgdiOlHrrKysoRjcCaTCe3t7cIxuNTUVCrMMQd84KmoqKCG2iKYnJxEU1OT3ytkxsTEIDMzE5mZmUKVUIvFgu7ubkRFRSEhIQFOpxMqlUr4O4ODg7j++uvx29/+Fueee67fxhJqKPSIbGRkBMePH59Tw9G5iIyMDGhz1Z6eHgwPD+Pss88OSFlePtT5fD5ERET4dcWP/8CiCY04+OeXqgTNTkxMDDIyMpCRkQGWZTE+Pg6z2YyOjg4sW7YMSqXyhBLAk5OTaGxspIIbIuGf37KyMjoSJAL++Z1pQs43Pk1LSzvh6I/BYAAA4UioFJURgwX//FZWVooy9wh3/POrUqlmVZVwviIiIpCUlCR8xrtcLrS0tOChhx7C0NAQ1q5di4svvhjPPvssnnnmGVx44YWijSUUhP2dHpZl4fV6/f5z+YajY2Njojeu+/DDD3H++eeL9vOBL6sWMQyDioqKgBw34DgOPp8PH3/8MRISEqBUKv3WBNJiseDYsWOorKwU9QMrXPE9YioqKugLd4E4joPD4YDJZILFYkFkZCRiY2OFzxZ6/fofX1Jd7AlNuOIvfc/n+fV4PEIFrKmpKeEYXHJyMh2D+3/4I1cUeMTB76BJ/fw6nU5otVo8++yzmJycRGVlJTZt2oSNGzdi+fLlko1LJqiQwUzECD0+nw9NTU1YsmQJiouLRf8gFjv08EfA0tLSAla1aHrBAuCLD3GTyQSr1YolS5YITSDnc9Sht7cXJpMJKpWKjkqIgO8Ro1KpZNGkM9T09/eju7sbS5cuhdfrRWpqKhQKBRITE2nV2w/4BRF/VNckXzU+Po62trZTlvWdC5ZlhWpwY2NjWLZsmbALFK6f7XzgocAuDrkEHuCLxcXt27fjkUcewdVXX4329nbU19fj4MGDcDgc2LBhAzZt2oS1a9eGY/sCCj0z8Xfo4RuO5uTkBKzPgJihZ3JyEkajUZT7SKdypvs7DodDqPgTEREBhUIBpVJ5xi9QlmXR3t4OlmXDuoKYWPjdzcnJSVHKl5MvArvZbIZKpUJMTAwYhoHVaoXZbIbdbkdiYqJwD4ie/7kbHh5GX18fNXUVidVqxbFjx0Tp0cXviPKVEQGc0BQ1HBYE+B20hVyqJ6cmpztS4+Pj2L59O3bv3g2NRvOV37fZbDh8+DDq6+uxatUqPPjggxKMUlIUembCcRw8Ho9fftbo6KjQZT6QZ+w//PBDrFu3zu8f6mazGUePHoVKpQrYG3yuBQv4xncmk0mofqVUKhEXF3fC3/V6vWhsbERycjL12BABy7JoaWlBTEwMNW0UAcdxOHbsGFwuF8rLy2cM7BzHCfeARkdHhR3RtLQ02nGbhb6+PiFQUsl6/zOZTOju7hY61YuNPwZnsVjgcDhC/hjcmRpjkoWZmpqCwWCQReCx2+3YsWMHvv3tb2Pnzp2SjkXGKPTMxF+hx58NR+fq448/9uv2Jd9E0mQyBewLin9cvmDBfCbNPp8PFosFJpMJDocDKSkpUCgUWLx4MZqampCXl0clZ0Xg9XphNBqhUCiQk5Mj9XBCDh8oFy1ahKKiolm/N6bviALht+o9W3yPromJCVRWVobkhFhqQ0ND6O/vR1VVlSRNifljcBaLBaOjoyF3DM6fRwbJV/GBRw5lvycnJ7Fz507cfffduOGGGyQdi8xR6JnJQkMP33DU5/OhvLxckiMln3zyid+OY/ATLAAoKysLWMECfzccZVkWo6Oj6O/vh9VqRUpKCrKysvxWCIF8wel0wmg0Uslkkfh8PhiNRqSmpiI3N3feP2f65W+n0yksCCQmJob1JJ/juBOOvFIY9L/+/n6MjIxArVbLYgeNPwbHvx8ACJXiTj4hEAzGxsbQ3t5Od9BEwpetLysrkzzwOBwOXHfddbjllltwyy23SDqWIEChZyYLCT1SXPCfyeeff47S0tIFr/DwTfiUSiVyc3MDXrDAn/13gC8u1Pf396OyslI4BscXQuDL/4bCKp9UbDYbWlpaqKSvSPgeRzk5OcjIyPDbz2UYBqOjozCbzbDZbIiPjxd6oMhhUhooLMuiubkZS5cuRUFBQdBNdoNBT08PxsbGUFlZKdvFJo/HI9yLczgcQn+s5ORk2Y6ZNzo6iqNHj2L16tV0hFUEfOApLS2V/DvO6XTi+uuvx7XXXos77rhD0rEECQo9p+J2u+f8dyYmJmA0GlFcXAyFQiHCqGbPYDCgoKBgQZVE+AaqRUVFAfv3iBV4OI7D8ePH4XA4ZrxQP738b0REBJRKJRQKBR0LmAOTyYSuri6oVCp63kTgcDjQ2Ngoeo8jjuNgt9uFBYGYmBihMmIorxozDAOj0YiUlJQF7aCRmfFFTfjP4GDZTZzeH2tsbEzW9+L4ohBVVVWyG1socLlc0Ov1sgg8brcbN954IzZv3oz//M//pAWa2aHQcyoej2dODT7Fbjg6V01NTcjOzp73G9NkMuHYsWMB/ffMtWDBbDEMg6amJixbtgyFhYVn/Lkul0u493C6QgjkSydXECP+xe+gSXFhdmpqSjj2wzCMcA8olN4PXq8Xer0eWVlZ1MtCBBzHoaOjA16vF2VlZUH7uuE4DlNTU0I1OJZlZfN+sFgs6OzsDOid23DCB55Vq1ZJ3vjZ4/Hg5ptvxqWXXor//u//Dtr3kwQo9JzKbEMPv4Ngs9lkNeFrbW1Fenr6nFeEOY5Dd3c3LBZLQEu08gULAPh1BdDtdsNoNM57MuP1eoUJH18IQalUIikpiT5o8MV/t6NHjwqTmWBZvQ0mZrMZx48fl8WF5JPfD6FQ/crlcsFgMGDlypWS79CHIo7j0NbWhsjIyJCr4njy+yEpKQlpaWkBLw9vNpvR1dVFgUckcgo8Xq8Xt99+O8455xz8z//8T0i9nwKAQs+pzCb0+Hw+NDY2YunSpSgpKZHVi+/o0aPChGS2+PPskZGRAetZI+b9Hb4h26pVq5CcnLzgn8cXQjCZTLDZbEhISBDuPcj9nLcY+B20uLg4rFy5Ulav/1AxMDCAwcFBWfaIObkJZGxsrHDsRy6LP2cyNTUFo9GIkpISv3xGkBPxRXCWLFkS8nekZjoGx+8CiXnUzGQyoaenR7IqeKHO7XajoaFBFp8RPp8Pd9xxByoqKvC9730vpN9PIqHQcyperxcsy57y9/lyhYFsODoXx48fR2xs7KwvO7vdbuj1emRkZCAnJyfoCxaYTCZ0dnZCpVKJ0pCN4zjYbDaYTCaMjo5i6dKlwr2HcPjiWegOGjk9vmSy3W6X9YVvHsdxmJychMlkgtVqRVRUlPB+kHp36lT4RRE59NgIRSzLorGxEQkJCcjPz5d6OAHHV4OzWCxgGAapqalQKBSIj4/323fdyMgIent7KfCIhA88Yt+jnA2GYfCf//mfyM3NxaOPPkqBZ34o9JzK6UKPVA1H56K7uxsxMTGzCmR8AYaSkhKkpaUFYHTiFizo6emB1WoN2HHD6V2/zWZzUEz4FoK/UF9UVITU1FSphxNy+ONAHMdh1apVQXls7OR7cfyELyEhQRZf1nxJX7EWRcIdwzAwGAxQKBTIzs6WejiS83q9QjW4yclJJCYmQqFQLOgY3NDQkNAHMJwqLAaKx+NBQ0MDioqKJA88LMvi3nvvRWpqKp544omg/E6QCQo9p3Kq0NPb24uhoSGo1WpZVzLq6+sDx3FnbAwpRQEGsQoWsCyL1tZWRERESDpZ5Cd8JpNJuPitVCpDogEkP1msqKiQRcGOUMMwzAmr48H+egG+OJLBT/gmJib8MuFbCLPZLFz4pgpX/ufz+aDX67F8+XLaBZ4By7Kw2Wwwm80YHR3F4sWLhWOhs51TDA4OCvMQCjz+xweewsJCyRf2WJbF/fffj8WLF+OZZ54RZV4zPj6OXbt2oampCREREfjzn/+MdevW+f1xZIBCz6mcHHr4CTXDMJI1HJ2LwcFBuN3uUx4r4I/PWK3WgG6Ni1WwwOv1wmg0Ii0tLWDH82Zj+kXXqakpoQFkMBZCGBoaQl9fH9RqNU0WReD1emEwGJCZmSnLI7P+MP3ew+joKJYtWyZM+AJxZ2lwcFC4I0XHgfyP7+uWm5uL9PR0qYcTFKZXg+OPwaWlpZ1yV3RgYEBo7Cr3eUgw4l/DBQUFsgg8Dz30ELxeL37729+KtpB7yy234KKLLsKuXbvg8XgwNTUl21NMC0Sh51R8Ph8YhgHw5ZtAoVBI2nB0LkZGRjAxMYHCwsKv/B7DMGhubkZ0dHRAd0QYhhHl/g5/3KqgoEDW1ZdObgCZkJAApVIp2Yr3bPEB2WazobKyklYWReB0OmE0GsOqgtj0Y6F8fyz+4ndsbKzfH6+npwejo6NQqVSyfr8FK/5eaEFBQcCOSYca/hicxWIRdkXT0tKEYjn9/f0wmUwUeETi9XrR0NCAlStXSv4aZlkWjzzyCEZHR/HHP/5RtHmazWZDVVUVOjs7g2Juu0AUek6FDz1yajg6FxaLBVarFSUlJSf8Ov/FlJmZecajb/4iZsECvvt0eXl5UF1Gnl4IwWq1YtmyZVAqlbKrfMXvcEZGRqKkpITOEouAv1BfVlYmecM7KbndbmFX1OVyCfeAEhMTF/SZwbcVcDqdKC8vp9ewCPgu9XK48B0qOI7D+Pi48F3OnzxRq9WiLAqEOz7w5OfnSz7X4zgOjz/+OHp7e/Hiiy+KGnD1ej3uvPNOlJWVwWAw4KyzzsKvfvWrUH2NUeg5FZ/Ph4GBAXR2dgblh8zY2BiGhoZQVlYm/JrdbkdjYyNWrVoVsG1bMQNPf38/hoaGoFKpgvq4Fb/ibTKZYLFYEBUVBaVSCYVCIem9Ma/Xi8bGRqFDfRisAgUcH9orKyuD7jNGTAzDCPeA7Hb7vMvDcxx3Qmin17D/8TvtcuhSH6p6enpgsViQlpYGi8UCn88nu+IgwYwPPHl5eVAqlZKOheM4PPXUU2hra8NLL70k+smKTz/9FOeddx7++c9/4txzz8W9996LhIQEPProo6I+rkQo9JxKT08PBgYGZNVwdC7sdjt6enpQWVkJABgeHg54gBOrYAHf3dvlcgXF/aq5crlcMJlMMJvNkhVC4Bs25uXl0dl8kQwPD6O3t5fuSJ0BvytqNpthtVqFi99n6n/Cl0yOj48PmaIQckNlv8XX3d0tHC3mdylPLg4S7j3jFsLn86GhoQG5ubmyCDy//vWv8emnn+Jvf/tbQOaew8PDOO+889Dd3Q0AeP/99/HEE0+gvr5e9MeWwIxfAnRgH0B6ejrS09OD9osyKipKCBydnZ0YGxvD2rVrg75ggc/nQ1NTE+Lj41FZWRm0/31OZ8mSJcjJyUFOTo5QCIE/npOSkgKlUrngIz+nY7fb0dzcjNLS0lC9zCg5vqz6mjVr6I7UGURERCApKQlJSUkoKioS+p80NjaCZVkhAE1fFPD5fDAajVQyWUQ2mw0tLS1QqVS0SymSzs5OTE5OnhB4ACA6OlqYo0xfFOjq6kJMTIxQHCQUWyb4Ex94cnJyZBF4nnvuOXz00UfQarUBm6tlZGQgOzsb7e3tKCkpwZEjR044IRQOaKcHXxyv4CftwcjlcqGpqQkxMTFYtGhRQO9jiFWwwOVywWg0Ijs7G5mZmX77ucGCL4RgMplgt9tFKf1rNpuFEubUv8T/OI7D0aNH4fV6UVZWRvdLFsjj8XylOmJycjK6u7uRk5Mz6+bMZG740vVqtZom1iLgFyudTuecPyecTqdQHESOPbLkgi+tnp2dLflpBo7j8Pzzz+PQoUOora0N+LF2vV4vVG5buXIlXnjhBSQnJwd0DAFCx9tOhWVZeL1eqYcxb5OTk/j4449RUlISsJVOMe/v8KuKtPvwBf6Sq8lkwujoKGJjY4XVvfmuEPX392N4eBgqlSog5YPDDcuyaG5uxpIlS1BYWEgTED9jGAYjIyM4evQooqKikJSUBKVSidTUVNpN8yN+55n6HImDL7zhdrtRVla2oM8J/hicxWIR7sbx1eDC+T3BB54VK1bIYmHkr3/9K3Q6Hfbv30+LCOKi0HMqwRx6bDYbmpqawLIsLrroooA8Jh94GIZBZGSkXyd0IyMj6O7uhkqlog+EGXAch8nJSWF1Lzo6WjjyM5sVI47jcOzYMaG6FZ0J9z+fzyd0qA9U1cRww1+oX7VqFRITEzExMSG8J/gjP1IXBwl2JpMJ3d3dqKqqooUREfCfxV6vF6WlpX79HuWPwfHV4GJiYoQS8eH0vcowjNA8Vw4nRl599VW88sorqKuro2Oi4qPQcyrBGnqGhobQ1dWFqqoq6PV6nH/++aI/ppgFC6g/zNzxxxv4Qggz3Xng8T2bli5dSrsPIuGPZVLDRvHYbDa0traioqICcXFxX/n96e8Jn88nTPbi4+PpNT9LQ0NDGBgYoMauIuGPvrIsi1WrVon+unQ6ncLRUK/XKzTOFvO+qNT4wJOZmYnly5dLPRxotVo8//zzqKuro0IggUGh51Q4joPH45F6GLPGrxDZ7Xao1WpER0fjww8/FD30iBV4WJZFS0sLoqOjUVxcTHcf5mn6nQen03lC7xOv1wuj0YiMjAysWLFC6qGGJH73oaSkJFTPSEvOarWio6Nj1vdL+AaQZrMZk5OTwjG45ORk+pw5hb6+PpjNZmqKKRKO49De3g4AkpRW9/l8QuNsu92O+Ph4oRpcqCw2MgwDg8GAjIwMWQSe/fv343//939RX19PR/YDh0LPqQRT6GEYBo2NjViyZMkJH5hihx6xChZ4PB4YjUakp6dT5SU/mt77ZHx8HF6vF9nZ2cjPz6fJngjGx8fR2tqKysrKGXcfyMKNjIygp6dn3setWJYV7saNjY355W5cqOnu7sb4+DgqKysp8IiA7yUVHR2NoqIiyXdZOI6D3W4XSsRPPy4drMfg+MCTnp6OrKwsqYeD119/HU899RTq6+upmW9gUeg5lWAJPS6XS7iQd/JqvVihR8yCBZOTk2hqakJRUVHAGqiGm/HxcbS0tCA3NxeTk5NCIQT+0jdN9hbOZDKhq6sLarWa7pCIpL+/HyMjI8LO9kKdfDcuMjJSmOyFYyVDvoLY1NQUysvLaWFEBBzHoaWlBYsWLZLt8WKXyyW8J9xu9wmnBeQ43pOxLCvcp5TDiYbDhw/jsccew8GDB5GWlib1cMINhZ7TcbvdUg/htMbHx9Hc3IyysrIZj858+OGHWLdund8vQ4oVeCwWC44dO3bKc/lk4YaHh9HT03PCZJyf7JlMJrr07Qd8FTy6+yAOjuPQ3d0Nu92OiooK0XYfXC6XcDTU7XYL94DCofQvf7+EYRi/X6gnX+A4TrhPuXLlyqB4jvnTAhaLBTabTTgGl5KSIsvPOj7wpKWlyeLUyDvvvIMf/OAHqK+vp/ud0qDQczpyDj2Dg4PC5PVUq5Aff/wx1q5d67dJgVj3d4AvzoyPjIxQuWSRcByHnp4ejI6OQqVSnXZlnL/0bTKZwHEc0tLSoFQqqbLMGfAr45OTk6JOxsMZPxn3+XwoLS0N2O4Df+fBZDJhYmJClB5ZcsEft4qKikJxcXFQTMaDDV++PjY2FitXrpR6OPPCH4Pjq8FFRUUJCwNy2BllWRZGoxGpqamyCDwffPABvvvd76Kurk4Wd4rCFIWe0/F4PDjDcxFwHMeho6MDk5OTZ5y8fvLJJ1Cr1X4JEWIWLOAnMdSsURwsy6K9vR0cx2HVqlVzeo75QggmkwkulwupqalQKpVhsdo9FyzLChNFKS4ihwO+uMnixYslPQrE98gym80YHR3FkiVLhJ3RYF+w4Sfjy5YtC5rdh2DDsiyampoQHx+P/Px8qYfjNyfvjE6vBhfo73WWZdHY2Ijk5GRZtAj4+OOPcd999+HAgQOyCGBhjELP6cgt9Ph8PjQ2NmLZsmWzWoH7/PPPUVpauuDLh3zBAgB+/fDyer1oampCUlIS8vLy6AtWBPxrxh/PMX+0gV/tTkpKEla7wzmsMgwDo9FIr2MR8cVa+OdYThwOh1AOG8AJJeKDiZyf41DBT8aTkpKQm5sr9XBEwzCMUA3OZrMhLi5OqAYn9jE4uT3Hn332Gb75zW9i37599L6SHoWe05FT6HE6ndDr9cjJyZl19RGDwYCCgoJ5348R8/6O0+mE0WhEXl4enW0VCd8fJjs72+9N2PiqV/xqN/+llpaWFjIlTmfD4/HAYDAgKyuLjiyIxOv1wmAwIDMzUxaVl07H4/EIAcjlcgmr3UlJSbIOw3x1K6VSKYvL3qGIP26VkpIii92HQOE4TmgUPP0YXFpamt8XBvhdtISEBFkEDIPBgP/4j/9AbW0tCgsLpR4OodBzel6vV9jhkNLY2BhaWlpQXl4+p3ruTU1NyM7ORmJi4pwfU8zAw5fyLSsrm9fYyJlNTEygqakJq1atEr0/zEyFEJRKJRQKBRYvXizqY0tpamoKRqORKg2KyO12w2AwIC8vD0qlUurhzMnJq91y7X3Ch8qsrCxZdKgPRfxusFwu1EuJPwZnsViEhYG0tDQkJSUt6MSA3I4NNjU1YdeuXaipqUFJSYnUwyFfoNBzOnIIPQMDA+jt7UVVVdWcj6m1trYiPT19znXgxQw8Q0ND6Ovrg0qlospgIuGbNVZWVkpyxGZqakpY7eY4LmiP+5yO3W5Hc3MzysvLkZCQIPVwQpLT6YTBYEBxcXHQ97I4uffJokWLhJ1RKT8HPR4P9Hp9UIbKYEG7aKfGLwxYLBaMj48jLi5O2AWayzE4juPQ1NQkm8IQbW1tuPXWW/Haa6+hvLxc6uGQL1HoOR0pQw9fpWhqagqVlZXzWhk8evQokpOToVAo5vS4YhQs4DgOx48fFypbyWmlM5QMDAxgcHDQbwUsFurk4z6hUAiBD5UqlUoWVYpCEb9TGaqhkl8YsFgsYBhGqHoVFxcXsPeF2+2GXq9HQUEB9QsRCcMw0Ov1yMzMpOOvZzBTnyz+fXG6BbPppb8LCgoCOOKZdXR04KabbsJLL70EtVot9XDIiSj0nI7P5wPDMJI8rtFoRHx8/IKqFB0/fhyxsbHIyMiY1Z/nOA4+nw+AfwsWMAyD5uZmLFmyRBYdp0MRHyodDodsyyUzDCNU9+ELISiVSiQnJwdNIYShoSH09/fLJlSGovHxcbS1tUm2UxloXq9XeF84HA5hoUrM9wW/i1ZSUiL68ddw5fP5oNfr6djgPLndbuF94XK5hPfF9GNwfHPXJUuWyCLwdHd34/rrr8eLL76INWvWSD0c8lUUek5HitAzNTUFg8GA3NzcBa8MdXd3IyYm5oyXf8U8zuZ2u2E0GrF8+XLZX0IOVnyZ2cWLFwdNqOQLIZhMJoyNjSEuLg5KpVJ29x14fJ+jsbExqFQqWYbKUGA2m9HZ2XlC89xwwrIsxsbGYDabhfeFv6teORwOGI1GulMpIp/Ph4aGBmRnZ8960ZGcGsMwwvtifHwcsbGxSEtLg9VqFXZ4pP7e6+vrw7XXXos//OEPOPfccyUdCzklCj2nE+jQM9+CBafS19cHjuNOWylGzMDDH1EpKSkJ+jP5cuX1emE0GqFUKoP2gixf3cdkMp1w30EuhRA4jkN7e7vQnT5YdqWCDe2inWimqlf8+2K+bQj4z+TKysp5V/Ukp+f1eoVKq1SZ1P/490VrayvcbjeWLl0q3I+LjY2VJPwMDg5i586d+M1vfoMLL7ww4I9PZo1Cz+kEMvT09/ejv78fVVVVflvhHBwchNvtPmUlEzEDj9lsxvHjx8PmiIoU+CMqBQUFc7q3JXczFUJQKpWS3J/hj2YuW7ZMFquJoaq3txcWi+WMDZfDmcvlEt4XXq8XqampUCgUs74fZ7PZ0NraSp/JIvJ6vWhoaKDCECLiOA5tbW2IiopCUVGR0EDbYrFgampKqAYXqGPTw8PD2LlzJ55++mlccskloj8eWRAKPafDMIxwx0Us/Cqyy+VCZWWlX4/NjIyMYGJiYsb68GIWLJg+gRG7EVm4stlswq5gKF705vGFEEwmE9xut3CxNRCFEPhSvhkZGVR1SSQcx6Gzs1O4i0a7aLPj8/lgtVpPuB93ukbBo6OjOHr0KNRq9YKbVZOZ8ZXw8vPzQ2oRSk74+VJERMSMDdpZlhWqwY2NjQnH4NLS0kTZPTaZTNixYweeeOIJbNiwwe8/n/gdhZ7TETv08EeTEhMTRVlFtlgssFqtX6kRL1bBApZl0dbWBgBYtWoVTWBEYjKZ0NXVBZVKFVYTmJMnemJe+Ha5XDAYDMjPz6cVW5HwK7bAF58XtIs2Pyc3Cl62bJlw3GfRokWwWCw4fvw4qqqqZHFcNBR5PB40NDRQJTwR8RVtAcwYeGb685OTk8IuEIATqsEt9PPGarVi+/bteOSRR7Bx48YF/SwSMBR6TkfM0DM1NQW9Xo+VK1eKdtFxbGwMQ0NDKCsrAyDucTY+wKWmpiI3N5cmMCLp6emhXTR89cK3Pxs/Tk5OorGxEaWlpX65W0e+im8kyPfVoM8L/+A4Dg6HQyj76/V6wTAMVCoVFS0QCV/6u7CwkJoUi4QPPBzHoaSkZF6fF/wxOLPZjKmpqQUtmo2NjWHHjh3YvXs3NBrNnMdCJEOh53RYloXX6/X7z7VarWhra0NFRYWoX0R2ux09PT2orKwUNfDwnelXrlxJq+Ii4bf1fT4fysrKaBdtmumNHy0WCxYvXgylUgmFQjHnIw1jY2Nob2+new8i4ps1pqWlnbbIClmYwcFB9Pf3IyMjA1arFW63W7gHlJiYSEHTD1wuF/R6fUg00JUrjuPQ0dEBhmH8tiN88qLZybujp2Oz2bBz5058+9vfxs6dOxc8FhJQFHpOR4zQ09fXh4GBAb8WLDgVh8OBjo4OqNVq0QLP6Ogo2tvbUVFRgfj4eL/9XPIlhmHQ2NiI+Ph4WhWfBX6l22w2A4AQgM5UCGFkZAQ9PT1QqVRhWS45EPjKVitWrKDeJSLq6+uD2WyGWq0W7okyDCMcD7Xb7UhISBB2R6kE+9zxgYd6HYmH4zgcO3YMXq8XpaWlonz3nbw7CkAohJCQkHDCAuPExASuueYa3H333bjhhhv8PhYiOgo9p+PP0MOyLNrb2+HxeALWPNLlcqG5uRlqtdrvBQsAYGBgAIODg1CpVHRWXCRutxsGgwErVqygjt7z4Ha7hQDk8XiQmpoKpVKJ+Pj4E94Lvb29MJvNYX9sUEz8PamVK1fSRW8RdXV1wW63o7Ky8pQ7whzHwWazCeWwFy9eLKsy8XLHV85ctWoVHYEVCd9w2+12o6ysLGCLffwxuCNHjuBnP/sZzj77bGzevBkXX3wxbr31Vtx66624+eabAzIW4ncUek6H4zh4PJ4F/xy+ClRycnJAV+q9Xi8+/PBDYZLhr8kcv/ridDpRXl5Oq4QimZycRFNTEx2d8BO+EILJZMLk5KRwpttqtcLlclH1MBE5HA40NjbSJFFE/CTR5XLN+Qjs9JVujuP8euE71PDHuUtLS+melIimv5aleg26XC688cYb2L9/P959911kZGTgW9/6FjZt2kRH+YMThZ7T8UfocTgcQi+VQDYqYxgGLMvC4XDAZDLBYrEgJiYGSqUSSqVy3uUbGYYRLiBT3xLx8CVmKyoqqImgCFiWhdVqxdGjR+H1epGWlgalUklHfURgt9vR3NxMR2BFxN/54zhuwfceTr7wnZKSAoVCgaSkpLBfFHA4HDAajSHfKkBqnZ2dmJqaQnl5ueRzDLfbjRtvvBGbN2/GpZdeirq6Ohw8eBA+nw9XX301tmzZgsrKSsnHSWaFQs/pLDT08AULKisrA/YBebqCBVNTUzCZTDCbzYiIiBAC0GzvL7hcLhiNRmRnZ9N5fBHxF5DVajUdNRGJz+cTqg3m5OTAbrfDZDLBarViyZIlwlEfMXo7hBM+vKtUKkmay4YDjuPQ0tKCmJgYFBUV+XXyxTCMcOF7fHzcr1USgw0feCi8i6urqwuTk5OoqKiQPEh4PB7cdNNNuPzyy3HvvfeeMB6r1YpDhw7hwIEDaG1txbe//W3ceuut0g2WzAaFnjNxu93z+nu9vb0YGhoKaG8EPvAwDIPIyMjTfmDw3b1NJhMYhhG63p+qYhW/WktlfMXDN2qcmJjwe6Na8iX+nlROTs6M5eKnF0KIiIgQ3hvh1BPJH0wmE7q7uym8i4gv/R0XF4f8/HxRJ4nTqyRarVbExMQIiwOhXviDL2NfWVlJO+8i6urqwsTEhCyOGnu9Xtx2220477zz8MADD5z2veX1ejE+Pk53FeWPQs+ZzDX08A06fT5fQO+7cBwHhmHmVbCAP84wMjIidL1PT09HXFwcIiIiMDIygu7ublRWVtJqrUhYlkVrayuioqLm3YeAnBl/t2S296T4Qggmk+mEY3D8e4PMbGBgAENDQ1Cr1VQYQiQMw8BoNCIlJQW5ubkBf3yn0yksDjAMIxQJCbX3xsTEBJqamijwiKy7uxt2u10Wgcfn8+GOO+5ARUUFvve974XU6znMUeg5E4/HgzM8Hyf8WYPBgNTUVNFX3aabHngW+mHh8/lgsVhgMpngcKMuU0cAAGAHSURBVDgQHR0NlmWxevVqOuojEr6xK9+3hD5gxWGz2dDS0jLv4yknvzforsPMuru7MT4+TruVIvL5fDAYDEhPT8eKFSukHg68Xq9QJMThcCyo8aOc8IFHpVJR3y4R9fT0CJ8ZUr9eGIbB3Xffjfz8fPz4xz+m7+PQQqHnTGYbeiYnJ2E0GlFYWBjQqh58wQJ/l6NmWRbNzc3w+XyIiYnBxMQEkpKSoFQqg/6LTE6cTieMRiPy8vICWugi3JjNZhw/fhxqtdovx9RYlsXo6OgJdx3CvRACX9WRLzFLnxHikHuvo5MbP8bGxgqNH4Np189ut6OlpYXuo4mst7cXY2Njsgg8LMvinnvuQVpaGp544gnJx0P8jkLPmcwm9FgsFrS3t0OlUgXsguPpChYslMfjgdFohFKpFDqmsyyL8fFxmEwmjI2N0STPD/h7UmVlZVT6VER8Pym1Wi3KbiV/12F6IQSlUjmr7t6hgj+eGR0djeLiYlodFYnH44Fer0deXl5QlMzlOA6Tk5NCOeyoqCjhHpCc78jZbDa0trb6bZGEzKyvrw9WqxUqlUrygMGyLL7zne9g6dKlePrppyUfDxEFhZ4z8Xq9YFl2xt/jOA49PT0wmUyoqqoK2ARHzMDD94YpLCxEWlraKR9/emO7ZcuWCV9k4VbRZ774nQdaRRQPx3EnNGoMVDifXiaer5Io90neQvBl7BMSEpCXl0eBRyR8c9fCwkKkpqZKPZx5cblcwhFRr9eL1NRUKBQKJCQkyOZ1Mz4+jra2Ngo8Iuvr64PFYoFarZY8YLAsi927d4NhGPzmN7+RfDxENBR6zuRUoYdlWbS0tIDjOJSXlwfsTbKQggVnYrVa0dHRMafeMPxKHj/JW7RokTDJC5dV7rnq6+vDyMgIXfIWEcdxaGtrE/qWSPUlxldJNJvNIVkIQW53S0KV0+mEwWAIqeaufLNgs9mMiYkJJCYmQqFQICUlRbLTA2NjY2hvb0dVVVXIV6STUn9/P8xmM1QqleQnRViWxY9+9COMj4/jD3/4AwWe0Eah50xmCj38EQOFQhHQlU0xA09fXx+Gh4cXfARoob2AQhnHcejo6IDb7Q5oUA43/M5DfHx8QAuKnMnJl71TUlKgVCqRlJQkmzHOBf85mJubS/fRRBQODTFZlhVOD4yOjmLp0qXCPaBALZ7xPaVWr15NJdZFNDAwICz6SR14OI7DY489hv7+frzwwguSj4eIjkLPmfh8PjAMI/x/vmBBUVFRQGuyi1WwgOM4HD16FB6PB2VlZX5908+1F1Ao4yfisbGxKCgoCMpJbjDwer0wGAzIzMxEVlaW1MM5Jb4Qgslkgs1mQ0JCgtD0MRi+ePmdh6KioqA9ahUMwrFcMsdxwuKZxWIBAOH4tFjfHVarFceOHQtoX71wNDg4KCyuSv05x3Ecfv7zn6O9vR0vvfQSHc0PDxR6zmR66DGbzejo6IBKpQrYF5CY93d8Ph8aGxuRmJgo+oq4x+MRApDb7RYCUKgc8zkdvpT58uXLZT0RD3Z8JbyVK1cGVZO4k+/I8avcCoVClscf+Xt/paWlVIBDRPzdknC/98d/d5jNZrhcrhNKxfvju8NisaCzszOg93LD0eDgoNCwXQ6B59e//jU+++wzvPbaa7L8nCWioNBzJj6fDz6fD93d3TCbzSFTsICfIObm5s7YlV5MJ/c74ZvaJSYmhlwA4pth0oq4uPgV8WCvhMdxHBwOhzDJk1u1K76qVWVlZVju2AYKf9SK7paciGEYoVT89B3SlJSUea3Um81mdHV1UeAR2dDQEAYHB2UTeJ577jm8++670Gq19N89vFDoORO+fHNERERAe0+IeX9nfHwcra2tspgg8l9iJpMJdrs9pHoBjY2Noa2tbd7NMMns8BPEUJyITy+E4PP5kJaWBoVCIckOKX8ESK1W00RcRPxEXK1W01Gr0zi5VPzixYuFBYLZPG8mkwk9PT2oqqqilX4RDQ8Po7+/H1VVVZIfIeM4Ds8//zwOHTqE2tpaUT/HGIbB2WefjaysLNTV1Yn2OGROKPScSVdXFzweD3Jzc0OiYMHw8DB6e3tRWVkpi5Xj6fimdiaTCePj40hISIBSqZS0ms988c+zSqWiCaKI+Oc5HCaIXq8XFosFZrM54IUQhoeH0dfXJ1qvI/IF/nmmifjcTU1NCQsELMuedoFgZGQEvb299DyLbGRkRHg9Sx14AOAvf/kL9uzZg3379ok+/3n66afx6aefwm63U+iRDwo9Z8IwDHw+X8Aej+M4+Hw+UQoWdHZ2YmJiAhUVFbL4ADod/p4Dv4oXGxsrNHyU89g5jkN3dzfGx8dRWVkp67EGu56eHqGxXbg9zzMd8xFrgaCvr08oLxtuz3MgDQwMCJe86XlemJkWCPh7QCMjIxgYGJDNRDxU8cFy9erVsnieX331Vbz66qs4cOCA6CcC+vv7ccstt+Dhhx/G008/TaFHPij0nAnLsvB6vaI/jpj3dxiGQUtLCxYtWhSU3dKDpRcQy7Joa2sDAEl7w4Q6vvQ3X3Ew3J/n6QsEfLlffoFgIavYfHPXiYkJVFZWhv3zLKbe3l4hwAfbrrbc8ZUS+V0gjuNQWFgIpVJJuzwikdvRQa1Wi+effx719fUBKUK1c+dO7N69GxMTE3jqqaco9MjHjJNf6SN5mBEz8LjdbhiNRmRmZgZt88CIiAjEx8cjPj4eBQUFQsd7vV6PqKgoIQBJeYzM5/PBaDQiJSUloEchww3LsmhubsaSJUtQXl5OzzO+eH8kJSUhKSlJKIRgMpnQ0NAgFEKYa68svpQ9wzBQqVT0PIuoq6sLdrtdFp3pQ1FkZCTS0tLgdrvhcDhQWFgIq9V6wvtDLoVCQoHZbEZ3dzdWr14ti8Czb98+/OEPfwhY4Kmrq4NSqcRZZ52Fd955R/THIwtHOz3TiL3TI2bgmZiYQHNzM4qLi5GSkuK3nysnLpcLJpMJJpMJHMcJE7xAlnh1uVwwGo3IyckJeCW8cOLz+WAwGKBQKJCTkyP1cILCyb2y0tLShF5Zp/qsYVkWLS0tWLJkCfWUEhHHcTh27BjcbjftWIqsv78fJpPpK/1hnE6ncAzO6/UK94Di4+PpdT8PfBEOuQSegwcP4he/+AXq6+sDNgfavXu30PfH5XLBbrdj+/btePnllwPy+OS06HjbmYgZesQsWGA2m3H8+PGQrGh1KtN7AXk8HmGCJ2alK75UcmlpKZKSkkR5DPLFjqXBYEBubi7S09OlHk5Q4u85mEwmOJ1OoRDC9FLxDMOcsGNJxMFxHNrb2wEAJSUlNMEWUV9fHywWyxmPDnq9XlitVpjNZkxOTiIpKUkoh02B9Mz4fkdyCTyHDx/GY489hoMHDyItLU2SMbzzzjt0vE1e6HjbmYj1ZcQXLADg1w9UjuOEi8dr1qyR1Z0XsS1atAhZWVnIysoSJnhdXV2YmpoSegElJCT47b+pxWLBsWPHoFKpwiZYSoHvdVRSUoLk5GSphxO0YmJikJmZiczMTKEQwsDAAFpbW5GYmIiUlBT09fUhKysLy5cvl3q4IYtlWbS2tmLRokUoLCykwCOinp4ejI2NzeroYExMDDIyMpCRkQGWZTE+Pi40JF+2bJmsGwZLzWq1Cg1e5fD8vPPOO3j00UclDTwkeNBOzzQcx8Hj8fj154l1nI1lWbS3t4NlWZSWltLq1P/DMAysVitMJhMmJiaQnJwslPqd73PU39+PoaEhKuErMr6nVGVlZUDOY4cjjuNgNpvR2tqKyMhIJCYmQqFQLLgQAvkqlmXR1NSE+Ph45OfnSz2ckNbd3Q2bzbbgIhzTGwZbLBZEREQIASiQx6jliu/ftXr1all8F77//vvYvXs36uvrkZmZKfVwiLzQ8bYz8WfoETPweL1eNDY20kX6M1hoLyD+HL7T6UR5eTlVWhKRyWQSmjRSryPxTE1NwWg0oqSkBElJSUIhBIvFIptCIaGAPzqYmppKd9JE1tnZicnJSVRUVPh98c/tdguV4NxuN1JTU6FQKE44JhouRkdH0dHRIZvA8/HHH+O+++5DXV1d0BZuIqKi0DMbbrd7wT9DzMAzNTWFxsZG5OfnQ6lU+u3nhrq59gJiGEaoHFZUVBR2X3CB1N/fL/Qsod0G8fB30ioqKhAfH/+V33c6ncIEj2EYYYX7dIUQyFfxRTgyMjKQlZUl9XBCFt+Pzul0BqQ4BH+KwGw2w263IyEhAQqFAqmpqSG/IMYHnqqqKlk0hv7000/xrW99C/v376f7iORUKPTMxkJDj5gFC8bGxtDW1oby8nIkJCT47eeGG74X0MjICCwWCxYvXoz09HSkpaVh0aJF8Hg8MBqNSE9PR3Z2ttTDDVn8pIVfpQ31iYOUxsbG0N7eDpVKNatjOl6vVwhATqczrFe458Lr9UKv1yM7O5uqO4qI4zgcP35cqIYX6Nckv4hmNpthtVqxZMkS4ZioHEKBP/GfHatXr5bFv02v1+Ouu+5CbW0tCgsLpR4OkS8KPbPh8XhwhufklMQqWAAAg4OD6O/vh0qloqMnfsYf8TGbzQC+KP1bWFhIF7xFxF/wjoqKoopWIjObzejs7Jz30UG+EILJZILdbkdiYqJwTJTuEn6JrzqYn58PhUIh9XBCFn/s2Ov1orS0VBafHdPvAXEcJ5TDDvZd0vHxcbS1taGqqkoW846mpibs2rULNTU1KCkpkXo4RN4o9MzGfEMPwzCiHGfjP+CnpqZoNVxk4+PjaG5uhkKhgN1ul6wXUKjj7zskJSUhLy8vqCcFcjc4OIiBgQG/VVpiWVY4Jjo6OiocE01NTQ3ro4kulwt6vR5FRUVITU2Vejghi2+ky7IsVq1aJcvPDo/HI/QDcjqdQjGdxMTEoFokkFvgaW1txW233YbXXnsN5eXlUg+HyB+FntmYa+gR8/4OwzBoamrCsmXLqNypyEZGRtDd3X3CarjH4xGaofLN7NLT04N+9U5KHo8HBoOBSiUHQE9PD0ZHR8/Ys2S++GOi/D256Oho4R6QHCZJgcIXh1i1ahX17xIR3+8oIiICxcXFQfEZzO+Sms1m2Gw2xMfHC/eAZrpLKhc2mw2tra2yCTxHjx7FzTffjJdffhkqlUrq4ZDgQKFnNrxeL1iWndWfFTPwuFwuGI1GrFixgiaHIuI4Dr29vbBaraisrDzlavXJzR7F6AUU6pxOJwwGAwoLC6mfgoj4+w581cFArS5PL4TAsqzQMDiU+1pNTk6isbGR7lmKjOM4tLa2Ijo6OmgLy3AcB7vdLtwDiomJkeUiAR941Go1li5dKvVw0NXVhRtuuAEvvvgi1qxZI/VwSPCg0DMbsw09YhYssNvtaG5uxqpVq6hBo4hYlsXRo0fBMMyceh2J0Qso1PGvaZocioufHEZGRkp6V4o/4mMymeByuUKyEAL/mqa+UuLiOA4tLS0h1+D15GqJ/D2guLg4yf6N/Gu6qqpKFoGnt7cX1113Hf74xz/inHPOkXo4JLhQ6JmN2YQeMQsW8P1KKisr6R6JiHw+HxobG5GYmIj8/Px5f8nwvYBGRkZgs9mEXkCpqakUgP4fq9WKjo6OWVcOI/PDN8OMi4tb0Gva304u9ZuUlASFQhHUhRD4+w70mhYXx3Fobm7G0qVLsXLlStm8pv2NP0lgNpvhcDiQnJwMhUKB5OTkgL1H+JL2arVaFq/pgYEBXHvttfjNb36DCy64QOrhkOBDoWc2fD4fGIY55e+LWbCAP4N/umNWZOH4KkvZ2dl+7eLMcRzGx8eFS95xcXFCAJLz+W0xDQ0Nob+/H2q1WhYN7UKVz+eD0WiEQqGQdZl1lmUxPj4Os9l8QiGEU/XLkiM+xMvlvkOoYlkWzc3NiI2NxcqVK6UeTsDwC2lmsxljY2OIjY0VymGLNS+QW+AZHh7Gzp078fTTT+OSSy6RejgkOFHomY1ThR4x7+/w5Xv5IynBuvoZDCYnJ9HU1ITi4mKkpKSI9jgcx2FiYkLodr9kyRKh2304BFo+xI+NjYl2kZ58gS8OEWy9YaYXQrBYLIiJiRHeI3LoBzITk8mE7u5uVFVVUYgXEb9rGR8fj/z8fKmHIxn+PcKXw46KihLuAfnr+Bl/L02lUsni/p3JZMKOHTvwxBNPYMOGDVIPhwQvCj2zMVPo4QMPwzCIjIz0a+DxeDxobGwUVmhDdfteDvgV2oqKioCfwZ/eCygqKgpKpRJKpVK2k7uF4KsszfWuFJk7l8sFg8GAgoKCoC8O4XQ6hfcIXy6e73UiB/yupb/Kf5OZsSyLxsZGJCUlITc3V+rhyIrL5RLuAXm9XuGu3HwL6sgt8FitVmzfvh2PPPIINm7cKPVwSHCj0DMbDMMI93UAcQsWOBwONDY2oqCggJrZiYzvV6JSqSQPGvwFVpPJBI7jhAAkh4ujC8UwDJqbm7Fs2TIUFBRQiBcR//kRiqWSPR6PMLnjCyFIWS2xv78fIyMjUKvVQXMMLxixLAuj0YiUlBTk5ORIPRxZ8/l8wl25iYkJJCYmCnflZrOzLrfAMzY2hu3bt+Phhx/G1q1bpR4OCX4UemZjeugRM/Dwuw7l5eWIj4/3288lJ+I4Dp2dnZicnJRlc1e32y0EIK/XKzRDDcZeQF6vF0ajEenp6VixYoXUwwlpfFlZKXYtA+3kaolJSUlQKpUBu+Qtdr8j8gW+aXFaWpqs76XJ0fSmwWNjY1i6dKlwD2imY5gOhwNGo1E2lQdtNht27NiB73znO9ixY4fUwyGhgULPbPChR6yCBcAXq4ZDQ0Oy2HUIZSzLoqWlBTExMUHRzC6YewHxx6zy8/OhVCqlHk5IGx0dxdGjR2XTRyOQ+EII/OROzGIhHMehq6tLWDChY5riYRgGBoMBSqWSFkwWiOM4OBwO4R4QgBOOisot8ExMTOCaa67B3XffjRtuuEHq4ZDQQaFnNhiGgdvtFq1C29GjR+HxeFBWVkarhiLidx0UCkVQHpM4VS+g5ORk2QUgvjhEKB6zkpuRkRH09PTQRXp8WSyEn9wtWrRImNwtdDGJ4zh0dHTA6/WirKxMdu+5UMIwDPR6PTIzM6kRtwjcbrdQDntqagoejwfFxcXIzMyU/HXtcDhw7bXX4rbbbsPNN98s6VhIyKHQMxvPPPMM3G43qqur/VpYgO8Lk5CQENL9BuTA6XTCaDSGzK4Dy7IYHR2FyWSCzWZDYmIilEqlLPqcjI2Nob29PSyOWUmN7pWc3tTUlHAPaCGFEDiOQ1tbGyIiIiRt8BoOfD4f9Ho9srKy/No+gHzV1NQU9Ho9VqxYgYmJCdjtdiQkJEChUCA1NTXgi7BOpxPXXXcdrr/+euzatSugj03CAoWe2RgcHERtbS1qa2sxNTWFzZs3Q6PRLCioOJ1ONDY2IicnJ6hKygYjm82GlpYWlJWVITExUerh+N2pegGlpaUF/EuL33VQqVTUr0REHMehu7sbNpsNlZWVtEM8C3whBJPJBLfbLXS7P9NRUf5I7JIlS6gQh8h8Ph8aGhqCrtR6MHI6nTAYDCgrK0NCQgKALz5XbDYbzGYzrFYrFi9e7Led0jNxuVy48cYbsXXrVtx99930PiNioNAzVyaTCXv27IFOp8PY2Bg2btwIjUYzp9U/fhJeWlpKR39EZjKZ0NXVBZVKFRZ3HaTsBdTX1weTyQSVSkXle0U0/ZgVlf+en5OrXJ2q2z1fKjkhISGse8MEgtfrhV6vR05ODtLT06UeTkjjA09paelpFwKn75SyLHvCTqm/23TcdNNNWL9+Pe655x4KPEQsFHoWwmq1Yt++faitrcXQ0BCuvPJKVFdXo6ys7JQTkXfffRdLliwJm0m4lHp7e2E2m8N6Ej690WN0dLQojR45jsOxY8fgdDrpcrfI+F2HRYsWoaioiCYHfnByt3t+pzQpKQnNzc1CvzQiHq/Xi4aGBuTl5YXE8WM5c7lc0Ov1Zww8J/N4PCfcA0pJSYFCoUBSUtKCPvO9Xi9uu+02rFu3Dvfffz99phExUejxl/HxcRw4cAC1tbXo6urChg0bUF1dDbVajcjISLAsi+9+97toaGjAgQMH6OiPiPjiEPyFY5qEf4Fv9GgymQDAL72Agq0aXjBjGEZo0JiXlyf1cEISv1M6PDyM/v5+LF26FDk5OVAoFGFfJEIsHo8Her0e+fn51JtOZHzgWWiBGf5Oqdlsxvj4OOLi4oRy2HO5W+jz+XDHHXegsrISDz/8MH1/ELFR6BHDxMQEDh48CK1Wi/b2dnzta19Dc3Mzli9fjmeffZYuHIuIYRg0NTUhLi6OikOcxvReQD6fD2lpaVAqlXMqPODz+WA0GpGamkpd0kXGVx7MyMhAVlaW1MMJadOPWcXHx8NkMsFsNgOA0DNr2bJlEo8yNHg8HjQ0NKCgoABpaWlSDyek+SvwnOzkiokxMTFCADrdghrDMLj77ruRn5+PH//4x/RdTQKBQo/Yenp6sGnTJqSmpsJqteLiiy9GdXU1zjvvPLp87GcejwcGgwFZWVlU5nQOvF6vEIBcLpcQgOLj40/5ReR2u2EwGJCdnU0VlkTGP9d09Ed8brcber0eK1eu/MquA79QYDab4fF4hJ5Zp3ufkFPjn+vCwkKkpqZKPZyQ5na70dDQgJKSEiQnJ4v6WE6nU3if8AtqNpsNVVVVwqkLlmVxzz33QKFQ4Kc//SmdxiCBQqFHTE1NTbjpppvw85//HOvXr4fb7cbhw4dRU1ODzz77DOeffz62bduGCy64gHZ/FsjhcKCxsRFFRUX0BboADMMIzVAnJyeRkpIi3G/gJ3ZTU1MwGo0oLi5GSkqKxCMObfyFY3quxTeX55ovhMC/T/ieWQu93xAu+F0Hel2LL5CB52Rerxcmkwn33HMP2tvbcfbZZ2Pz5s14//33ERsbi6effpreLySQKPSI5dChQ3jooYfw6quvYtWqVV/5fY/Hg7fffhtarRYfffQRzjnnHGg0Glx88cV0dnyOxsbG0NbWJptu0qFipl5AsbGxGBgYQGVlJeLj46UeYkibnJxEY2MjysvLhZKyRBx8kJ/r5W7gy0IIJpMJ4+PjiI+PF4730G7+V/GBR4pJeLjhjw8WFRVJHi7dbjcOHTqE3//+92hpacHFF18MjUaDjRs3Sj42EjYo9IjhN7/5DXQ6HWpqamZ1Ttnn8+G9996DVqvFe++9h9WrV0Oj0eCyyy6jggdnMDQ0hL6+PuoLIzKO49DT04Pu7m7ExMQIzVClaGAXDsbHx4UgP9dGmmRu+HBZUVGx4CDPcRzsdrtwv2Hx4sVCxURazPpyN83f90rIV/GBRy7HB1mWxQ9/+EPYbDYh+Ozfvx8HDx7E4sWLsWXLFqH/ISEiodDjbxzH4c9//jO+/vWvz6ssMMMw+Oc//wmdToe33noLZWVlqK6uxoYNG+jy7DQcx6Grq0tozkjHA8U1MDCAwcFBqNVqxMTEwG63w2QywWq1YunSpUIz1HAtDe5PFosFx48fh1qtpiAvMr5nmkqlEiVcOhwO4X5DRESE0OckHD/LF7KbRuZGboGH4zj85Cc/wcDAAF544YWvLJQNDg6irq4O+/fvh8lkwrvvvkstPYgYKPTIGcuy+OSTT1BTU4M333wTRUVFqK6uxhVXXBHWR4tYlkVraysiIyNRUlJCZ4JFxIdLu92OysrKr3xZcRwHh8PxlV5ASqWSVrbnYWhoCP39/VCr1fT8iWxsbAzt7e1Qq9UBmWBNr5jo9XqRlpYGhUIRFoUQHA4HjEYjHdUMAL7n0cqVK2VREY/jOPz85z/H0aNH8de//vWMC5ROp5MCDxELhZ5gwbIs9Ho9ampq8PrrryMnJ0c4DxtOq2Z8meSUlBTk5uaG/GRBShzHoa2tDRzHYdWqVbMKl9N7AfEr2wvtBRQuent7YbFYoFKpaOdSZFarFceOHZNsN83n8wmNHkO9EAIfePxxfJCcHh945NLziOM4/OpXv0JDQwNeffVVOglApEahJxhxHIempibU1NTg4MGDUCgU0Gg02Lx5c0hfCHS5XELp3vT0dKmHE9L4fkfx8fHIz8+fV7h0u91CAGIYRghAdEflRBzHobOzEw6HAxUVFSE36ZUbk8mE7u5uVFVVyWI3baZCCKFyX46/L0VFZsQnx8Dz7LPP4v3330dNTY0s3msk7FHoCXb8arxWq0VdXR0SEhKwdetWbNmyBQqFImR2QiYmJtDU1ITS0lK6ACsyr9cLg8GAzMxMvzXCnE8voHDAcRza29uF3bRwfi4CYWhoCAMDA8LdNLnhCyHw9+WWLFki3JcLtkkj/5lNgUd8fEPd3NxcWfTy4jgOzz//PA4dOoQ9e/bM634zISKg0BNKOI7D8ePHodPpsG/fvhMqomRkZATthMpiseDYsWNQqVRheQE4kJxOJ4xG44zNGf3l5B4nM/UCCgcsy6K5uRnLli3DypUrw+rfLoX+/n6YTCao1eqg2UGZfl8uIiJCqAQn9+OifOARq0AE+ZLP50NDQwNycnJkcwLiL3/5C/bs2YN9+/bJ/rVKwgqFnlDFcRx6e3uh0+mwZ88eAMDmzZtRXV2NFStWBM0Eq7+/H8PDw1CpVEG30hls+IlKWVlZwO6JsSwrBCC73Y7ExESkp6cjOTk5pI95MQwDo9GI1NRU5OTkSD2ckNfd3Y3x8fEZi3EEi5kKISiVSsTFxcnq89xutwsV8WiRSlw+nw96vR7Z2dmyCTyvvPIKXnvtNdTV1dF/fyI3FHrCAcdxGBoagk6nQ21tLVwuFzZv3gyNRjPv+xpi4zgOx44dg9PpRHl5edBOVILF6Ogojh49KmlfGJZlMT4+DpPJhLGxsZC62zAdfxRlxYoVyMzMlHo4IY2/LzU1NYXy8vKQCdJer1dYLHA4HEhJSYFCoZC8EILNZkNra2vAKuKFMz7wrFixAhkZGVIPBwBQU1ODP//5z6ivr6cjjUSOKPSEG47jYDKZsGfPHtTW1mJ8fBwbN26ERqNBcXGxLAIQwzBobm7G0qVLUVhYKIsxhbLh4WH09vZCrVbL5uz1yXcbQqUXEF+MQ8zjg+QLHMfh6NGjYBgGpaWlIfs5wrIsRkdHYTKZYLPZkJCQAIVCEfDFAr6hLgUe8TEMA71ej+XLl8tm4WTv3r343e9+h/r6+rCqKEuCCoWecGe1WrF3717U1tZieHgYV111Faqrq1FaWirJiqHH44HRaERGRgZWrFgR8McPNz09PbBarbIuk8z3AhoZGYHFYkFMTExQ9gJyOBxobGykbvQBwHEcWltbERUVJZvFnEDgOA42mw1ms1lYLFAoFKIXQuB7HlVVVVFDXZHJMfDU19fj6aefRn19fUhXkCVBj0IP+dL4+Dj279+P2tpa9PT0YP369di2bRtUKlVAAhDfsbuwsFAWTdVCGcdx6OjogMfjQVlZWVAd+5mamoLJZBK63PMBSM6TLbvdjubmZupVEgBUIOJL0wshREZGQqFQ+L0QwujoKDo6OlBVVSWbneJQxTAMDAYDMjIysHz5cqmHAwB488038fjjj+PgwYP0vU3kjkIPmdnExATq6+uh0+lw9OhRXHbZZaiursZZZ50lygR5fHwcra2tNCkMAH5SuGTJkqA/PuhyuYTL3XLtBcTfl6KL3eJjWRZGoxFJSUnIy8uTejiywr9XzGYzfD4f0tLSoFAoFlQIgW/ySoFHfHzgSU9P91srgYV6++238cMf/hAHDx6URalsQs6AQg85s6mpKbz++uvQ6XRobGzExRdfjOrqapx77rl+OTM+MjKCnp4eqFQqWa/WhwKfzweDwQCFQhFyVcM8Hg8sFssJvYDS09MlrW7FN8KU032pUMVPChUKBbKzs6Uejqx5vV5YLBaYzWahEMJcy8ZbLBZ0dnbKpslrKOOrPSoUCtkc+37//fexe/du1NfXi3bMrq+vDzfffDNGRkYQERGBO++8E/fee68oj0XCAoUef/vf//1f/Pa3v0VUVBQ2bdqEn/3sZ1IPya9cLhcOHz6MmpoafP7557jggguwbds2nH/++XO+E8JxHHp6ejA6OirrOyWhwu12w2AwIDc3VzblTcXC9wIaGRmBw+FAamoqlEolEhMTAxaABgYGMDQ0JNtGmKGEb6i7fPly2Rz7CRYzFUJQKpVISUk55aKW2WxGV1cXBZ4AYFlWCPNyCTwfffQR7r//fhw4cEDUMQ0NDWFoaAhr1qzBxMQEzjrrLOzduxdlZWWiPSYJaRR6/Ontt9/GY489hvr6eixevBgmkymkt3w9Hg/eeustaLVafPzxxzj33HOh0Wjwta997YxfhCzLntCJPpjulAQj/hJ9cXFx2F00ZRhGmNTZ7XYkJSVBqVSK2guou7sbY2NjUKlUIVVuW448Ho/QjT7Uw7zY+EIIJpMJo6OjQiEEhUIhBHeTyYSenh5UVVVRmBcZf1wzNTVVNruXn376Kb71rW9h//79yM3NDehjazQafPOb38SGDRsC+rgkZFDo8adrr70Wd955J9avXy/1UALO5/Ph3XffhVarxfvvv481a9ZAo9Hgsssu+8qxnvHxcdx11134wQ9+ENKlZOWCvy9VWVkZ9r0TxO4FxPeXcrvdQVcgIhi53W7o9XoUFBTQJWo/46sm8veAoqKisGTJEkxOTmLNmjUUeETGB56UlBTZHEXW6/W46667sGfPHhQUFAT0sbu7u/G1r30NTU1NSEhICOhjk5BBocefqqqqoNFocOjQISxZsgRPPfUU1q5dK/WwAo5hGPzzn/+EVqvF22+/jfLyclRXV2P9+vUYGRnBtddeizvuuAN33nmn1EMNeSaTCV1dXVCr1XRf6iQn9wJatmyZsKo9n6OWLMuitbUV0dHRYVUmWSpOpxMGgwElJSVITk6Wejghr7e3F729vViyZAlYlkVaWppQNIRe6/7FsiwaGxuRlJQU8N2UU2lqasKuXbtQU1ODkpKSgD725OQkLr74Yjz88MPYvn17QB+bhBQKPXO1fv16DA8Pf+XXH3vsMTz88MO49NJL8etf/xqffPIJrrvuOnR2dob1FwLLsvj3v/+NmpoaHDhwAA6HA7t27cJ//dd/hf2ug9j6+/sxPDxMd0pmgeM4TE5OCuV9Fy1aBKVSCYVCMas7CwzDCCuQeXl5Yf2eDwT+uGZpaSk1QgyAwcFB4X5adHS0UAjBZDJhampKkjtzoYpl2RM+S+SgtbUVt912G/72t78F/D6N1+vF5s2bceWVV+K+++4L6GOTkEOhx5+uuuoqPPjgg7j00ksBAAUFBfj444+p8zqAw4cP48EHH8RDDz2ETz/9FIcOHUJubi62bt2KjRs30sTFjziOQ2dnJyYnJ1FRUUF3SuZhLr2A+Ip46enpsrloHMomJibQ1NRE5e0DZGBgAMPDw6iqqprxs4S/M2c2m2Gz2ZCYmAiFQnHaQghkZnIMPEePHsXNN9+Ml19+GSqVKqCPzXEcbrnlFqSkpOCXv/xlQB+bhCQKPf703HPPYXBwED/+8Y9x9OhRXH755ejt7Q37la8///nPeOGFF6DT6YTCDvyHe01NDQ4ePIj09HRoNBps2rQp7C7a+xPLsmhra0NkZCRKSkrC/rXnD6frBUSX6APLZrMJ99Pk1IspVPX398NkMkGtVs8qwMxUCEGpVCItLY12m8+A4zg0NTUhLi4O+fn5Ug8HANDV1YUbbrgBL774ItasWRPwx//ggw9w0UUXobKyUrgf+fjjj2Pjxo0BHwsJCRR6/Mnj8eD222+HXq/HokWL8NRTT+Gyyy6TeliS4TgO3//+99Ha2oqXX375lF3AOY5Da2srtFot6urqkJSUhK1bt2LLli20SzYHfC8HvjEjBR7/83g8QgByuVzweDxYuXIlVqxYQc+3yPgmr2q1+pSfJcR/ent7YbVa512BkC+EwB8ZjYqKEo6M0v3CE3Ech+bmZixbtgwrV66UejgAvvjvf9111+GPf/wjzjnnHKmHQ4g/UOgh4rn33nsRHR2Nn/3sZ7P+0uSrX+l0Ouzfvx9LlizBli1boNFokJ6eThPLU/B4PDAYDMjKyqI+JQEwOTmJxsZGpKenY3JyUrJeQOHCYrHg+PHjqKqqoiavAdDT0yOUXPdXBUKn0ylUguN3TBUKRdgXQuA4Di0tLViyZEnAK6KdysDAAK655hr89re/xQUXXCD1cAjxFwo9RDwL7VPENy/V6XTYu3cvAGDLli2orq5GVlZWWH9RTsdXsSosLKSyvQHAH7GqqKgQinFI0QsoXIyMjAh9YagRpvi6u7ths9lOOFLkb16vVwhATqcTKSkpYblgwAeexYsXo6CgQBb/9uHhYezYsQPPPPMMLrnkEqmHQ4g/UeghwYHjOAwODkKn02HPnj1wu93YvHkzNBpNWB/lstvtaG5uRnl5OfUuCACr1Ypjx45BpVKd8ojVyb2AZtPhnsxscHAQg4ODqKqqmlcZcTI30wugBCqsn7xgkJiYKLxfQnnBgD/WHRMTg8LCQll8h5lMJuzYsQNPPvlkWPYbJCGPQg8JPhzHwWQyoba2FrW1tbDZbNi0aRM0Gg2Kiopk8eURCFarFR0dHVCpVFi2bJnUwwl5IyMj6O3thVqtnvWOw/SL3VarFbGxscLFbprEn15fXx/MZvOsL9GT+eMrPjqdTkmb6nIcJywYjI6OIjY2FgqFIuQKIXAch7a2NkRHR8sm8FgsFuzYsQM//vGPcfXVV0s9HELEQKGHBD+LxYK9e/eitrYWJpMJV199NTQaDUpLS2XxZSKGoaEh9Pf3z2kCTuavr69PqGI137Cy0F5A4SQQR6zIFziOw/Hjx+F2u1FWViabz0z+/WI2m2GxWBAdHS3cAwrmQggcx6G9vR2RkZGyWaQbGxvD9u3b8fDDD2Pr1q1SD4cQsVDoIaFlbGwMBw4cgE6nQ29vLzZs2IBt27aFzOSJv+fEXzKmFXBxcRyHrq4uTExM+P015HA4hHsNkZGRVNkKX07AXS6XpDsO4YLjOHR0dMDn88l+kWimQgh86fhgwQeeiIgIFBcXy+L5ttls2LFjB77zne9gx44dUg+HEDFR6CGhy263o76+HjqdDh0dHbj88stRXV2NNWvWBOVkiv/CZBgGpaWlQflvCCYcx+Ho0aPC8y3mBMXlcgnNUFmWFSZ04XRsMZDPN/ny+WZZFqtWrQqq59vj8cBisQiFEFJTU6FQKGRdCIF/vjmOk00PtYmJCezcuRP/9V//heuvv17q4RAiNgo9JDxMTU3h4MGD0Gq1aG5uxiWXXAKNRoNzzz03KHZLGIYR+jjIpcpPKGNZVqiqFOgz99N7AXk8HqSlpUGpVCIuLi5k/7vzl7qjo6Nlc+QnlMlxx2G+GIaB1WqF2WyWbSEEfkeNYRjZBEyHw4Frr70Wt912G26++Waph0NIIFDoIeHH5XLhzTffRE1NDRoaGnDhhReiuroa559/viwvl3u9XhiNRiiVSmRnZ0s9nJDHN3lNTk5GXl6epGPx+XywWCwwmUyYmpoSegElJCTIYuLkDyzLoqmpCbGxsVi5cmXI/LvkKpQDJl850Ww2C4UQpC4cwvee83q9stnBdDqduO6663DDDTfgG9/4htTDISRQKPSQ8ObxeHDkyBFotVp8/PHHWLduHTQaDb72ta/JolqQy+WCwWBAfn7+gnoekdnxer0wGAxYvny57Jq88ivaJpMJExMTSE5OhlKpRFJSkmxWtOeKYRg0NjYiOTkZubm5Ug8n5PF9YRYtWiSbqmFiOblwSExMjHBvLlANbvk7ah6PRzaBx+Vy4cYbb4RGo8Fdd90lizEREiAUegjheb1evPvuu9Bqtfjggw9w1llnQaPR4NJLL5WkC/zk5CQaGxtRWlqKpKSkgD9+uHG73TAYDMjLy5N9wGRZFmNjYzCZTBgfHw/KXkA+n0/YwVyxYoXUwwl5HMehubkZS5cuDcsdNafTKdyb4zhOODYqZiGE6UU55PB8ezwefP3rX8eGDRtwzz33yGJMhAQQhR5CZsIwDD744ANotVq8/fbbqKysRHV1NdavX3/KppT+NDY2hvb2dlRUVCAuLk70xwt3U1NTMBqNKC4uRkpKitTDmZNg7AXk9Xqh1+uxYsUKZGZmSj2ckMeyLJqbm4UjhOGOL4RgMpngcrlEOTZ6/PhxOJ1OlJeXyyJceL1e3HbbbVi3bh3uv/9+WYyJkACj0EPImbAsi48//hharRb/+Mc/UFxcjG3btmHDhg2iBJKRkRH09PRApVKFdfniQJmYmEBTUxPKy8uRkJAg9XAWhD/SMzIyAovFgsWLFyM9PR1paWmy6QXk8Xig1+uDYkctFPB3puLj45Gfny/1cGTn5GOjSUlJUCqVSE5Onvex0c7OTkxNTckm8Ph8PuzatQtqtRoPPfSQLMZEiAQo9BBx/eIXv8D9998Ps9mMtLQ0qYezYCzL4vPPP0dNTQ3eeOMN5OXlYevWrdi4caNfJsx8E0yVSiWLO0Whjt9Rq6ysDKp+H7PlcDiEIz1RUVGS9wLi76gVFhYiNTVVkjGEE5Zl0djYiKSkJLozNQsnF0KIi4uDQqGY065pV1cXJicnUVFRIYtwwTAM7r77bqxcuRKPPPKILMZEiEQo9BDx9PX1YdeuXWhra8Nnn30WEqFnOn5CUVNTg9dffx0ZGRnYunUrNm/ejOTk5Dn9LL7Cj9PpREVFRdBeTA8mZrMZnZ2dUKvVYbGjxjd3NJlM4Dgu4L2AnE4nDAYDSkpK5vz+IHPHsiyMRiNSUlKQk5Mj9XCCDsdxmJiYgNlshsViwaJFi6BQKE5bCKG7uxt2u102n+EMw+Cee+6BUqnET3/6U1mMiRAJUegh4tm5cye+//3vQ6PR4NNPPw250DMdXwZWq9Wirq4OycnJQgBSKBSn/bt8T5iYmJig75kRLAYHBzEwMICqqqqw3FGbqRdQeno6YmNjRXn9ORwOGI3GkDhCGAz4sutpaWlU5t5PpqamYDabhUIIfADid4h7enowPj6OyspKWYQLlmVx3333ITY2Fr/4xS9kMSZCJEahh4hj3759eOutt/CrX/0KeXl5IR96puN3bbRaLfbv34+lS5di69at2Lp1K9LT00+YVI6Pj+PGG2/Eo48+irPOOkvCUYeP3t5eWK1WqFSqoKl0Jiav1ytc6ua72/vzUjd/Z6qyspKKcgQAwzAwGAxUFU9E0xcN3G43YmJiwHEcVq9eLYvPFJZl8d3vfhccx+F///d/KfAQ8gUKPWT+1q9fj+Hh4a/8+mOPPYbHH38cb775JhITE8Mu9EzHcRy6u7uh0+mwd+9eREZGYsuWLaiurgbDMNi5cyd27dqFO++8U+qhhjy+ZwZfUYkmAl/l715A4+PjaGtrg0qlCtgxunDGMAz0ej0yMzNl12cqVHV3d2NkZATLli3D5OQkkpOToVAoFlQIYSFYlsUPf/hD2O12/P73v6fPOUK+RKGH+F9jYyMuv/xyYZLT39+P5cuX49///jcyMjIkHp10OI7DwMAAdDodXn31VXR3d+Oaa67Bf/7nfyI3N5eOtYmIP34YGRmJkpISeq5n4VS9gFJTU2c1kRodHcXRo0dRVVUVFnempObz+aDX65GVlUVlwAOkr68PFosFarUakZGRQiEEk8mEsbExxMXFCe+ZQJSP5zgOP/nJTzA4OIg///nPsth1IkRGKPQQ8YXzTs9MPvnkE9xxxx34xS9+gfb2dtTW1sJut2PTpk2orq4O+U7pgcaX7OV7lNBzO3d8L6CRkRGhqtXpJnN8kYiqqipJGvuGG5/Ph4aGBmRnZ4f1wlIg9ff3w2w2C4HnZHwhBL5/1mwKISwEx3H42c9+ho6ODvz1r3+VbY8uQiREoYeIj0LPl15//XV8//vfh1arRV5envDrZrMZe/fuRW1tLcxmM66++mpoNBqUlpbSJH0BfD4fjEYjFAoFXej2k+mTOYvFgiVLlgilsGNiYjAyMoLe3t6wLRIRaHyj15ycHKSnp0s9nLAwMDCAkZERqNXqWe+mTE1NCeXjAfi1eiLHcfjlL3+JhoYGvPbaa/S+I2RmFHoICZQXX3wRL7zwAmpra0/bo2RsbAz79++HTqdDf38/NmzYgG3btsmmDGqw8Hg8MBgMtPotsum9gHw+HziOQ1VVVUj2PZIbr9eLhoYGavQaQIODgxgeHp5T4DnZydUT+eIh8fHxc17k4jgOv/vd7/DBBx+gpqZGNk2ICZEhCj2EBMILL7yAffv24dVXX53Typ7dbkddXR10Oh2OHz+Oyy+/HNXV1Vi9ejUFoNPgm2AWFBTQDmOA9Pb2wmQyIS0tDRaLBRzHQalUQqlUYunSpVIPL+R4PB7o9Xrk5+efsSw+8Y/BwUEMDQ2hqqrKb/dlfD6fUDxkroUQOI7D888/jzfeeAO1tbV0lJSQ06PQQ0ggjI2NIT4+fkHnrCcnJ/H6669Dq9WipaUFl156KTQaDc455xy6sDqNw+FAY2MjVq1ahaSkJKmHExa6urpgt9tP6FHidruF1Wyv1ysc5xGrF1A48Xg8aGhooFAfQENDQxgcHPRr4DkZXzzEbDYL3xkKhQJpaWkzPuaLL76Iffv2Ye/evbSwQMiZUeghJBi5XC688cYb0Gq1aGhowEUXXQSNRoPzzz8/rC+w2u12tLS0oKKignrCBADfk8rtdqOsrOyUK9Ni9wIKJ263G3q9HoWFhac9Jkv8Z3h4WGhmHKgFJo7jYLfbYTab8e9//xsvvvgiNm3ahGuuuQbLly/HK6+8gr/97W84cOAAlYMnZHYo9BAS7NxuN44cOQKtVot//etfWLduHaqrq3HRRReF1YVWvkSyWq2mVc8A4DgO7e3t4DgOq1atmnV4OVUvoOTkZApAZ+ByuaDX61FcXIyUlBSphxMWRkZG0NfXh6qqKkkXlBobG1FTU4PDhw/D5/PB5XJh7969qKyslGxMhAQZCj2EhBKv14t33nkHOp0O77//Ps4++2xUV1fjkksuCenz3iMjI+jp6YFarQ7pf6dccByHlpYWLFq0aEEl1lmWxejoKEwmE2w2GxITE6FUKpGSkkJ31k7CB56SkhIkJydLPZywwFciXL16tWx20Pfu3Yvf/e53uPbaa/HGG28I1T6rq6tRVVVFCweEnBqFHkJClc/nwwcffACtVot33nkHKpUK1dXVuPzyy0NqJ2RgYADDw8NQqVRhtbMlFb7vUVxcHPLz8/02yeI4TmjsOL0X0KnuM4QTp9MJg8FA99QCyGQyoaenR1aBp76+Hk8//TQOHjwoBF+73Y5Dhw5h7969aG5uxiWXXCIsdFEAIuQEFHoICQcMw+Djjz+GVqvFkSNHUFJSgurqalxxxRVBXVq4u7sb4+PjqKysDPuJcSAwDAOj0YjU1FTk5OSI9jhn6gUUTqampmA0GlFaWorExESphxMW+MAjp15Tb775Jh5//HEcPHjwlMUrPB4P3nnnHXz00Uf44Q9/GOAREiJ7FHoICTcsy+Kzzz5DTU0N3nzzTeTn52Pr1q24+uqrkZCQIPXwZoXjOHR0dMDr9aK0tJSOQgWAz+eDwWBARkYGsrKyAvrYk5OTMJvNMJvNiI6OFgJQqB9ldDgcMBqNKC8vD5r3ZrAzm83o6urC6tWrZRN43n77bfzwhz/EwYMHqR8TIfNHoYeQcMayLIxGI2pqavD6669j+fLl2Lp1KzZt2iTbewMsy6K1tRUxMTEoKiqiIxwB4PV6odfrZdHo1el0Cs1QQ7kXEB94KioqEB8fL/VwwoLFYkFnZ6esAs/777+P3bt3o76+HpmZmVIPh5BgRqGHEPIF/nK6VqtFXV0dUlJSoNFosHnzZtn0AmEYBo2NjUhKSkJeXp7UwwkLcm6COb0XkM/nQ1paGpRKZdCXK5+cnERjYyMqKyuD/t8SLPjAU1VVhUWLFkk9HADARx99hPvvvx8HDhzAihUrpB4OIcGOQg8h5Kv442NarRYHDhzA0qVLodFosGXLFqSnp0uyu+L1emE0GiU5XhWu+IphRUVFsu8Jc3IvID4AxcfHB9Vu4MTEBJqamijwBJDVasWxY8ewevVq2QSeTz/9FPfccw/27duH3NxcqYdDSCig0EMIOT2O49DV1QWdToe9e/ciOjoaW7ZsQXV1NTIzMwMyoXS73TAYDMjLy6Mz7QHCX6APxophDMMIAWhychIpKSlQKpVISkqSdQDiA49KpQrqAiPBZHR0FB0dHbIKPHq9HnfddRf27t2LlStXSj0cQkIFhR5CyOxxHIf+/n7odDrs2bMHPp8PW7ZswdatW5GbmyvKhJIv10sNGQOHP14VChfog6UXkM1mQ2trK1QqFZYtWyb1cMICH3iqqqpkUxSjqakJu3btglarRXFxsdTDISSUUOghRGwPPPAADhw4gEWLFqGgoAAvvPBC0K2cz4TjOAwPD6O2tha1tbWYnJzEpk2boNFoFtSwcrrJyUk0NTWhrKws6CffwcJut6O5uTkkj1ed3AsoPj4eSqUSqampkpY85wOPWq0OuYIMcjU2Nob29nasXr1aNoGntbUVt912G/72t7+hrKxM6uEQEmoo9BAitjfffBOXXXYZoqOj8eCDDwIAnnzySYlH5X9msxl79uxBbW0trFYrrr76amg0GqxatWpeAWh8fBxtbW2orKykoz4Bwj/n4bDbwHEc7HY7TCYTrFYrli5dKjRDDWTlLv45p8ATOPxzLqfAc/ToUdx88814+eWXoVKppB4OIaGIQg8hgbRnzx5otVq88sorUg9FVKOjo9i/fz90Oh0GBgZwxRVXYNu2bSgvL5/VkSKLxYLjx49DrVZjyZIlARgxsVqtwlGfcHvOOY6Dw+EQmqHyvYCUSqWo9zz43YZwfM6lwgceOT3nXV1duOGGG/Diiy9izZo1Ug+HkFBFoYeQQNqyZQuuu+46fP3rX5d6KAFjs9lQV1cHnU6Hzs5OrF+/HtXV1aiqqpoxAP3pT39CfHw8tm3bJpuLxaGOb8gop3K9UuJ7AZlMJkREREChUPi9F5Ac75OEOv4YoZwCT29vL6677jr86U9/wtq1a6UeDiGhjEIPIf6wfv16DA8Pf+XXH3vsMWg0GuF/f/rpp6itrZV1BSkxTU5O4uDBg9BqtWhra8Oll14KjUaDtWvXIioqCk888QQOHTqEPXv2yLY5aqgZHh5GX18fqqqqZNOQUU7cbrcQgBiGEQLQQo5c8iWSKfAEjhzvTQ0MDOCaa67Bb3/7W1xwwQVSD4eQUEehh5BAePHFF/H73/8eR44cCfm7ErPldDrxxhtvQKvVoqGhAUqlEl6vFzqdjjrQB8jAwACGh4ehVqsRHR0t9XBkz+v1Cs1QXS7XvHoB0a5a4NntdrS0tMgq8AwPD2PHjh345S9/iYsvvljUxzp06BDuvfdeMAyDXbt24bvf/a6oj0eITFHoIURshw4dwn333Yd3331Xdh3t5YBlWfzXf/0Xuru7oVQq8dlnn2HdunWorq7GhRdeSLsPIunt7YXVaoVKpZK0clmwmk8vIAo8gSfHwGMymbB9+3b87Gc/w/r160V9LIZhUFxcjMOHD2PFihVYu3YtXnvtNaoOR8IRhR5CxFZYWAi32y10tD/vvPPw3HPPSTwqefB4PLj11ltRUFCAH//4x4iIiIDX68Xbb78NnU6HDz74AGvXrkV1dTUuueQSmij6Ad9sdnJyEhUVFbLqVROsZuoFlJ6ejuTkZOH5NZlM6OnpoWOEAcQ3e1Wr1bLZYbdYLNixYwceffRRXHXVVaI/3kcffYQf/ehHeOONNwAAP/3pTwEAu3fvFv2xCZGZGUMPnXEgxI+OHTsm9RBkyeFw4Nprr8UVV1yBe++9V/j1mJgYXHHFFbjiiivg8/nwwQcfoKamBt/73vegVqtRXV2Nyy+/XDYXkYMJx3E4duwYPB4PKisrw/Zumb9FRkYiLS0NaWlpYFlW6AV09OhRxMfHIyYmBjabDatXr6bAEyByDDxjY2O45ppr8IMf/CAggQf44ghrdna28P9XrFiBf/3rXwF5bEKCAYUeQoioRkdHUV1djTvvvPO0leyio6NxySWX4JJLLgHDMPjoo4+g1Wrx4x//GKWlpaiursaGDRuoj88scByHtrY2REREoKysjAKPSCIjI5GSkoKUlBRwHIfOzk4MDg4iOjoaLS0tkvQCCjd8U2M59Zuy2Wy45ppr8OCDD2LLli1SD4cQ8v9Q6CGEiMrpdOLhhx/GlVdeOeu/ExUVhQsvvBAXXnghWJbFp59+ipqaGjz55JMoKCjA1q1bcfXVV1MRhBmwLIuWlhYsWbIEBQUFFHgCZGhoCOPj41i3bh2ioqKEXkCff/45YmJiAtILKNxMTk6isbERKpVKNoshExMTuPbaa3HPPfdg+/btAX3srKws9PX1Cf+/v78fWVlZAR0DIXJGd3oIIUGDZVkYDAbU1NTg0KFDyMrKwtatW7Fp0yYkJSVJPTzJsSyLxsZGJCQkID8/X+rhhA2+Ml5VVdWMhSKmpqZgMplgNpsREREhBCA6tjl/DocDRqMRlZWViIuLk3o4AL48xnvbbbfh5ptvDvjj+3w+FBcX48iRI8jKysLatWvx6quvory8POBjIURiVMiAEBI6OI5Dc3MztFot6urqkJqaiurqamzevFkoJBFOGIaB0WhEamoqcnJypB5O2Ojv74fJZIJarZ5VZTyXyyWUwvZXL6BwI8fA43Q6ce211+LGG2/EN77xDcnGcfDgQfz3f/83GIbB7bffjocffliysRAiIQo9hJDQxHEcjh49Cq1WiwMHDiA2NhYajQZbtmyBUqkM+SNePp8PBoMBmZmZWL58udTDCRsLLQU+Uy+g9PR0xMXFhfxrdr6mpqZgMBhQUVEhm+OtLpcLN954IzQaDe666y76b0eI9Cj0EEJCH3+hXKfTYe/evYiJicHWrVuh0WiQmZkZchMSr9cLvV6PnJwcpKenSz2csNHT04OxsTGoVCq/lAL3+XywWq0YGRmBw+FAamoqlEolEhMTQ+41O19yDDxutxs33XQTNmzYgHvuuYf+WxEiDxR6CCHhheM49Pf3Q6vVYu/evfD5fNiyZQuqq6uRnZ0d9BMUt9sNg8GA/Px8aoYbQN3d3bDZbKisrBSl9xHDMEIvILvdjqSkJCiVyhN6AYUbp9MJg8GAsrIyJCQkSD0cAF8sONx66604//zzcf/99wf95wkhIYRCDyEkfHEch6GhIdTW1mLPnj1wOBzYtGkTNBpNUFY5c7lc0Ov1KC4uRkpKitTDCRudnZ0BbfY6vRfQ2NgY4uPjoVQqkZqaOq8jdcFIjoHH5/PhG9/4BqqqqvDQQw8F3ecHISGOQg8hhPBMJhP27NmD2tpajI6OYuPGjdBoNCgpKZH9BGZqagpGoxGrVq2iqnUBwh+bdDqdKCsrk2THheM42O12mEwmWK1WLFu2DAqFAgqFAtHRodmBgg88paWlSExMlHo4AL7YibvrrrtQWFiIH/3oR7L/vCAkDFHoIYSQmYyOjmLfvn3Q6XQYHBzElVdeiW3btkk2uT0dvjeJnO41hDqO43D8+HG43W7ZNHvlOA6Tk5MwmUywWCxYtGgRlEolFApFyPQC4ncz5RZ47rnnHqSnp+OnP/2pLF4LhJCvoNBDCCFnYrPZcODAAeh0OnR1dWHDhg2orq6GWq2WPADZ7XY0NzfLqhljqOM4Dh0dHfD5fCgtLZXtJDfUegHxgUdOu5ksy+K+++5DXFwcnnrqKck/Dwghp0ShhxBC5mJiYgIHDx6EVqtFe3s7LrvsMmg0GqxduzbgE56xsTG0t7dDrVZj6dKlAX3scMWXQmdZFqtWrZJt4DlZsPcCcrvdaGhoQElJCZKTk6UeDoAvAs93v/tdAMCvf/1rCjyEyBuFHkIImS+n04lDhw5Bp9NBr9fj4osvhkajwbp160S/UG61WnHs2DGo1eqgXbkPNhzHob29HRERESguLg6awHMyj8cDi8WCkZEReDwepKWlQalUyrYXkNvtFgp0yCnw/OAHP8DExAR+//vfU+AhRP4o9BBCiD+43W4cPnwYWq0Wn376KdatW4dt27bhggsuQExMjF8fy2Qyobu7G1VVVSFzV0PuOI5Da2sroqOjUVRUJMtwMB8+nw8WiwUmk0mWvYA8Hg8aGhpQVFQkm4qEHMfh0UcfxdDQEP785z+HTcU8QoIchR5CSGAcOnQI9957LxiGwa5du4RjIaHI4/Hg7bffhk6nwz//+U+cc845qK6uxsUXX7zgkDI0NISBgQGo1Wq/hykyM47j0NLSgkWLFqGwsFAWYUAMcusFJNfA8+STT+L48eP4y1/+ErIV8ggJQRR6CCHiYxgGxcXFOHz4MFasWIG1a9fitddeQ1lZmdRDE53P58P777+PmpoavPvuu1i9ejWqq6tx2WWXzflYWn9/P0wmE9RqNa0uBwjHcWhubsbSpUuxcuXKkA08Jzu5F1BCQgKUSiVSUlIC8trjA09hYSFSU1NFf7zZ4DgOv/zlL6HX6/Hqq6/SogMhwYVCDyFEfB999BF+9KMf4Y033gAA/PSnPwUA7N69W8phBRzDMPjwww+h1Wrx1ltvoaysDNXV1diwYQOWLVt22r97/PhxTExMoLKykgJPgLAsi+bmZsTGxmLlypVSD0cyHMfBZrMJvYBiY2OhVCqRlpYmyk6Hx+OBXq/HypUrkZaW5vefPx8cx+F3v/sd/vnPf+Lvf/87HSslJPjMGHpor5YQ4lcDAwPIzs4W/v+KFSvwr3/9S8IRSSMqKgoXXXQRLrroIrAsi08++QQ1NTV44oknUFhYCI1GgyuvvPKEXjssy+J//ud/EBERgSeffJIuTAcIy7JoampCfHw88vPzpR6OpCIiIpCUlISkpKQTegH19PT4vReQ1+uVZeD505/+hHfeeQe1tbUUeAgJIRR6CCFEZJGRkTj33HNx7rnngmVZ6PV6aLVaPPPMM8jOzsbWrVtx1VVX4aGHHoLdbsdf//pXCjwBwrIsGhsbkZSUhNzcXKmHIysRERGIj49HfHw8CgoK4HA4YDabYTAYEBkZKQSg+VQU9Hq9aGhoQH5+vmwCDwD85S9/wcGDB7Fv3z4sXrxY6uEQQvyIQg8hxK+ysrLQ19cn/P/+/n5kZWVJOCJ5iYyMxJo1a7BmzRo89thjaGpqQk1NDc455xxkZ2fj1ltvhc1mk83dhlDGsiyMRiNSUlKQk5Mj9XBkLzY2FrGxscjLy4PL5YLJZEJzczNYlhV6AZ3p6Cbw5Q5PXl4eFApFAEY+O6+88gp0Oh0OHDhApeEJCUF0p4cQ4lc+nw/FxcU4cuQIsrKysHbtWrz66qsoLy+Xemiy5PP5cOuttyI/Px833nijMOlKSEjA1q1bsWXLFigUirC5VB8oDMPAaDQiLS3thOOYZO48Ho/QDPVMvYB8Ph8aGhqQm5sLpVIp0Yi/6u9//zteeOEF1NfXIy4uTurhEEIWhgoZEEIC4+DBg/jv//5vMAyD22+/HQ8//LDUQ5Ilt9uNG264AevWrcMDDzwg/DrHcTh+/Dh0Oh327duHRYsWYevWrdBoNMjIyKAAtEAMw8BgMECpVGLFihVSDyekTO8FNDU1hdTUVCgUCiQmJoJhGDQ0NCAnJwfp6elSD1Wwd+9ePPvss6irq0NiYqLUwyGELByFHkIIkYupqSns2LEDW7duxd13333KP8dxHHp7e6HT6bB3716wLIstW7aguroaK1asoAA0RwzDQK/XIzMzE8uXL5d6OCGNYRhYrVahF5DP58Py5cuxcuVK2dxZq6+vxzPPPIP6+nokJydLPRxCiH9Q6CGEEDngOA5XXnklvv71r+Pmm2+e098bGhqCTqfDnj17MDU1hc2bN0Oj0YRVX5n58vl80Ov1yMrKQmZmptTDCRv8Dk9SUhK8Xi/Gx8cD3gtoJm+++SZ++tOf4uDBg3SHjpDQQqGHEELkYnBwcME7DSMjI9izZw9qa2sxNjaGjRs3orq6GsXFxRSATsLfJcnOzkZGRobUwwkb/M7a8uXLhaAZ6F5AM3nrrbfwox/9CAcPHpTV3SJCiF9Q6CGEkFBltVqxb98+6HQ6DA8P48orr8S2bdtQWloqm6NEUuGrhcntLkmom81Rwum9gCwWCxYtWoT09HSkpaWJ1iPnvffew8MPP4z6+noKwISEJgo9hBASDsbHx3HgwAHodDp0d3djw4YNqK6uhlqtDrsAxPeDycvLoxX9AOKLRaSnp8+pZL3D4YDJZILZbEZUVNSCegHN5KOPPsL999+Puro6KqVPSOii0EMIIeFmYmIC9fX10Ol0aG9vx+WXXw6NRoOzzz475AOQx+OBXq9Hfn6+rPrBhLr5Bp6T8b2ATCYTOI6bUy+gmXzyySe49957sX//furLREhoo9BDCCHhbGpqCocOHYJOp4PRaMTFF18MjUaD8847T7LL5GLxeDxoaGhAQUEB0tLSpB5O2GBZFgaDAQqFwq/lwGfqBZSeno7Y2NhZ3V9raGjA3Xffjb1792LlypV+GxchRJYo9BBCCPmCy+XC4cOHodVq8dlnn+H888/Htm3bcMEFFwTsMrlY3G439Ho9CgsLqSpXALEsC6PRiNTUVFEbvnq9XqEXkNPpRGpqKpRKJRISEmYMQI2Njbjjjjug1WpRXFws2rgIIbJBoYcQQshXeTwevPXWW9DpdPjwww9x7rnnorq6Gl/72tdEu0wuFpfLBb1ej+LiYqSkpEg9nLARqMBzsum9gI4dO4b9+/djx44d2LBhAxYtWoSWlhbcfvvt+Nvf/oaysrKAjYsQIikKPYQQQk7P5/PhvffeQ01NDd577z2sWbMG1dXVuPTSS/12mVwsfOApKSmhRpMBxLIsGhsbkZycLOldGbfbjddffx179uzBZ599hlWrVqGjowN/+9vfcNZZZ0k2LkJIwFHoIYQQMnsMw+Cf//wntFot3n77bZSXl0Oj0WDDhg3zvkwuFqfTCYPBgFWrViEpKUnq4YQNPvAkJSUhNzdX6uEIOjo6cNddd6GgoACNjY0oLy/H9u3bcdVVVyEuLk7q4RFCxEWhhxBCyPywLIt///vfqKmpweHDh1FUVITq6mpceeWVkk8ip6amYDQaUVpaisTEREnHEk5YlkVTUxMSEhKQl5cn9XAEPT09uP766/GnP/0Ja9euBcdx0Ov1qK2txeuvv44VK1Zg+/bt2LJlC+0IEhKaKPQQQghZOJZl0dDQAK1Wi0OHDiEnJwdbt27Fxo0bAx46HA4HjEYjysvLkZCQENDHDmccx6GpqQlxcXHIz8+XejiCgYEBXHPNNXj22Wexbt26Gf9MW1sb9uzZgwMHDqCuro7ufhESeij0EEII8S9+tV+r1aK+vh5KpRIajQabN28WfTLJB56KigrEx8eL+ljkSxzHobm5GcuWLZNV+eehoSHs3LkTv/zlL3HxxRdLPRxCiHQo9BBCCBEPx3Foa2uDVqtFXV0dEhMTsXXrVmzevBkKhWJW/VRma3JyEo2NjaisrJT8eF044QPP0qVLUVBQIPVwBCMjI9ixYwd+/vOf4/LLL5d6OIQQaVHoIYQQKfT19eHmm2/GyMgIIiIicOedd+Lee++Velii4jgOx48fh1arxf79+7F48WJs2bIFGo0GGRkZCwpAExMTaGpqosATYBzHoaWlBYsXL0ZhYaHUwxFYLBZs374dP/nJT3DVVVdJPRxCiPQo9BBCiBSGhoYwNDSENWvWYGJiAmeddRb27t0bNn1DOI5DT08PdDod9u7dCwDYvHkzqqursWLFijkFILvdjubmZqhUKsTGxoo0YnIyjuPQ2tqKRYsWoaCgwK+7dgsxOjqK7du34/vf/z62bNki9XAIIfJAoYcQQuRAo9Hgm9/8JjZs2CD1UAKO4zgMDg5Cp9Nhz549cLlc2Lx5MzQaDfLz8087mR4ZGUFXVxdUKpXsSmaHMv7YYnR0NAoLC2UTeGw2G7Zv344HHngA27dvl3o4hBD5oNBDCCFS6+7uxte+9jWh1G844zgOJpMJe/bsgU6ng81mw8aNG6HRaFBcXHzC5Prtt9/Gd7/7XRw5coSOtAUQH3iioqJQVFQkm8AzMTGBnTt34pvf/Cauu+46qYdDCJEXCj2EECKlyclJXHzxxXj44YdpZXoGFosF+/btg06nw8jICK666ips27YN/f39eOCBB6DValFUVCT1MMMGx3Fob29HRETEV0KolBwOB6655hp84xvfwE033ST1cAgh8kOhhxBCpOL1erF582ZceeWVuO+++6QejuyNj49j//79+MMf/oCOjg5cd911uP7666FSqRAZGSn18EIex3E4evQoOI5DSUmJbAKP0+nEtddei//v//v/cPvtt0s9HEKIPM34gRUd6FEQQki44TgO3/jGN1BaWkqBZ5aSkpKwYsUKuN1uvP/++/jss8/w9NNPo6OjA5dddhmqq6tx1llnUQASAcdx6OjokF3gcblcuPHGG3HNNdfgtttuk3o4hJAgQzs9hBAisg8++AAXXXQRKisrhUn6448/jo0bN0o8Mvk6fPgwHn74YRw4cADp6enCr09NTeHgwYOora1FY2MjLr74YlRXV+Pcc89FVFSUhCMODRzH4dixY/D5fFi1apVsAo/b7cZNN92EK664At/61rdkMy5CiCzR8TZCCCHy9/rrr+ORRx7BgQMHoFAoTvnnXC4X3nzzTWi1Wnz++ee44IILsG3bNpx//vmIjqaDDHPF91byeDwoLS2VTbDwer245ZZbcOGFF+I73/mObMZFCJEtCj2EEELk7ciRI/j+97+PAwcOIDU1ddZ/z+Px4MiRI9DpdPjoo49w3nnnobq6GhdddBEWLVok4ohDx7Fjx+B2u1FWViabYOHz+XD77bdjzZo12L17d8DG9cADD+DAgQNCX6IXXngBSUlJAXlsQsiCUeghhBAib2azGdHR0UhOTp73z/B6vXjvvfdQU1OD999/H2eddRY0Gg0uu+wyLF682I+jDR3Hjx+Hy+WSVeBhGAb/8R//gaKiIvzoRz8K6LjefPNNXHbZZYiOjsaDDz4IAHjyyScD9viEkAWh0EMIISS8MAyDDz74AFqtFu+88w4qKiqg0Wiwfv16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"/>


```python
df = pd.DataFrame(data = x_train, columns = x.columns)
df                            
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Time_min</th>
      <th>Time_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.007274</td>
      <td>0.474937</td>
      <td>0.223980</td>
      <td>-0.481952</td>
      <td>0.303134</td>
      <td>-0.571332</td>
      <td>0.700344</td>
      <td>-0.025371</td>
      <td>-0.247199</td>
      <td>-0.223258</td>
      <td>...</td>
      <td>-0.612348</td>
      <td>0.124840</td>
      <td>0.004893</td>
      <td>-1.063352</td>
      <td>0.253799</td>
      <td>0.600826</td>
      <td>0.259707</td>
      <td>-0.774818</td>
      <td>-1.148344</td>
      <td>1.191669</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.595377</td>
      <td>0.147610</td>
      <td>0.095786</td>
      <td>0.342616</td>
      <td>-0.157039</td>
      <td>-0.771296</td>
      <td>0.308038</td>
      <td>-0.195672</td>
      <td>-0.483848</td>
      <td>0.081477</td>
      <td>...</td>
      <td>-1.339388</td>
      <td>0.206031</td>
      <td>0.885019</td>
      <td>0.446872</td>
      <td>0.217719</td>
      <td>-0.170673</td>
      <td>0.019689</td>
      <td>0.272561</td>
      <td>0.975166</td>
      <td>0.162408</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.650220</td>
      <td>0.137863</td>
      <td>0.197353</td>
      <td>0.362041</td>
      <td>-0.165517</td>
      <td>-0.522780</td>
      <td>0.052893</td>
      <td>-0.161942</td>
      <td>0.051250</td>
      <td>-0.065653</td>
      <td>...</td>
      <td>-1.145077</td>
      <td>0.107428</td>
      <td>-0.191973</td>
      <td>0.548220</td>
      <td>0.259483</td>
      <td>-0.073468</td>
      <td>0.045123</td>
      <td>-0.429514</td>
      <td>0.630813</td>
      <td>0.333952</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.494170</td>
      <td>-0.494214</td>
      <td>1.066663</td>
      <td>0.247405</td>
      <td>-1.031336</td>
      <td>0.576071</td>
      <td>-0.933914</td>
      <td>0.395149</td>
      <td>1.319652</td>
      <td>-0.442554</td>
      <td>...</td>
      <td>0.593444</td>
      <td>-0.040693</td>
      <td>0.578249</td>
      <td>0.178571</td>
      <td>2.353092</td>
      <td>0.008320</td>
      <td>0.043717</td>
      <td>0.544156</td>
      <td>0.688205</td>
      <td>-0.009135</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.436816</td>
      <td>0.987243</td>
      <td>0.556139</td>
      <td>1.263973</td>
      <td>-0.490482</td>
      <td>-0.242393</td>
      <td>-0.238900</td>
      <td>0.611539</td>
      <td>0.294178</td>
      <td>-0.631416</td>
      <td>...</td>
      <td>-0.223364</td>
      <td>0.470059</td>
      <td>0.513982</td>
      <td>-2.051359</td>
      <td>-1.183402</td>
      <td>-0.311946</td>
      <td>0.173473</td>
      <td>0.093765</td>
      <td>-0.057893</td>
      <td>-1.381482</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>170879</th>
      <td>0.718821</td>
      <td>-0.712755</td>
      <td>-1.297836</td>
      <td>0.097117</td>
      <td>0.592898</td>
      <td>1.108797</td>
      <td>0.117697</td>
      <td>0.265530</td>
      <td>0.331400</td>
      <td>-0.028647</td>
      <td>...</td>
      <td>-0.620138</td>
      <td>0.106713</td>
      <td>-1.529695</td>
      <td>-0.842082</td>
      <td>0.684636</td>
      <td>-0.213964</td>
      <td>-0.111509</td>
      <td>1.329413</td>
      <td>1.376911</td>
      <td>-0.180679</td>
    </tr>
    <tr>
      <th>170880</th>
      <td>1.041211</td>
      <td>-0.068335</td>
      <td>-0.941296</td>
      <td>0.134285</td>
      <td>0.070934</td>
      <td>-0.657911</td>
      <td>0.069291</td>
      <td>-0.167937</td>
      <td>0.444552</td>
      <td>0.193509</td>
      <td>...</td>
      <td>1.184423</td>
      <td>-0.062173</td>
      <td>-0.532832</td>
      <td>0.482308</td>
      <td>-0.216729</td>
      <td>-0.071862</td>
      <td>-0.222746</td>
      <td>-1.516657</td>
      <td>-0.172677</td>
      <td>0.333952</td>
    </tr>
    <tr>
      <th>170881</th>
      <td>0.980405</td>
      <td>-0.347498</td>
      <td>-0.329220</td>
      <td>0.088848</td>
      <td>-0.380486</td>
      <td>-0.061013</td>
      <td>-0.506067</td>
      <td>0.016965</td>
      <td>1.322873</td>
      <td>-0.231978</td>
      <td>...</td>
      <td>1.007595</td>
      <td>0.199506</td>
      <td>1.085185</td>
      <td>-0.287211</td>
      <td>0.202714</td>
      <td>0.024384</td>
      <td>-0.089529</td>
      <td>0.418347</td>
      <td>-0.402246</td>
      <td>-0.009135</td>
    </tr>
    <tr>
      <th>170882</th>
      <td>0.648419</td>
      <td>-0.563452</td>
      <td>0.711854</td>
      <td>-0.319424</td>
      <td>-1.234643</td>
      <td>-0.371599</td>
      <td>-0.918423</td>
      <td>0.034653</td>
      <td>0.019981</td>
      <td>0.471720</td>
      <td>...</td>
      <td>1.204952</td>
      <td>-0.199081</td>
      <td>0.700350</td>
      <td>0.795275</td>
      <td>-0.097881</td>
      <td>0.090942</td>
      <td>0.091516</td>
      <td>0.479224</td>
      <td>0.688205</td>
      <td>1.020125</td>
    </tr>
    <tr>
      <th>170883</th>
      <td>1.032404</td>
      <td>-0.026512</td>
      <td>-0.937984</td>
      <td>0.107213</td>
      <td>0.199223</td>
      <td>-0.475986</td>
      <td>0.087325</td>
      <td>-0.155606</td>
      <td>0.267909</td>
      <td>0.157432</td>
      <td>...</td>
      <td>1.311575</td>
      <td>0.040092</td>
      <td>1.328817</td>
      <td>0.493614</td>
      <td>-0.340813</td>
      <td>-0.067173</td>
      <td>-0.183249</td>
      <td>-0.968893</td>
      <td>-1.033559</td>
      <td>-0.523765</td>
    </tr>
  </tbody>
</table>
<p>170884 rows × 31 columns</p>
</div>



```python
y_f = pd.Series(y_train.flatten())
a, b = y_f.value_counts()
num = a - b

new_data = generating_data(num)
gen_df = pd.DataFrame(data = new_data, columns = x.columns)
```


```python
gen_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Time_min</th>
      <th>Time_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.165219</td>
      <td>0.870731</td>
      <td>0.996099</td>
      <td>0.958943</td>
      <td>0.976154</td>
      <td>-0.987933</td>
      <td>-0.999630</td>
      <td>0.907773</td>
      <td>-0.943535</td>
      <td>0.999974</td>
      <td>...</td>
      <td>-0.999186</td>
      <td>0.631774</td>
      <td>0.980529</td>
      <td>-0.933450</td>
      <td>-0.198305</td>
      <td>0.974120</td>
      <td>-0.353116</td>
      <td>-0.988463</td>
      <td>-0.999980</td>
      <td>0.829192</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.762698</td>
      <td>0.963720</td>
      <td>0.995324</td>
      <td>0.907733</td>
      <td>0.549134</td>
      <td>-0.999980</td>
      <td>-0.996285</td>
      <td>0.933210</td>
      <td>-0.980389</td>
      <td>0.999999</td>
      <td>...</td>
      <td>-0.834882</td>
      <td>0.163097</td>
      <td>0.840782</td>
      <td>-0.078388</td>
      <td>-0.993612</td>
      <td>0.532782</td>
      <td>-0.878611</td>
      <td>-0.999786</td>
      <td>-0.984417</td>
      <td>0.998733</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.845856</td>
      <td>0.983253</td>
      <td>0.708059</td>
      <td>0.935839</td>
      <td>0.387381</td>
      <td>-0.971113</td>
      <td>0.607851</td>
      <td>0.864022</td>
      <td>-0.038962</td>
      <td>0.999999</td>
      <td>...</td>
      <td>-0.962818</td>
      <td>-0.483413</td>
      <td>-0.892406</td>
      <td>0.999910</td>
      <td>-0.944640</td>
      <td>-0.688118</td>
      <td>-0.515916</td>
      <td>-0.999700</td>
      <td>0.999648</td>
      <td>0.970639</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.828410</td>
      <td>0.738734</td>
      <td>0.702996</td>
      <td>0.926159</td>
      <td>-0.783304</td>
      <td>0.845583</td>
      <td>0.751912</td>
      <td>0.500486</td>
      <td>0.773429</td>
      <td>0.999999</td>
      <td>...</td>
      <td>-0.903254</td>
      <td>0.080720</td>
      <td>-0.999253</td>
      <td>0.999982</td>
      <td>0.263672</td>
      <td>-0.882510</td>
      <td>0.155447</td>
      <td>-0.832607</td>
      <td>0.998881</td>
      <td>-0.994316</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.980319</td>
      <td>0.966662</td>
      <td>0.983600</td>
      <td>0.949761</td>
      <td>0.927061</td>
      <td>-0.324670</td>
      <td>-0.664392</td>
      <td>0.944043</td>
      <td>0.828564</td>
      <td>0.999997</td>
      <td>...</td>
      <td>-0.999933</td>
      <td>0.239915</td>
      <td>0.597162</td>
      <td>0.998042</td>
      <td>0.025318</td>
      <td>0.703757</td>
      <td>0.768541</td>
      <td>-0.993173</td>
      <td>-0.892862</td>
      <td>-0.630355</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>170289</th>
      <td>-0.801456</td>
      <td>0.971025</td>
      <td>-0.065070</td>
      <td>0.914025</td>
      <td>-0.153794</td>
      <td>-0.960953</td>
      <td>0.651190</td>
      <td>0.618203</td>
      <td>0.294000</td>
      <td>0.999998</td>
      <td>...</td>
      <td>-0.702776</td>
      <td>-0.485996</td>
      <td>-0.978242</td>
      <td>0.999965</td>
      <td>-0.868960</td>
      <td>-0.790089</td>
      <td>-0.556687</td>
      <td>-0.998903</td>
      <td>0.999947</td>
      <td>0.827502</td>
    </tr>
    <tr>
      <th>170290</th>
      <td>0.988528</td>
      <td>0.408065</td>
      <td>0.655748</td>
      <td>0.917650</td>
      <td>-0.985297</td>
      <td>-0.999949</td>
      <td>0.659603</td>
      <td>-0.428200</td>
      <td>-0.963527</td>
      <td>0.999999</td>
      <td>...</td>
      <td>0.985336</td>
      <td>0.739728</td>
      <td>-0.995644</td>
      <td>0.999465</td>
      <td>-0.999818</td>
      <td>-0.787551</td>
      <td>-0.987513</td>
      <td>-0.996360</td>
      <td>0.999999</td>
      <td>0.999121</td>
    </tr>
    <tr>
      <th>170291</th>
      <td>0.838168</td>
      <td>0.234251</td>
      <td>0.054434</td>
      <td>0.765039</td>
      <td>-0.991848</td>
      <td>-0.996654</td>
      <td>0.518166</td>
      <td>-0.629423</td>
      <td>-0.875301</td>
      <td>0.999996</td>
      <td>...</td>
      <td>0.970727</td>
      <td>0.529590</td>
      <td>-0.999301</td>
      <td>0.999740</td>
      <td>-0.979201</td>
      <td>-0.904412</td>
      <td>-0.945858</td>
      <td>-0.966840</td>
      <td>1.000000</td>
      <td>0.728408</td>
    </tr>
    <tr>
      <th>170292</th>
      <td>0.983299</td>
      <td>0.370063</td>
      <td>0.889434</td>
      <td>0.870955</td>
      <td>-0.985107</td>
      <td>-0.999996</td>
      <td>0.697117</td>
      <td>-0.096197</td>
      <td>-0.987841</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.964094</td>
      <td>0.647713</td>
      <td>-0.962607</td>
      <td>0.996969</td>
      <td>-0.998371</td>
      <td>-0.751535</td>
      <td>-0.983631</td>
      <td>-0.998795</td>
      <td>0.999994</td>
      <td>0.999176</td>
    </tr>
    <tr>
      <th>170293</th>
      <td>-0.737267</td>
      <td>0.994782</td>
      <td>0.916281</td>
      <td>0.923894</td>
      <td>0.925581</td>
      <td>-0.995830</td>
      <td>-0.419903</td>
      <td>0.967831</td>
      <td>0.375171</td>
      <td>0.999999</td>
      <td>...</td>
      <td>-0.998515</td>
      <td>-0.463559</td>
      <td>-0.274363</td>
      <td>0.999485</td>
      <td>-0.917533</td>
      <td>0.050413</td>
      <td>0.016243</td>
      <td>-0.999966</td>
      <td>0.713525</td>
      <td>0.993664</td>
    </tr>
  </tbody>
</table>
<p>170294 rows × 31 columns</p>
</div>



```python
gen_df.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Time_min</th>
      <th>Time_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>...</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
      <td>170294.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.165608</td>
      <td>0.810281</td>
      <td>0.722485</td>
      <td>0.896805</td>
      <td>0.135813</td>
      <td>-0.730733</td>
      <td>-0.115157</td>
      <td>0.590979</td>
      <td>-0.347246</td>
      <td>0.999430</td>
      <td>...</td>
      <td>-0.436366</td>
      <td>0.101571</td>
      <td>-0.310111</td>
      <td>0.795268</td>
      <td>-0.582673</td>
      <td>-0.147601</td>
      <td>-0.527674</td>
      <td>-0.897711</td>
      <td>0.389096</td>
      <td>0.558983</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.729527</td>
      <td>0.389233</td>
      <td>0.365811</td>
      <td>0.087573</td>
      <td>0.766965</td>
      <td>0.581409</td>
      <td>0.717845</td>
      <td>0.549228</td>
      <td>0.663379</td>
      <td>0.024673</td>
      <td>...</td>
      <td>0.767320</td>
      <td>0.380860</td>
      <td>0.740923</td>
      <td>0.501270</td>
      <td>0.647729</td>
      <td>0.635748</td>
      <td>0.522664</td>
      <td>0.361618</td>
      <td>0.824662</td>
      <td>0.737364</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.999981</td>
      <td>-0.999967</td>
      <td>-0.992312</td>
      <td>-0.120296</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-0.999004</td>
      <td>-0.999988</td>
      <td>-0.999988</td>
      <td>...</td>
      <td>-0.999999</td>
      <td>-0.889422</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-0.999992</td>
      <td>-0.999972</td>
      <td>-0.999892</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.852140</td>
      <td>0.871784</td>
      <td>0.596323</td>
      <td>0.866258</td>
      <td>-0.716700</td>
      <td>-0.999939</td>
      <td>-0.860717</td>
      <td>0.422094</td>
      <td>-0.941019</td>
      <td>0.999994</td>
      <td>...</td>
      <td>-0.997758</td>
      <td>-0.197246</td>
      <td>-0.977001</td>
      <td>0.957017</td>
      <td>-0.995293</td>
      <td>-0.745137</td>
      <td>-0.922054</td>
      <td>-0.999962</td>
      <td>-0.555654</td>
      <td>0.462113</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.414175</td>
      <td>0.961485</td>
      <td>0.891746</td>
      <td>0.921059</td>
      <td>0.369932</td>
      <td>-0.997711</td>
      <td>-0.189443</td>
      <td>0.872755</td>
      <td>-0.632340</td>
      <td>0.999998</td>
      <td>...</td>
      <td>-0.940988</td>
      <td>0.090759</td>
      <td>-0.682491</td>
      <td>0.998217</td>
      <td>-0.956379</td>
      <td>-0.234267</td>
      <td>-0.760471</td>
      <td>-0.999752</td>
      <td>0.970877</td>
      <td>0.992588</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.594280</td>
      <td>0.983231</td>
      <td>0.979723</td>
      <td>0.955972</td>
      <td>0.892124</td>
      <td>-0.905365</td>
      <td>0.607869</td>
      <td>0.964585</td>
      <td>0.230374</td>
      <td>0.999999</td>
      <td>...</td>
      <td>0.206063</td>
      <td>0.393290</td>
      <td>0.440839</td>
      <td>0.999833</td>
      <td>-0.424188</td>
      <td>0.402999</td>
      <td>-0.277337</td>
      <td>-0.995757</td>
      <td>0.999788</td>
      <td>0.999709</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.999999</td>
      <td>0.998909</td>
      <td>0.999929</td>
      <td>0.999741</td>
      <td>0.999133</td>
      <td>1.000000</td>
      <td>0.997772</td>
      <td>0.999651</td>
      <td>0.998050</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.980001</td>
      <td>0.999278</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.999453</td>
      <td>0.996087</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>



```python
gen_df['Class'] = 1
df['Class'] = y_f
```


```python
gen_x = gen_df.drop(['Class'], axis = 1)
gen_y = gen_df.Class
```


```python
# generating
gan_data = pd.concat([df, gen_df], axis = 0, sort = False).reset_index(drop = True)
```


```python
gan_data
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Time_min</th>
      <th>Time_hour</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.007274</td>
      <td>0.474937</td>
      <td>0.223980</td>
      <td>-0.481952</td>
      <td>0.303134</td>
      <td>-0.571332</td>
      <td>0.700344</td>
      <td>-0.025371</td>
      <td>-0.247199</td>
      <td>-0.223258</td>
      <td>...</td>
      <td>0.124840</td>
      <td>0.004893</td>
      <td>-1.063352</td>
      <td>0.253799</td>
      <td>0.600826</td>
      <td>0.259707</td>
      <td>-0.774818</td>
      <td>-1.148344</td>
      <td>1.191669</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.595377</td>
      <td>0.147610</td>
      <td>0.095786</td>
      <td>0.342616</td>
      <td>-0.157039</td>
      <td>-0.771296</td>
      <td>0.308038</td>
      <td>-0.195672</td>
      <td>-0.483848</td>
      <td>0.081477</td>
      <td>...</td>
      <td>0.206031</td>
      <td>0.885019</td>
      <td>0.446872</td>
      <td>0.217719</td>
      <td>-0.170673</td>
      <td>0.019689</td>
      <td>0.272561</td>
      <td>0.975166</td>
      <td>0.162408</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.650220</td>
      <td>0.137863</td>
      <td>0.197353</td>
      <td>0.362041</td>
      <td>-0.165517</td>
      <td>-0.522780</td>
      <td>0.052893</td>
      <td>-0.161942</td>
      <td>0.051250</td>
      <td>-0.065653</td>
      <td>...</td>
      <td>0.107428</td>
      <td>-0.191973</td>
      <td>0.548220</td>
      <td>0.259483</td>
      <td>-0.073468</td>
      <td>0.045123</td>
      <td>-0.429514</td>
      <td>0.630813</td>
      <td>0.333952</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.494170</td>
      <td>-0.494214</td>
      <td>1.066663</td>
      <td>0.247405</td>
      <td>-1.031336</td>
      <td>0.576071</td>
      <td>-0.933914</td>
      <td>0.395149</td>
      <td>1.319652</td>
      <td>-0.442554</td>
      <td>...</td>
      <td>-0.040693</td>
      <td>0.578249</td>
      <td>0.178571</td>
      <td>2.353092</td>
      <td>0.008320</td>
      <td>0.043717</td>
      <td>0.544156</td>
      <td>0.688205</td>
      <td>-0.009135</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.436816</td>
      <td>0.987243</td>
      <td>0.556139</td>
      <td>1.263973</td>
      <td>-0.490482</td>
      <td>-0.242393</td>
      <td>-0.238900</td>
      <td>0.611539</td>
      <td>0.294178</td>
      <td>-0.631416</td>
      <td>...</td>
      <td>0.470059</td>
      <td>0.513982</td>
      <td>-2.051359</td>
      <td>-1.183402</td>
      <td>-0.311946</td>
      <td>0.173473</td>
      <td>0.093765</td>
      <td>-0.057893</td>
      <td>-1.381482</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>341173</th>
      <td>-0.801456</td>
      <td>0.971025</td>
      <td>-0.065070</td>
      <td>0.914025</td>
      <td>-0.153794</td>
      <td>-0.960953</td>
      <td>0.651190</td>
      <td>0.618203</td>
      <td>0.294000</td>
      <td>0.999998</td>
      <td>...</td>
      <td>-0.485996</td>
      <td>-0.978242</td>
      <td>0.999965</td>
      <td>-0.868960</td>
      <td>-0.790089</td>
      <td>-0.556687</td>
      <td>-0.998903</td>
      <td>0.999947</td>
      <td>0.827502</td>
      <td>1</td>
    </tr>
    <tr>
      <th>341174</th>
      <td>0.988528</td>
      <td>0.408065</td>
      <td>0.655748</td>
      <td>0.917650</td>
      <td>-0.985297</td>
      <td>-0.999949</td>
      <td>0.659603</td>
      <td>-0.428200</td>
      <td>-0.963527</td>
      <td>0.999999</td>
      <td>...</td>
      <td>0.739728</td>
      <td>-0.995644</td>
      <td>0.999465</td>
      <td>-0.999818</td>
      <td>-0.787551</td>
      <td>-0.987513</td>
      <td>-0.996360</td>
      <td>0.999999</td>
      <td>0.999121</td>
      <td>1</td>
    </tr>
    <tr>
      <th>341175</th>
      <td>0.838168</td>
      <td>0.234251</td>
      <td>0.054434</td>
      <td>0.765039</td>
      <td>-0.991848</td>
      <td>-0.996654</td>
      <td>0.518166</td>
      <td>-0.629423</td>
      <td>-0.875301</td>
      <td>0.999996</td>
      <td>...</td>
      <td>0.529590</td>
      <td>-0.999301</td>
      <td>0.999740</td>
      <td>-0.979201</td>
      <td>-0.904412</td>
      <td>-0.945858</td>
      <td>-0.966840</td>
      <td>1.000000</td>
      <td>0.728408</td>
      <td>1</td>
    </tr>
    <tr>
      <th>341176</th>
      <td>0.983299</td>
      <td>0.370063</td>
      <td>0.889434</td>
      <td>0.870955</td>
      <td>-0.985107</td>
      <td>-0.999996</td>
      <td>0.697117</td>
      <td>-0.096197</td>
      <td>-0.987841</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.647713</td>
      <td>-0.962607</td>
      <td>0.996969</td>
      <td>-0.998371</td>
      <td>-0.751535</td>
      <td>-0.983631</td>
      <td>-0.998795</td>
      <td>0.999994</td>
      <td>0.999176</td>
      <td>1</td>
    </tr>
    <tr>
      <th>341177</th>
      <td>-0.737267</td>
      <td>0.994782</td>
      <td>0.916281</td>
      <td>0.923894</td>
      <td>0.925581</td>
      <td>-0.995830</td>
      <td>-0.419903</td>
      <td>0.967831</td>
      <td>0.375171</td>
      <td>0.999999</td>
      <td>...</td>
      <td>-0.463559</td>
      <td>-0.274363</td>
      <td>0.999485</td>
      <td>-0.917533</td>
      <td>0.050413</td>
      <td>0.016243</td>
      <td>-0.999966</td>
      <td>0.713525</td>
      <td>0.993664</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>341178 rows × 32 columns</p>
</div>



```python
gan_data = gan_data.sample(frac = 1.0).reset_index(drop = True)
```


```python
last_gan_data = gan_data.to_csv('last_gan_data.csv', index = 'False')
```


```python
```
