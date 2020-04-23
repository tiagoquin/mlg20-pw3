

```python
# Enable fullscreen

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:85% !important; }</style>"))
```


<style>.container { width:85% !important; }</style>



```python
import numpy as np
from matplotlib import pyplot as pl
import os
import scipy.io.wavfile as wav

%matplotlib inline
```


```python
PATH = './vowels/vowels/'
```

### file name format:
* n = natural
* s = synthetic
* a = adult
* m = male
* f = female




```python
from python_speech_features import mfcc
```

## 1. Man vs Woman. Use only the natural voices of men and women to train a neural network that recognizes the gender of the speaker. 


```python
# Read all files starting with a pattern
def readFiles(pattern):
    sounds = []
    for file in os.scandir(PATH):
        if file.name.startswith(pattern):
            sample_rate, signal = wav.read(os.path.join(PATH, file.name))
            sound_class = [1] if file.name.startswith(pattern + 'f') else [-1]
            sounds.append((sample_rate, signal, file.name, sound_class))
            
    print(len(sounds), ' elements have been read')
    return sounds

# Plot one sound (signal) with its pretty name
def plotSound(signal, name):
    pl.figure(figsize=(15,6))
    pl.subplot(2,1,1)
    pl.plot(signal)
    pl.xlim(0, len(signal))
    pl.title(name)
    pl.grid()

sounds = readFiles('na')

# Let's test if the read worked...
plotSound(sounds[0][1], sounds[0][2])

# Here we print the filename and the class. We chose Female=1 and Male=-1
for sound in sounds[4:8]:
    print(sound[2], sound[3])
```

    72  elements have been read
    nafsher.wav [1]
    nafdlii.wav [1]
    namcrer.wav [-1]
    namphul.wav [-1]



![png](output_6_1.png)



```python
def doMFCC(signal, sample_rate):
    ceps = mfcc(signal, samplerate=sample_rate, nfft=1024)
    return ceps

def plotBoxSound(ceps, name):
    pl.figure(figsize=(15,4))
    pl.subplot(1,2,1)
    pl.boxplot(ceps)
    pl.title(name)
    pl.grid()
    
ceps_data = list(map(lambda s: doMFCC(s[1], s[0]) , sounds))

# List of all file names
ceps_names = list(map(lambda s: s[2] , sounds))

# List of all class
ceps_class = list(map(lambda s: s[3] , sounds))


# Some boxplot
plotBoxSound(ceps_data[0], ceps_names[0])
```


![png](output_7_0.png)



```python
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
# We need this imports to normalize our data
```


```python
# We noramlize the data
def feature(array):
    return min_max_scaler.fit_transform ( 
        list(map(lambda s: np.append( np.median(s, axis=0), np.std(s, axis=0).reshape(-1,1) ), array)),
    (-1,1))

normalized_ceps_data = feature(ceps_data)
```


```python
print(len(normalized_ceps_data[0]))
```

    26



```python
import pandas as pd

dataset = np.hstack((np.array(normalized_ceps_data), np.array(ceps_class)))

cols = np.array([
    'median 1', '2', '3', '4', '5', '6','7', 
    '8', '9', '10', '11', '12',
    '13', 'std 1', '2', '3', 'std 4', 'std 5', '6','7', 
    '8', '9', '10', '11', '12',
    '13', 'class'
])

df = pd.DataFrame(data=dataset, columns=cols, index=ceps_names)
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
      <th>median 1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>std 5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nafshuh.wav</th>
      <td>0.882007</td>
      <td>0.486714</td>
      <td>0.269900</td>
      <td>0.250803</td>
      <td>0.451598</td>
      <td>0.432843</td>
      <td>0.338669</td>
      <td>0.475385</td>
      <td>0.766298</td>
      <td>0.554415</td>
      <td>...</td>
      <td>0.250534</td>
      <td>0.613449</td>
      <td>0.042223</td>
      <td>0.301378</td>
      <td>0.337314</td>
      <td>0.419636</td>
      <td>0.229256</td>
      <td>0.082579</td>
      <td>0.229743</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>nafshoo.wav</th>
      <td>0.681413</td>
      <td>0.554893</td>
      <td>0.567343</td>
      <td>0.217012</td>
      <td>0.535919</td>
      <td>0.479145</td>
      <td>0.358913</td>
      <td>0.284798</td>
      <td>0.719476</td>
      <td>0.463529</td>
      <td>...</td>
      <td>0.396570</td>
      <td>0.203702</td>
      <td>0.049126</td>
      <td>0.857502</td>
      <td>0.030690</td>
      <td>0.150149</td>
      <td>0.663513</td>
      <td>0.530058</td>
      <td>0.273112</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>namshul.wav</th>
      <td>0.845150</td>
      <td>0.310623</td>
      <td>0.620370</td>
      <td>0.336266</td>
      <td>0.472974</td>
      <td>0.555015</td>
      <td>0.326577</td>
      <td>0.624587</td>
      <td>0.378150</td>
      <td>0.469253</td>
      <td>...</td>
      <td>0.764202</td>
      <td>0.130602</td>
      <td>0.445704</td>
      <td>0.429050</td>
      <td>0.335281</td>
      <td>0.328161</td>
      <td>0.367139</td>
      <td>0.323468</td>
      <td>0.143025</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>nafkguh.wav</th>
      <td>0.862250</td>
      <td>0.585113</td>
      <td>0.126722</td>
      <td>0.413901</td>
      <td>0.273950</td>
      <td>0.457119</td>
      <td>0.604921</td>
      <td>0.389095</td>
      <td>0.818174</td>
      <td>0.649888</td>
      <td>...</td>
      <td>0.304888</td>
      <td>0.452978</td>
      <td>0.585889</td>
      <td>0.506091</td>
      <td>0.389230</td>
      <td>0.255155</td>
      <td>0.134551</td>
      <td>0.382815</td>
      <td>0.774554</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>nafsher.wav</th>
      <td>0.682297</td>
      <td>0.631829</td>
      <td>0.716459</td>
      <td>0.073996</td>
      <td>0.357795</td>
      <td>1.000000</td>
      <td>0.315055</td>
      <td>0.356690</td>
      <td>0.546347</td>
      <td>0.391519</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.535127</td>
      <td>0.160494</td>
      <td>0.119893</td>
      <td>0.358936</td>
      <td>0.110411</td>
      <td>0.399293</td>
      <td>0.231206</td>
      <td>0.208479</td>
      <td>1.0</td>
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
      <th>nampher.wav</th>
      <td>0.585012</td>
      <td>0.750241</td>
      <td>0.386442</td>
      <td>0.257769</td>
      <td>0.420445</td>
      <td>0.681084</td>
      <td>0.688134</td>
      <td>0.141646</td>
      <td>0.351949</td>
      <td>0.829921</td>
      <td>...</td>
      <td>0.295594</td>
      <td>0.184981</td>
      <td>0.485324</td>
      <td>0.006996</td>
      <td>0.351461</td>
      <td>0.396676</td>
      <td>0.221318</td>
      <td>0.170630</td>
      <td>0.068843</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>namshuu.wav</th>
      <td>0.616046</td>
      <td>0.251532</td>
      <td>1.000000</td>
      <td>0.507461</td>
      <td>0.527331</td>
      <td>0.706880</td>
      <td>0.508846</td>
      <td>0.752297</td>
      <td>0.213969</td>
      <td>0.191885</td>
      <td>...</td>
      <td>0.324281</td>
      <td>0.278327</td>
      <td>0.363385</td>
      <td>0.509152</td>
      <td>0.350560</td>
      <td>0.130744</td>
      <td>0.375203</td>
      <td>0.294692</td>
      <td>0.364334</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>nafkgal.wav</th>
      <td>0.806668</td>
      <td>0.587196</td>
      <td>0.059843</td>
      <td>0.419891</td>
      <td>0.344732</td>
      <td>0.626202</td>
      <td>0.620310</td>
      <td>0.520342</td>
      <td>0.878859</td>
      <td>0.418037</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.420077</td>
      <td>0.139226</td>
      <td>0.232656</td>
      <td>0.178302</td>
      <td>0.076778</td>
      <td>0.136362</td>
      <td>0.154142</td>
      <td>0.253009</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>namshoo.wav</th>
      <td>0.617427</td>
      <td>0.491325</td>
      <td>0.671437</td>
      <td>0.230781</td>
      <td>0.427814</td>
      <td>0.335044</td>
      <td>0.538261</td>
      <td>0.870325</td>
      <td>0.650936</td>
      <td>0.223632</td>
      <td>...</td>
      <td>0.421972</td>
      <td>0.718514</td>
      <td>0.370733</td>
      <td>0.129998</td>
      <td>0.372822</td>
      <td>0.293832</td>
      <td>0.197089</td>
      <td>0.475113</td>
      <td>0.539821</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>namcrii.wav</th>
      <td>0.533502</td>
      <td>0.278901</td>
      <td>0.723767</td>
      <td>1.000000</td>
      <td>0.676326</td>
      <td>0.328397</td>
      <td>0.291755</td>
      <td>0.688971</td>
      <td>0.504892</td>
      <td>0.303077</td>
      <td>...</td>
      <td>0.266258</td>
      <td>0.252659</td>
      <td>0.203263</td>
      <td>0.156511</td>
      <td>0.112150</td>
      <td>0.135671</td>
      <td>0.056949</td>
      <td>0.148732</td>
      <td>0.083204</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
<p>72 rows × 27 columns</p>
</div>



## The MLP &  Cross-validation
Import the code of the backpropagation with momentum algorithm and the code of the cross-valdidation test


```python
import mlp_backprop_momentum as mlp
import k_fold_cross_validation as cv
```

### Util functions


```python
def split_dataset(dataset, n_parts=5):
    n_rows = dataset.shape[0]
    index_all = np.arange(n_rows)
    np.random.shuffle(index_all)
    parts = []
    current_start = 0
    for p in np.arange(n_parts):
        current_end = current_start + int(np.floor(n_rows / (n_parts-p)))
        parts.append(dataset[index_all[current_start:current_end],:])
        n_rows -= current_end - current_start
        current_start = current_end
    return parts
```


```python
def k_fold_cross_validation(mlp, dataset, K=5, learning_rate=0.01, momentum=0.7, epochs=100):
    MSE_train_mean = 0.0
    MSE_test_mean = 0.0

    parts = split_dataset(dataset, K)
    
    for k in np.arange(K):
        mlp.init_weights()
        
        training_parts = set(np.arange(K))
        training_parts.remove(k)
        dataset_train = np.concatenate([parts[i] for i in list(training_parts)])
        dataset_test = parts[k]

        input_data = dataset_train[:,0:nn.n_inputs]
        output_data = dataset_train[:,nn.n_inputs:(nn.n_inputs+nn.n_outputs)]
        input_data_test = dataset_test[:,0:nn.n_inputs]
        output_data_test = dataset_test[:,nn.n_inputs:(nn.n_inputs+nn.n_outputs)]
        
        MSE_train = mlp.fit((input_data, output_data),
                            learning_rate=learning_rate, momentum=momentum, epochs=epochs)
        temp, _ = mlp.compute_MSE((input_data, output_data))
        MSE_train_mean += temp
        temp, _ = mlp.compute_MSE((input_data_test, output_data_test))
        MSE_test_mean += temp

    return (MSE_train_mean / K, MSE_test_mean / K)
```

## Experiment

### Configuration of the MLP


```python
# Configuration

N_INITS = 10
EPOCHS = 65 # En gros, le nombre d'exécution
N_NEURONS = [2, 4, 8, 12, 16]
LEARNING_RATE = 0.002 # Pour la correction
MOMENTUM = 0.6 # On pousse la boule avec de l'élan

N_FEATURES = 26
ACTIVATION_FUNCTION = 'tanh'
```


```python
def runMLP(LEARNING_RATE, MOMENTUM):
    MSE = np.zeros((len(N_NEURONS), N_INITS, EPOCHS))

    for i_h, h in enumerate(N_NEURONS):                                     # looping over the number of hidden neurons
        print('Testing', h, 'neurons...')
        nn = mlp.MLP([N_FEATURES,h,1], ACTIVATION_FUNCTION)
        for i in np.arange(N_INITS):                                        # looping over the initializations
            nn.init_weights()

            MSE[i_h, i, :] = nn.fit((dataset[:,0:26], dataset[:,26:27]),
                                    learning_rate=LEARNING_RATE,
                                    momentum=MOMENTUM,
                                    epochs=EPOCHS)
        
    print('Done :D')
    
    return MSE
```


```python
def plotFigure(MSE):
    pl.figure(figsize=(15,4))
    p_count = 0
    for n in np.arange(MSE.shape[0]):
        pl.subplot(1, MSE.shape[0], n+1)
        for i in np.arange(MSE.shape[1]):
            pl.plot(MSE[n,i,:], c='b')
        pl.ylim(0,1)
        pl.xlabel('Epochs')
        pl.ylabel('MSE')
        pl.title(str(N_NEURONS[n]) + ' neurons')
        pl.grid()
    pl.tight_layout()
```


```python
MSE = runMLP(0.012, 0.5)
plotFigure(MSE)
```

    Testing 2 neurons...
    Testing 4 neurons...
    Testing 8 neurons...
    Testing 12 neurons...
    Testing 16 neurons...
    Done :D



![png](output_21_1.png)



```python
MSE = runMLP(0.011, 0.6)
plotFigure(MSE)
```

    Testing 2 neurons...
    Testing 4 neurons...
    Testing 8 neurons...
    Testing 12 neurons...
    Testing 16 neurons...
    Done :D



![png](output_22_1.png)



```python
MSE = runMLP(0.013, 0.6)
plotFigure(MSE)
```

    Testing 2 neurons...
    Testing 4 neurons...
    Testing 8 neurons...
    Testing 12 neurons...
    Testing 16 neurons...
    Done :D



![png](output_23_1.png)



```python
MSE = runMLP(0.014, 0.5)
plotFigure(MSE)
```

    Testing 2 neurons...
    Testing 4 neurons...
    Testing 8 neurons...
    Testing 12 neurons...
    Testing 16 neurons...
    Done :D



![png](output_24_1.png)



```python
MSE = runMLP(0.016, 0.5)
plotFigure(MSE)
```

    Testing 2 neurons...
    Testing 4 neurons...
    Testing 8 neurons...
    Testing 12 neurons...
    Testing 16 neurons...
    Done :D



![png](output_25_1.png)



```python
MSE = runMLP(0.02, 0.5)
plotFigure(MSE)
```

    Testing 2 neurons...
    Testing 4 neurons...
    Testing 8 neurons...
    Testing 12 neurons...
    Testing 16 neurons...
    Done :D



![png](output_26_1.png)



```python
MSE = runMLP(0.014, 0.6)
plotFigure(MSE)
```

    Testing 2 neurons...
    Testing 4 neurons...
    Testing 8 neurons...
    Testing 12 neurons...
    Testing 16 neurons...
    Done :D



![png](output_27_1.png)



```python
MSE = runMLP(0.015, 0.6)
plotFigure(MSE)
```

    Testing 2 neurons...
    Testing 4 neurons...
    Testing 8 neurons...
    Testing 12 neurons...
    Testing 16 neurons...
    Done :D



![png](output_28_1.png)



```python
MSE = runMLP(0.016, 0.6)
plotFigure(MSE)
```

    Testing 2 neurons...
    Testing 4 neurons...
    Testing 8 neurons...
    Testing 12 neurons...
    Testing 16 neurons...
    Done :D



![png](output_29_1.png)



```python
MSE = runMLP(0.02, 0.6)
plotFigure(MSE)
```

    Testing 2 neurons...
    Testing 4 neurons...
    Testing 8 neurons...
    Testing 12 neurons...
    Testing 16 neurons...
    Done :D



![png](output_30_1.png)


## Exploring the number of hidden neurons
Knowing that there are no significant improvements after 50 iterations, we can now further explore how the complexity of the model (number of hidden neurons) is linked to the generalization performance (test error). The following snippet allows you to explore both the number of epochs and the number of hidden neurons without restarting the training.


```python
EPOCHS = 80
K = 5
N_TESTS = 10
LEARNING_RATE = 0.012 # Pour la correction
MOMENTUM = 0.5 
N_NEURONS = [2, 4, 6, 8, 10, 15, 20, 25, 30]
```


```python
MSE_train = np.zeros((len(N_NEURONS), EPOCHS, N_TESTS))
MSE_test = np.zeros((len(N_NEURONS), EPOCHS, N_TESTS))

for i_h, h in enumerate(N_NEURONS):                                     # looping the number of hidden neurons
    print('Testing', h, 'neurons...')
    nn = mlp.MLP([N_FEATURES,h,1], ACTIVATION_FUNCTION)
    for i in np.arange(N_TESTS):                                        # looping the tests
        nn.init_weights()                                               # the network has to be reinitialized before each test
        temp1, temp2 = cv.k_fold_cross_validation_per_epoch(nn,         # notice that we do not use cv.k_fold_cross_validation
                                                            dataset,    # but cv.k_fold_cross_validation_per_epoch which
                                                            k=K,        # returns a value of error per each epoch
                                                            learning_rate=LEARNING_RATE,
                                                            momentum=MOMENTUM,
                                                            epochs=EPOCHS)
        # temp1 and temp2 are the training and test error. One value per epoch
        MSE_train[i_h, :, i] = temp1
        MSE_test[i_h, :, i] = temp2
        
print('Done')
```

    Testing 2 neurons...
    Testing 4 neurons...
    Testing 6 neurons...
    Testing 8 neurons...
    Testing 10 neurons...
    Testing 15 neurons...
    Testing 20 neurons...
    Testing 25 neurons...
    Testing 30 neurons...
    Done



```python
MSE_train_mean = np.mean(MSE_train, axis=2)
MSE_test_mean = np.mean(MSE_test, axis=2)
MSE_train_sd = np.std(MSE_train, axis=2)
MSE_test_sd = np.std(MSE_test, axis=2)

v_min = min(np.min(MSE_train_mean), np.min(MSE_test_mean))
v_max = max(np.max(MSE_train_mean), np.max(MSE_test_mean))

n_rows = int(np.ceil(len(N_NEURONS)/3.0))
pl.figure(figsize=(12,3*n_rows))
for i_n, n in enumerate(N_NEURONS):
    pl.subplot(n_rows, min(3, len(N_NEURONS)), i_n+1)
    pl.fill_between(np.arange(EPOCHS), MSE_train_mean[i_n,:], MSE_train_mean[i_n,:]+MSE_train_sd[i_n,:], facecolor='blue', alpha=0.5, label='Train')
    pl.fill_between(np.arange(EPOCHS), MSE_train_mean[i_n,:], MSE_train_mean[i_n,:]-MSE_train_sd[i_n,:], facecolor='blue', alpha=0.5)
    pl.fill_between(np.arange(EPOCHS), MSE_test_mean[i_n,:], MSE_test_mean[i_n,:]+MSE_test_sd[i_n,:], facecolor='red', alpha=0.5, label='Test')
    pl.fill_between(np.arange(EPOCHS), MSE_test_mean[i_n,:], MSE_test_mean[i_n,:]-MSE_test_sd[i_n,:], facecolor='red', alpha=0.5)
    pl.ylim(0.95*v_min,0.5*v_max)
    pl.ylabel('MSE')
    pl.xlabel('Number of epochs')
    pl.title(str(K)+'-fold CV with '+str(n)+' hidden neurons')
    pl.legend()
    pl.grid()
pl.tight_layout()
```


![png](output_34_0.png)



```python
pl.figure(figsize=(15,8))
pl.subplot(2,1,1)
pl.imshow(MSE_train_mean, vmin=np.min(MSE_train_mean), vmax=np.percentile(MSE_train_mean, 90), aspect=3, interpolation='nearest')
pl.yticks(np.arange(len(N_NEURONS)), N_NEURONS)
pl.xlabel('Epochs')
pl.ylabel('Number of hidden Neurons')
pl.title('Training')
pl.colorbar()
pl.subplot(2,1,2)
pl.imshow(MSE_test_mean, vmin=np.min(MSE_test_mean), vmax=np.percentile(MSE_test_mean, 90), aspect=3, interpolation='nearest')
pl.yticks(np.arange(len(N_NEURONS)), N_NEURONS)
pl.xlabel('Epochs')
pl.ylabel('Number of hidden Neurons')
pl.title('Test')
pl.colorbar()
pl.tight_layout()
```


![png](output_35_0.png)


## The final model

An artificial neural network with 6 neurons and 60 iterations of the backpropagation algorithm is enough to solve the problem.


```python
N_INITS = 10
K = 5

EPOCHS = 60 # En gros, le nombre d'exécution
N_NEURONS = [2, 4, 8, 12, 16]
LEARNING_RATE = 0.014 # Pour la correction
MOMENTUM = 0.6 # On pousse la boule avec de l'élan


N_FEATURES = 26
ACTIVATION_FUNCTION = 'tanh'

h=10

nn = mlp.MLP([N_FEATURES,h,1], ACTIVATION_FUNCTION)
```


```python
MSE_train, MSE_test, conf_mat = cv.k_fold_cross_validation(nn,
                                                          dataset,
                                                          k=K,
                                                          learning_rate=LEARNING_RATE,
                                                          momentum=MOMENTUM,
                                                          epochs=EPOCHS,
                                                          threshold=0.0)
```


```python
print('MSE training: ', MSE_train)
print('MSE test: ', MSE_test)
print('Confusion matrix:')
print(conf_mat)
```

    MSE training:  0.005958495874789911
    MSE test:  0.1429870891720953
    Confusion matrix:
    [[36.  0.]
     [ 3. 33.]]


### Confusion matrix:

* True positif 36.
* False negative 0.
* False positive 3.
* True negative 33.

Over 72 files read


```python

```
