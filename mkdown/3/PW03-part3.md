

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
            if file.name.startswith(pattern + 'af') :
                sound_class = [1,-1,-1]
            elif file.name.startswith(pattern + 'am'):
                sound_class = [-1,1,-1]
            else:
                sound_class = [-1,-1,1]
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

sounds = readFiles('n')

# Let's test if the read worked...
plotSound(sounds[0][1], sounds[0][2])

# Here we print the filename and the class. 
for sound in sounds[4:8]:
    print(sound[2], sound[3])
```

    180  elements have been read
    nk7brii.wav [-1, -1, 1]
    nafshoo.wav [1, -1, -1]
    namshul.wav [-1, 1, -1]
    nafkguh.wav [1, -1, -1]



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
        list(map(lambda s: np.append( np.median(s, axis=0), np.append( np.std(s, axis=0), np.append(np.quantile(s,q=0.0, axis=0), np.quantile(s,q=1.0, axis=0)))  ), array)),
    (-1,1))

normalized_ceps_data = feature(ceps_data)
```


```python
print(len(normalized_ceps_data[0]))
```

    52



```python
dataset = np.hstack((np.array(normalized_ceps_data), np.array(ceps_class)))

```


```python
import pandas as pd

dataset = np.hstack((np.array(normalized_ceps_data), np.array(ceps_class)))

cols = np.array(
    list(map(lambda i: str(i), np.arange(55))) )

```


```python

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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nk3ajoo.wav</th>
      <td>0.311275</td>
      <td>0.489519</td>
      <td>0.732699</td>
      <td>0.476710</td>
      <td>0.388588</td>
      <td>0.787911</td>
      <td>0.576236</td>
      <td>0.284571</td>
      <td>0.502957</td>
      <td>0.579228</td>
      <td>...</td>
      <td>0.519089</td>
      <td>0.441543</td>
      <td>0.449743</td>
      <td>0.595171</td>
      <td>0.877510</td>
      <td>0.580715</td>
      <td>0.497641</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>nk3ajul.wav</th>
      <td>0.198954</td>
      <td>0.463016</td>
      <td>0.480158</td>
      <td>0.348438</td>
      <td>0.425590</td>
      <td>0.608561</td>
      <td>0.333217</td>
      <td>0.543064</td>
      <td>0.762421</td>
      <td>0.829486</td>
      <td>...</td>
      <td>0.332790</td>
      <td>0.569061</td>
      <td>0.675923</td>
      <td>0.735897</td>
      <td>0.194699</td>
      <td>0.255781</td>
      <td>0.542808</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>nk7bsee.wav</th>
      <td>0.700905</td>
      <td>0.463180</td>
      <td>0.481801</td>
      <td>0.745267</td>
      <td>0.285354</td>
      <td>0.237303</td>
      <td>0.620038</td>
      <td>0.498973</td>
      <td>0.579906</td>
      <td>0.423829</td>
      <td>...</td>
      <td>0.433794</td>
      <td>0.621704</td>
      <td>0.406403</td>
      <td>0.337185</td>
      <td>0.575399</td>
      <td>0.544515</td>
      <td>0.202571</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>nafshuh.wav</th>
      <td>0.618860</td>
      <td>0.538571</td>
      <td>0.306168</td>
      <td>0.250803</td>
      <td>0.570391</td>
      <td>0.469343</td>
      <td>0.511100</td>
      <td>0.515440</td>
      <td>0.639038</td>
      <td>0.554415</td>
      <td>...</td>
      <td>0.382558</td>
      <td>0.484425</td>
      <td>0.523992</td>
      <td>0.506293</td>
      <td>0.611827</td>
      <td>0.034775</td>
      <td>0.352318</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>nk7brii.wav</th>
      <td>0.576182</td>
      <td>0.270667</td>
      <td>0.945725</td>
      <td>0.881523</td>
      <td>0.561401</td>
      <td>0.261339</td>
      <td>0.929242</td>
      <td>0.327456</td>
      <td>0.511886</td>
      <td>0.309577</td>
      <td>...</td>
      <td>0.894715</td>
      <td>0.239189</td>
      <td>0.349558</td>
      <td>0.241640</td>
      <td>0.521740</td>
      <td>0.330494</td>
      <td>0.000000</td>
      <td>-1.0</td>
      <td>-1.0</td>
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
      <th>nk5mnaa.wav</th>
      <td>0.709989</td>
      <td>0.640708</td>
      <td>0.019445</td>
      <td>0.326556</td>
      <td>0.314349</td>
      <td>0.871987</td>
      <td>0.709652</td>
      <td>0.389138</td>
      <td>0.655766</td>
      <td>0.462395</td>
      <td>...</td>
      <td>0.801495</td>
      <td>0.476499</td>
      <td>0.577411</td>
      <td>0.406561</td>
      <td>0.377532</td>
      <td>0.196912</td>
      <td>0.404658</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>nafkgal.wav</th>
      <td>0.565999</td>
      <td>0.628901</td>
      <td>0.152694</td>
      <td>0.419891</td>
      <td>0.486673</td>
      <td>0.591761</td>
      <td>0.719308</td>
      <td>0.556964</td>
      <td>0.732906</td>
      <td>0.418037</td>
      <td>...</td>
      <td>0.602822</td>
      <td>0.466917</td>
      <td>0.656045</td>
      <td>0.368225</td>
      <td>0.254993</td>
      <td>0.272106</td>
      <td>0.163714</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>namshoo.wav</th>
      <td>0.433218</td>
      <td>0.542716</td>
      <td>0.599544</td>
      <td>0.230781</td>
      <td>0.551758</td>
      <td>0.407425</td>
      <td>0.658652</td>
      <td>0.880226</td>
      <td>0.542834</td>
      <td>0.223632</td>
      <td>...</td>
      <td>0.604000</td>
      <td>0.888862</td>
      <td>0.422476</td>
      <td>0.255664</td>
      <td>0.642024</td>
      <td>0.463152</td>
      <td>0.793962</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>nk3ajaa.wav</th>
      <td>0.455305</td>
      <td>0.391449</td>
      <td>0.325590</td>
      <td>0.363012</td>
      <td>0.443103</td>
      <td>0.900640</td>
      <td>0.724805</td>
      <td>0.312441</td>
      <td>0.482140</td>
      <td>0.649490</td>
      <td>...</td>
      <td>0.655133</td>
      <td>0.348787</td>
      <td>0.360299</td>
      <td>0.619706</td>
      <td>0.254810</td>
      <td>0.093693</td>
      <td>0.776951</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>namcrii.wav</th>
      <td>0.374332</td>
      <td>0.351753</td>
      <td>0.637777</td>
      <td>1.000000</td>
      <td>0.746439</td>
      <td>0.403217</td>
      <td>0.476418</td>
      <td>0.712718</td>
      <td>0.421044</td>
      <td>0.303077</td>
      <td>...</td>
      <td>0.373397</td>
      <td>0.728412</td>
      <td>0.259180</td>
      <td>0.272288</td>
      <td>0.285336</td>
      <td>0.390784</td>
      <td>0.444877</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
<p>180 rows × 55 columns</p>
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
EPOCHS = 200 # En gros, le nombre d'exécution
N_NEURONS = [4, 8, 12, 16, 20]
LEARNING_RATE = 0.01 # Pour la correction
MOMENTUM = 0.5# On pousse la boule avec de l'élan

N_FEATURES = 52
ACTIVATION_FUNCTION = 'tanh'

OUT= 3
```


```python
def runMLP(LEARNING_RATE, MOMENTUM):
    MSE = np.zeros((len(N_NEURONS), N_INITS, EPOCHS))

    for i_h, h in enumerate(N_NEURONS):                                     # looping over the number of hidden neurons
        print('Testing', h, 'neurons...')
        nn = mlp.MLP([N_FEATURES,h,OUT], ACTIVATION_FUNCTION)
        for i in np.arange(N_INITS):                                        # looping over the initializations
            nn.init_weights()

            MSE[i_h, i, :] = nn.fit((dataset[:,0:52], dataset[:,52:53]),
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
MSE = runMLP(0.01, 0.5)
plotFigure(MSE)
```

    Testing 4 neurons...
    Testing 8 neurons...
    Testing 12 neurons...
    Testing 16 neurons...
    Testing 20 neurons...
    Done :D



![png](output_23_1.png)


## Exploring the number of hidden neurons
Knowing that there are no significant improvements after 50 iterations, we can now further explore how the complexity of the model (number of hidden neurons) is linked to the generalization performance (test error). The following snippet allows you to explore both the number of epochs and the number of hidden neurons without restarting the training.


```python
EPOCHS = 200
K = 5
N_TESTS = 10
LEARNING_RATE = 0.01 # Pour la correction
MOMENTUM = 0.5 
N_NEURONS = [4, 6, 8, 10, 15, 20, 25, 30, 35]

OUT=3
```


```python
MSE_train = np.zeros((len(N_NEURONS), EPOCHS, N_TESTS))
MSE_test = np.zeros((len(N_NEURONS), EPOCHS, N_TESTS))

for i_h, h in enumerate(N_NEURONS):                                     # looping the number of hidden neurons
    print('Testing', h, 'neurons...')
    nn = mlp.MLP([N_FEATURES,h,OUT], ACTIVATION_FUNCTION)
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

    Testing 4 neurons...
    Testing 6 neurons...
    Testing 8 neurons...
    Testing 10 neurons...
    Testing 15 neurons...
    Testing 20 neurons...
    Testing 25 neurons...
    Testing 30 neurons...
    Testing 35 neurons...
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


![png](output_27_0.png)



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


![png](output_28_0.png)


## The final model

An artificial neural network with 6 neurons and 60 iterations of the backpropagation algorithm is enough to solve the problem.


```python
N_INITS = 10
EPOCHS = 200 # En gros, le nombre d'exécution
N_NEURONS = [2, 4, 8, 12, 16]
LEARNING_RATE = 0.01 # Pour la correction
MOMENTUM = 0.5 # On pousse la boule avec de l'élan


N_FEATURES = 52
ACTIVATION_FUNCTION = 'tanh'

OUT=3

h=30

nn = mlp.MLP([N_FEATURES,h,OUT], ACTIVATION_FUNCTION)
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

    MSE training:  0.0013294478379767363
    MSE test:  0.21362182575080285
    Confusion matrix:
    [[ 27.   1.   5.]
     [  4.  34.   0.]
     [  8.   1. 102.]]


### Confusion matrix:

TODO

Over 180 files read


```python

```
