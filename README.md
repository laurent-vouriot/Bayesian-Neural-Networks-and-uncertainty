# Uncertainty with Bayesian Neural Networks

**Bayesian Neural Networks** (BNN) can be understood as a combination of neural networks and bayesian inference. In traditional deep learning, weights are fixed values (initially random), that we iteratively update via gradient descent. BNN on the other hand learn __distribution parameters over weights__, these distribution parameters are learned using bayesian inference. This approaches allows, among others, to measure uncertainty among predictions.

# Bayesian inference 

In bayesian inference we try to compute $p(w | D_{train})$, the conditional distribution of the weights given the data, aka the *posterior distribution*, rather than static parameters values. 

### Exact Bayesian Inference

Thanks to Bayes' rule we can compute the posterior : 

```math
    p(w | D) & = \frac{p(D | w)p(w)}{p(D)}\\
             & = \frac{p(D | w)p(w)}{\int_w' p(D | w')p(w')} \\
``` 

With $\hat{y}$ the predicted output as a function of the input $x$, The posterior distribution over the weights allows us to compute the __predictive distribution__ : 

\begin{align*}
    p(\hat{y}(x) | D) = \int_w p(\hat{y}(x) | w) p(w | D) dw = \mathbb{E}_{p(w | D)} [p(\hat{y}(x) | w)]
\end{align*}

which can be useful to describe the epistemic uncertainty of our model. We will come back later on the uncertainty.

Computing the posterior in this way can be called *exact inference*, and needs the *prior* $p(w)$ and the *likelihood* $p(D | w)$ of the data. Unfortunately in the expression of the posterior and the predictive distribution we need to integrate over the weight space which can be intractable, in order to adresses theses issues we use a set of tools that allow us to do approximate inference. There is two family of approximate inference methods : sampling and variational, we will focus on variational methods.

### Variational inference

The spirit of variational inference is, when facing an intractable posterior $p(w | D)$, to surrogate it with a parametrized distribution $q_{\phi}(w)$, namely the *approximate posterior*. This surrogate distribution will be optimised (by tuning its parameters $\phi$) in order to be as close as possible to the original posterior. In practice the choice of the approximate distribution can be seen as an hyperparameter. Two main questions arise from this : 

* **How to check if the surrogate is close to the true posterior ?** 
* **How to maximize the similarity of our distributions ?**


### Measure of similarity : 

The Kullback-Liebler divergence is a metric that allows us to mesure the similiarity between two distribution. it is defined by the expectation of the log ratio between the two distributions : 

\begin{align*}
    D_{KL}(P \| Q) = \mathbb{E} \left[log \frac{P}{Q}\right]
\end{align*}

in our case : 

\begin{align*}
    D_{KL} (q_{\phi}(w) \| p(w | D)) & = \mathbb{E}_{q_{\phi}(w)}\left[log \frac{q_{\phi}(w)}{p(w |D)} \right] \\
                                    & = \int_w q_\phi (w) log \frac{q_\phi (w)}{p(w |D)}
\end{align*}


The KL divergence is a non-negative measure of similarity, that is 0 for identical distributions
... 

### Derive a tractable optimization problem : 

We have an intractable posterior $p(w | W)$, a surrogate distribution $q_{\phi}(w)$, a way to measure their similarity  $D_{KL} (q_{\phi}(w) \| p(w | D))$, now we need find a way to minimize their dissimilarity. To do this we will formulate it as an optimization problem. 

Directly minimizing  $D_{KL} (q_{\phi}(w) \| p(w | D))$ is 
difficult as $p(w | D)$ is still intractable. To bypass this we will derive a related quantity, equal to the KL divergence plus a constant, that will be our new objective.

\begin{align*}
    D_{KL} (q_{\phi}(w) \| p(w | D)) & = \mathbb{E}_{q_{\phi}(w)}\left[log \frac{q_{\phi}(w)}{p(w |D)} \right] \\
                                     & = \mathbb{E}_{q_{\phi}(w)}\left[log(q_{\phi}(w)) - log (p(w |D)) \right] \\
                                     & = \mathbb{E}_{q_{\phi}(w)}\left[log(q_{\phi}(w))\right] - \mathbb{E}_{q_{\phi}(w)}\left[log (p(D, w)) - log (p(D)) \right] \\        
                                     & = \mathbb{E}_{q_{\phi}(w)}\left[log(q_{\phi}(w))\right] - \mathbb{E}_{q_{\phi}(w)}\left[log (p(D, w))\right] + log p(D)\\                   
\end{align*}

thus 

\begin{align*} 
    log p(D) \geq \mathbb{E}_{q_{\phi}(w)}\left[log q_{\phi}(w) - log p(w, D)\right] \;\;\;\;\;\;\;\; \;\;\;\;\;\;\;\;\; [\small{\text{as}\;\;\; D_{KL}(q \| p) \geq 0}]
\end{align*}

## Make it work with neural networks.
### bayes by backprop

Once our optimization problem formulated we want to be able to optimize it, and in the context of neural networks, using gradient descent. Hence we need a proper algortihm adapted for backpropagation. To do so we will use bayes by backpropagation, a backpropagation compatible algorithm for learning a probability distribution on the weights of a neural network. \cite{Blundell et al.}. 

In bayesian neural networks, it is challenging to differentiate random nodes. To overcome this we will use a tool called the reparametrization trick. Intuitively, the reparametrization trick allows to flow gradient through random nodes by moving the randomness outside of the node, rending it deterministic. As such, backprop can be applied to our variational parameter $\phi$. This gives us the advantage of maintaining a training loop analogous to that of a standard neural network, which is convenient as it permits the use of traditional optimizers, such as Adam.

More formally, considering a gaussian posterior distribution $q_{\phi}(z|x)$, parametrized by $\mu$ and $\sigma$. say we want to minimize a loss function $E_{q_{\phi}(z | x)} [f(z))]$. 
Directly optimizing this expectation with respect to $\phi$ can be difficult due to its randomness in z complicating the computation of gradients. To overcome this we introduce an auxilary variable $\epsilon$ drawn from a distribution $p(\epsilon)$ independent from $\phi$ allowing us to express z as a deterministic function of $\phi$ and $\sigma$ such as $z = g_{\phi}(\epsilon, x) = \mu(x; \mu) + \sigma(w, \phi) \otimes \epsilon$. Typically $\epsilon$ is drawn from Gaussian ditribution $\epsilon \sim \mathcal{N}(0, I)$. Thus the expectation turns into : 

\begin{align*}
    \nabla_{\phi} E_{p(\epsilon)}[f(g_{\phi}(\epsilon, x))] &= E_{p(\epsilon)}[ \nabla_{\phi} f(g_{\phi}(\epsilon, x))] 
\end{align*} 

# Import, data and utils functions 


```python
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
```

    /tmp/ipykernel_248090/2317591137.py:3: DeprecationWarning: 
    Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),
    (to allow more performant data types, such as the Arrow string type, and better interoperability with other libraries)
    but was not found to be installed on your system.
    If this would cause problems for you,
    please provide us feedback at https://github.com/pandas-dev/pandas/issues/54466
            
      import pandas as pd
    2024-02-21 17:37:58.109927: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-02-21 17:37:58.110015: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-02-21 17:37:58.114070: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-02-21 17:37:59.802291: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


# Dataset

- Basic non-linear dataset for a classification problem


```python
X, y = make_moons(n_samples=20000, noise=0.3)
y = y.reshape(-1,1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_test = tf.cast(X_test, tf.float32)
X_train = tf.cast(X_train, tf.float32)
y_test = tf.cast(y_test, tf.float32)
y_train = tf.cast(y_train, tf.float32)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
ax1.scatter(X_train[:,0], X_train[:,1], c=y_train)
ax1.set_title('train set')
ax2.scatter(X_test[:,0], X_test[:,1], c=y_test)
ax2.set_title('test set')

plt.show()
```


    
![png](output_8_0.png)
    


# Baseline model
traditional deep net


```python
def create_network(units=100, activation='relu', lr=0.01) : 
    
    inputs = keras.Input(shape=(X_train.shape[1],))
    
    hidden1 = Dense(units, activation=activation)(inputs)
    hidden1 = BatchNormalization()(hidden1)
    
    hidden2 = Dense(units, activation=activation)(hidden1)
    hidden2 = BatchNormalization()(hidden2)
    
    hidden3 = Dense(units, activation=activation)(hidden2)
    hidden3 = BatchNormalization()(hidden3)
    
    outputs = Dense(1, activation='sigmoid')(hidden3)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='baseline')

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['Accuracy'])
    model.summary()
    return model
```


```python
baseline = create_network()
baseline.fit(X_train, y_train, batch_size=32, verbose=0, epochs=20) 
```

    Model: "baseline"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 2)]               0         
                                                                     
     dense (Dense)               (None, 100)               300       
                                                                     
     batch_normalization (Batch  (None, 100)               400       
     Normalization)                                                  
                                                                     
     dense_1 (Dense)             (None, 100)               10100     
                                                                     
     batch_normalization_1 (Bat  (None, 100)               400       
     chNormalization)                                                
                                                                     
     dense_2 (Dense)             (None, 100)               10100     
                                                                     
     batch_normalization_2 (Bat  (None, 100)               400       
     chNormalization)                                                
                                                                     
     dense_3 (Dense)             (None, 1)                 101       
                                                                     
    =================================================================
    Total params: 21801 (85.16 KB)
    Trainable params: 21201 (82.82 KB)
    Non-trainable params: 600 (2.34 KB)
    _________________________________________________________________





    <keras.src.callbacks.History at 0x7f77c43b8910>



- Evaluate the model


```python
y_pred = np.asarray(baseline(X_test))

print('accuracy : ', accuracy_score(y_test, y_pred.round()))

fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(8, 14))
ax1.scatter(X_test[:,0], X_test[:,1], c=y_test)
ax1.set_title('original dataset')

ax2.scatter(X_test[:,0], X_test[:,1], c=y_pred.round())
ax2.set_title('predictions rounded')

ax3.scatter(X_test[:,0], X_test[:,1], c=y_pred)
ax3.set_title('predictions')

```

    accuracy :  0.91675





    Text(0.5, 1.0, 'predictions')




    
![png](output_13_2.png)
    


* ### Aleatoric uncertainty (aka statistical) : 
Aleatoric uncertainty refers to the notion of randomness, That is the uncertainty rising from the datahimself, where the model cannot act.

To allow a model to capture the aleatoric uncertainty we will output not a point estimate as usual but a probability distribution. In the case of binary classification the output will be a bernoulli distribution. 

$
    \mathbb{P}(X = x) = p^x(1 - p)^{1-x},\;\; x \in [0, 1]
$

As the output is no more a single estimate we need to change the loss function, we will use the **negative loglieklihood**, in order to get how likely it is to encounter targets in our data from the estimated distribution of the model.


```python
def nll(y, y_pred): 
  return -y_pred.log_prob(y)
```

In practice the only modification to our baseline model is the ouput layer. In our case we will use a *IndependentBernoulli* layer.


```python
def create_aleatoric_model(units=100, activation='relu', lr=0.001):

  inputs = keras.Input(shape=(X_train.shape[1],))

  hidden1 = Dense(units, activation=activation)(inputs)
  hidden1 = BatchNormalization()(hidden1)
  hidden1 = Dropout(0.1)(hidden1)

  hidden2 = Dense(units, activation=activation)(hidden1)
  hidden2 = BatchNormalization()(hidden2)
  hidden2 = Dropout(0.1)(hidden2)
  
  hidden3 = Dense(units, activation=activation)(hidden2)
  hidden3 = BatchNormalization()(hidden3)
  hidden3 = Dropout(0.1)(hidden3)

  # output Bernoulli distribution ! 
  outputs = Dense(tfp.layers.IndependentBernoulli.params_size(1))(hidden3)  
  outputs = tfp.layers.IndependentBernoulli(1)(outputs)

  model = keras.Model(inputs=inputs, outputs=outputs, name='aleatoric_BNN')

  opt = keras.optimizers.Adam(learning_rate=lr)
  model.compile(loss=nll, optimizer=opt, metrics=['Accuracy'])
  model.summary()
  return model
```


```python
aleatoric_model = create_aleatoric_model()
aleatoric_model.fit(X_train, y_train, batch_size=32, verbose=0, epochs=20) 
```

    Model: "aleatoric_BNN"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_7 (InputLayer)        [(None, 2)]               0         
                                                                     
     dense (Dense)               (None, 100)               300       
                                                                     
     batch_normalization (Batch  (None, 100)               400       
     Normalization)                                                  
                                                                     
     dropout (Dropout)           (None, 100)               0         
                                                                     
     dense_1 (Dense)             (None, 100)               10100     
                                                                     
     batch_normalization_1 (Bat  (None, 100)               400       
     chNormalization)                                                
                                                                     
     dropout_1 (Dropout)         (None, 100)               0         
                                                                     
     dense_2 (Dense)             (None, 100)               10100     
                                                                     
     batch_normalization_2 (Bat  (None, 100)               400       
     chNormalization)                                                
                                                                     
     dropout_2 (Dropout)         (None, 100)               0         
                                                                     
     dense_3 (Dense)             (None, 1)                 101       
                                                                     
     independent_bernoulli (Ind  ((None, 1),               0         
     ependentBernoulli)           (None, 1))                         
                                                                     
    =================================================================
    Total params: 21801 (85.16 KB)
    Trainable params: 21201 (82.82 KB)
    Non-trainable params: 600 (2.34 KB)
    _________________________________________________________________





    <keras.src.callbacks.History at 0x1fb4170a380>



**The standard deviation of the outputed distribution will represent te aleatoric uncertainty associated with the dataset :**


```python
pred_distribution = aleatoric_model(X_test)
pred_mean = pred_distribution.mean().numpy()
pred_stdv = pred_distribution.stddev().numpy()

print('accuracy : ', accuracy_score(y_test, pred_mean.round()))

fig, axs = plt.subplots(4, 1, figsize=(8,14))
axs[0].scatter(X_test[:,0], X_test[:,1], c=y_test)
axs[0].set_title('test set')
axs[1].scatter(X_test[:,0], X_test[:,1], c=pred_mean.round())
axs[1].set_title('predictions on the test set')
axs[2].scatter(X_test[:,0], X_test[:,1], c=pred_mean)
axs[2].set_title('predictions (color gradient on prediction mean i.e sharpness)')
points2 = axs[3].scatter(X_test[:,0], X_test[:,1], c=pred_stdv, cmap='plasma')
axs[3].set_title('standard deviation of the predicted distribution, the closer to yellow the higher aleatoric uncertainty is')

fig.colorbar(points2)

plt.tight_layout()
```

    accuracy :  0.907



    
![png](output_20_1.png)
    


* ### Epistemic uncertainty (aka systematic) : 
refers to uncertainty caused by a lack of knowledge. It can be seen as the uncertainty of the model itself on his predictions. The epistemic uncertainty is the ability of the model saying 'I don't know'.

In practice, we will use tfp ```DenseVariational``` layers to do variational inference. Recall variational inference relies on Bayesian inference. Thus we need to define a prior $p(w)$ and a posterior distribution $p(w|D)$. The choice of these distributions can be seen as hyperparameters. Prior are imortant as they have a regularization aspect.  Here are listed a few options available with tfp :  

### Priors

* Normal (or Gaussian) prior
  
A basic reasonable approach is a normal prior distribution.  The normal prior acts as a form of regularization, penalizing large weights by making them less probable a priori. This is analogous to L2 regularization (or Ridge regression) in frequentist statistics, where the penalty on the weights is proportional to their squared magnitude.

The normal prior promotes smoothness and smaller magnitudes in the weights but does not explicitly push them to zero. This can lead to models that are less sparse but potentially more stable, as small changes in the input data are less likely to result in large changes in the output.


```python
def normal_prior(kernel_size, bias_size, dtype=None): 
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
    [
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=tf.zeros(n),
                       scale=2*tf.ones(n)), 
          reinterpreted_batch_ndims=1)),
    ])
    return prior_model
```

* Multivariate Normal prior 

As opposed to independent normal priors, where each connexion has its own independent distribution, a mutlivariate normal distribution assigns weights to each neurones of a layer and this thus useful to capture correlation between weights.


```python
def multivariate_normal_prior(kernel_size, bias_size, dtype=None): 
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model
```

* Laplace prior

the Laplace prior induces sparsity in a more direct manner. It is analogous to L1 regularization (or the Lasso method) in frequentist statistics.

The Laplace prior encourages sparsity in the model parameters by having a sharp peak at zero and heavy tails. This means that it pushes coefficients towards exactly zero, effectively performing variable selection or feature elimination, which can be beneficial in models with many irrelevant features.


```python
def laplace_prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size # num of params
    return Sequential([
       tfp.layers.DistributionLambda(
           lambda t: tfd.Independent(tfd.Laplace(loc = tf.zeros(n), 
                                                 scale=tf.ones(n)),
                                     reinterpreted_batch_ndims=1))               
  ])
```

* Horseshoe prior

The Horseshoe prior introduces a form of regularization that is more adaptive compared to traditional techniques like L1 or L2 regularization. It can shrink less relevant weights more aggressively towards zero while allowing important weights to remain large, potentially leading to a more sparse and efficient network. It can be used aswell to prune the network by shifting useless connexions towards 0.


```python
def horseshoe_prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size # num of params
    return Sequential([
       tfp.layers.DistributionLambda(
           lambda t: tfd.Independent(tfd.Horseshoe(scale = tf.zeros(n),
                                                   reinterpreted_batch_ndims=1)))               
  ])
```

### Posteriors

In variational inference, using a independent normal distribution is a common practice as it ensures mathematical conveniance. They also enable the use of the reparametrization trick, allowing gradient estimations.


```python
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=tf.nn.softplus(t[..., n:])),
          reinterpreted_batch_ndims=1)),
    ])
```


```python
def posterior(kernel_size, bias_size, dtype=None): 
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    
    return posterior_model
```


```python
def posterior(kernel_size, bias_size, dtype=None):
  n = kernel_size + bias_size
  posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.IndependentNormal.params_size(n), dtype=dtype
            ),
            tfp.layers.IndependentNormal(n),
        ]
    )
    
  return posterior_model
```

## Epistemic model


```python
def create_epistemic_model(prior, posterior):
  inputs = keras.Input(shape=(X_train.shape[1],))

  x = tfp.layers.DenseVariational(units=200, 
                                  make_prior_fn=prior,
                                  make_posterior_fn=posterior,
                                  kl_weight = 1 / X_train.shape[0],
                                  activation='relu')(inputs)
  
  x = tfp.layers.DenseVariational(units=200, 
                                  make_prior_fn=prior,
                                  make_posterior_fn=posterior,
                                  kl_weight = 1 / X_train.shape[0],
                                  activation='relu')(x)
                        
  outputs = tfp.layers.DenseVariational(units=1, 
                                  make_prior_fn=prior,
                                  make_posterior_fn=posterior,
                                  kl_weight = 1 / X_train.shape[0],
                                  activation='sigmoid')(x)

  model = keras.Model(inputs=inputs, outputs=outputs, name='epistemic_BNN')

  opt = keras.optimizers.Adam(learning_rate=0.001)

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                optimizer=opt, metrics=['Accuracy'])
  model.summary()

  return model
```


```python
epistemic_model = create_epistemic_model(multivariate_normal_prior, posterior)
epistemic_model.fit(X_train, y_train, batch_size=32, verbose=1, epochs=500) 
```

    Model: "epistemic_BNN"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 2)]               0         
                                                                     
     dense_variational (DenseVa  (None, 200)               1200      
     riational)                                                      
                                                                     
     dense_variational_1 (Dense  (None, 200)               80400     
     Variational)                                                    
                                                                     
     dense_variational_2 (Dense  (None, 1)                 402       
     Variational)                                                    
                                                                     
    =================================================================
    Total params: 82002 (320.32 KB)
    Trainable params: 82002 (320.32 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    Epoch 1/500
    500/500 [==============================] - 5s 5ms/step - loss: 18.7105 - Accuracy: 0.4989
    Epoch 2/500
    500/500 [==============================] - 2s 4ms/step - loss: 14.4992 - Accuracy: 0.5025
    Epoch 3/500
    500/500 [==============================] - 2s 4ms/step - loss: 12.6232 - Accuracy: 0.4947
    Epoch 4/500
    500/500 [==============================] - 2s 4ms/step - loss: 9.5676 - Accuracy: 0.5119
    Epoch 5/500
    500/500 [==============================] - 2s 4ms/step - loss: 8.1739 - Accuracy: 0.5076
    Epoch 6/500
    500/500 [==============================] - 2s 4ms/step - loss: 7.0487 - Accuracy: 0.5109
    Epoch 7/500
    500/500 [==============================] - 2s 4ms/step - loss: 6.1731 - Accuracy: 0.4929
    Epoch 8/500
    500/500 [==============================] - 2s 4ms/step - loss: 5.1447 - Accuracy: 0.4928
    Epoch 9/500
    500/500 [==============================] - 2s 4ms/step - loss: 4.1803 - Accuracy: 0.5042
    Epoch 10/500
    500/500 [==============================] - 2s 4ms/step - loss: 3.4783 - Accuracy: 0.4948
    Epoch 11/500
    500/500 [==============================] - 2s 4ms/step - loss: 3.1385 - Accuracy: 0.4911
    Epoch 12/500
    500/500 [==============================] - 2s 4ms/step - loss: 2.5840 - Accuracy: 0.4989
    Epoch 13/500
    500/500 [==============================] - 2s 4ms/step - loss: 2.3383 - Accuracy: 0.5051
    Epoch 14/500
    500/500 [==============================] - 2s 4ms/step - loss: 2.0279 - Accuracy: 0.4939
    Epoch 15/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.8296 - Accuracy: 0.4937
    Epoch 16/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.5632 - Accuracy: 0.4912
    Epoch 17/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.4694 - Accuracy: 0.5053
    Epoch 18/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.4061 - Accuracy: 0.5025
    Epoch 19/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.3352 - Accuracy: 0.5036
    Epoch 20/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.2654 - Accuracy: 0.4995
    Epoch 21/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.2181 - Accuracy: 0.5051
    Epoch 22/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.1994 - Accuracy: 0.4928
    Epoch 23/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.1972 - Accuracy: 0.5016
    Epoch 24/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.1623 - Accuracy: 0.4949
    Epoch 25/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.1361 - Accuracy: 0.5077
    Epoch 26/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.1385 - Accuracy: 0.4844
    Epoch 27/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.1204 - Accuracy: 0.5008
    Epoch 28/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.1093 - Accuracy: 0.4970
    Epoch 29/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.1083 - Accuracy: 0.4981
    Epoch 30/500
    500/500 [==============================] - 2s 5ms/step - loss: 1.0776 - Accuracy: 0.5058
    Epoch 31/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.0834 - Accuracy: 0.4983
    Epoch 32/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.0743 - Accuracy: 0.4959
    Epoch 33/500
    500/500 [==============================] - 2s 5ms/step - loss: 1.0610 - Accuracy: 0.5076
    Epoch 34/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.0773 - Accuracy: 0.5029
    Epoch 35/500
    500/500 [==============================] - 2s 5ms/step - loss: 1.0484 - Accuracy: 0.4965
    Epoch 36/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.0426 - Accuracy: 0.4891
    Epoch 37/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.0334 - Accuracy: 0.5023
    Epoch 38/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.0126 - Accuracy: 0.5017
    Epoch 39/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9985 - Accuracy: 0.4985
    Epoch 40/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9984 - Accuracy: 0.5046
    Epoch 41/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9963 - Accuracy: 0.4964
    Epoch 42/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9951 - Accuracy: 0.5034
    Epoch 43/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9768 - Accuracy: 0.4987
    Epoch 44/500
    500/500 [==============================] - 2s 4ms/step - loss: 1.0215 - Accuracy: 0.4972
    Epoch 45/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9874 - Accuracy: 0.5067
    Epoch 46/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9671 - Accuracy: 0.4980
    Epoch 47/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9681 - Accuracy: 0.5029
    Epoch 48/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9576 - Accuracy: 0.5053
    Epoch 49/500
    500/500 [==============================] - 2s 5ms/step - loss: 0.9524 - Accuracy: 0.4985
    Epoch 50/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9444 - Accuracy: 0.5027
    Epoch 51/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9423 - Accuracy: 0.5031
    Epoch 52/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9716 - Accuracy: 0.5008
    Epoch 53/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9465 - Accuracy: 0.5004
    Epoch 54/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9301 - Accuracy: 0.5001
    Epoch 55/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9414 - Accuracy: 0.4979
    Epoch 56/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9250 - Accuracy: 0.5046
    Epoch 57/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9333 - Accuracy: 0.4993
    Epoch 58/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9260 - Accuracy: 0.4981
    Epoch 59/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9226 - Accuracy: 0.5014
    Epoch 60/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9208 - Accuracy: 0.5021
    Epoch 61/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9037 - Accuracy: 0.5059
    Epoch 62/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9146 - Accuracy: 0.4982
    Epoch 63/500
    500/500 [==============================] - 2s 5ms/step - loss: 0.9147 - Accuracy: 0.4988
    Epoch 64/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9096 - Accuracy: 0.5106
    Epoch 65/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9246 - Accuracy: 0.5059
    Epoch 66/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8919 - Accuracy: 0.5029
    Epoch 67/500
    500/500 [==============================] - 2s 5ms/step - loss: 0.9191 - Accuracy: 0.5019
    Epoch 68/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8974 - Accuracy: 0.5131
    Epoch 69/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8923 - Accuracy: 0.5047
    Epoch 70/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8901 - Accuracy: 0.5131
    Epoch 71/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9010 - Accuracy: 0.4989
    Epoch 72/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8906 - Accuracy: 0.5076
    Epoch 73/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9143 - Accuracy: 0.5016
    Epoch 74/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8952 - Accuracy: 0.5052
    Epoch 75/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8730 - Accuracy: 0.5129
    Epoch 76/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8813 - Accuracy: 0.5098
    Epoch 77/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8786 - Accuracy: 0.5073
    Epoch 78/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8853 - Accuracy: 0.5054
    Epoch 79/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8904 - Accuracy: 0.5104
    Epoch 80/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8934 - Accuracy: 0.5091
    Epoch 81/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8865 - Accuracy: 0.5072
    Epoch 82/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8825 - Accuracy: 0.5004
    Epoch 83/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8919 - Accuracy: 0.5116
    Epoch 84/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8635 - Accuracy: 0.5051
    Epoch 85/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8891 - Accuracy: 0.5110
    Epoch 86/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8685 - Accuracy: 0.5131
    Epoch 87/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.9017 - Accuracy: 0.5046
    Epoch 88/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8582 - Accuracy: 0.4989
    Epoch 89/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8709 - Accuracy: 0.5130
    Epoch 90/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8693 - Accuracy: 0.5126
    Epoch 91/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8735 - Accuracy: 0.5139
    Epoch 92/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8635 - Accuracy: 0.5166
    Epoch 93/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8759 - Accuracy: 0.5119
    Epoch 94/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8551 - Accuracy: 0.5119
    Epoch 95/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8768 - Accuracy: 0.5090
    Epoch 96/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8580 - Accuracy: 0.5098
    Epoch 97/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8716 - Accuracy: 0.5173
    Epoch 98/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8655 - Accuracy: 0.5170
    Epoch 99/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8587 - Accuracy: 0.5135
    Epoch 100/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8688 - Accuracy: 0.5123
    Epoch 101/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8503 - Accuracy: 0.5136
    Epoch 102/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8722 - Accuracy: 0.5203
    Epoch 103/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8474 - Accuracy: 0.5237
    Epoch 104/500
    500/500 [==============================] - 2s 5ms/step - loss: 0.8661 - Accuracy: 0.5185
    Epoch 105/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8600 - Accuracy: 0.5162
    Epoch 106/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8705 - Accuracy: 0.5104
    Epoch 107/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8525 - Accuracy: 0.5169
    Epoch 108/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8609 - Accuracy: 0.5272
    Epoch 109/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8941 - Accuracy: 0.5167
    Epoch 110/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8573 - Accuracy: 0.5150
    Epoch 111/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8560 - Accuracy: 0.5164
    Epoch 112/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8662 - Accuracy: 0.5172
    Epoch 113/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8627 - Accuracy: 0.5149
    Epoch 114/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8537 - Accuracy: 0.5123
    Epoch 115/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8538 - Accuracy: 0.5150
    Epoch 116/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8599 - Accuracy: 0.5181
    Epoch 117/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8679 - Accuracy: 0.5106
    Epoch 118/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8421 - Accuracy: 0.5297
    Epoch 119/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8675 - Accuracy: 0.5140
    Epoch 120/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8598 - Accuracy: 0.5206
    Epoch 121/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8520 - Accuracy: 0.5232
    Epoch 122/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8570 - Accuracy: 0.5262
    Epoch 123/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8537 - Accuracy: 0.5209
    Epoch 124/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8629 - Accuracy: 0.5120
    Epoch 125/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8591 - Accuracy: 0.5204
    Epoch 126/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8779 - Accuracy: 0.5232
    Epoch 127/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8411 - Accuracy: 0.5292
    Epoch 128/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8973 - Accuracy: 0.5231
    Epoch 129/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8545 - Accuracy: 0.5211
    Epoch 130/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8586 - Accuracy: 0.5301
    Epoch 131/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8433 - Accuracy: 0.5239
    Epoch 132/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8480 - Accuracy: 0.5357
    Epoch 133/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8549 - Accuracy: 0.5377
    Epoch 134/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8790 - Accuracy: 0.5220
    Epoch 135/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8586 - Accuracy: 0.5421
    Epoch 136/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8585 - Accuracy: 0.5430
    Epoch 137/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8638 - Accuracy: 0.5355
    Epoch 138/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8476 - Accuracy: 0.5429
    Epoch 139/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8555 - Accuracy: 0.5472
    Epoch 140/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8478 - Accuracy: 0.5645
    Epoch 141/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8287 - Accuracy: 0.5884
    Epoch 142/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8525 - Accuracy: 0.5983
    Epoch 143/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8125 - Accuracy: 0.6298
    Epoch 144/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.8167 - Accuracy: 0.6496
    Epoch 145/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.7789 - Accuracy: 0.6802
    Epoch 146/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.7880 - Accuracy: 0.6943
    Epoch 147/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.7418 - Accuracy: 0.7149
    Epoch 148/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.7066 - Accuracy: 0.7355
    Epoch 149/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.7052 - Accuracy: 0.7340
    Epoch 150/500
    500/500 [==============================] - 2s 5ms/step - loss: 0.7179 - Accuracy: 0.7386
    Epoch 151/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.7165 - Accuracy: 0.7499
    Epoch 152/500
    500/500 [==============================] - 2s 5ms/step - loss: 0.7396 - Accuracy: 0.7492
    Epoch 153/500
    500/500 [==============================] - 2s 5ms/step - loss: 0.6794 - Accuracy: 0.7579
    Epoch 154/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6981 - Accuracy: 0.7599
    Epoch 155/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6719 - Accuracy: 0.7596
    Epoch 156/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6705 - Accuracy: 0.7676
    Epoch 157/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6911 - Accuracy: 0.7607
    Epoch 158/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6859 - Accuracy: 0.7642
    Epoch 159/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6609 - Accuracy: 0.7742
    Epoch 160/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6553 - Accuracy: 0.7779
    Epoch 161/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6350 - Accuracy: 0.7824
    Epoch 162/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6328 - Accuracy: 0.7851
    Epoch 163/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6274 - Accuracy: 0.7914
    Epoch 164/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6113 - Accuracy: 0.7987
    Epoch 165/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6183 - Accuracy: 0.7974
    Epoch 166/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5992 - Accuracy: 0.7999
    Epoch 167/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.6006 - Accuracy: 0.8059
    Epoch 168/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5976 - Accuracy: 0.8096
    Epoch 169/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5804 - Accuracy: 0.8143
    Epoch 170/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5850 - Accuracy: 0.8148
    Epoch 171/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5802 - Accuracy: 0.8183
    Epoch 172/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5595 - Accuracy: 0.8199
    Epoch 173/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5709 - Accuracy: 0.8177
    Epoch 174/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5620 - Accuracy: 0.8190
    Epoch 175/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5543 - Accuracy: 0.8209
    Epoch 176/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5534 - Accuracy: 0.8211
    Epoch 177/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5344 - Accuracy: 0.8256
    Epoch 178/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5452 - Accuracy: 0.8238
    Epoch 179/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5405 - Accuracy: 0.8264
    Epoch 180/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5436 - Accuracy: 0.8263
    Epoch 181/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5446 - Accuracy: 0.8257
    Epoch 182/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5285 - Accuracy: 0.8341
    Epoch 183/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5295 - Accuracy: 0.8293
    Epoch 184/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5476 - Accuracy: 0.8291
    Epoch 185/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5258 - Accuracy: 0.8348
    Epoch 186/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5340 - Accuracy: 0.8326
    Epoch 187/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5255 - Accuracy: 0.8316
    Epoch 188/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5246 - Accuracy: 0.8324
    Epoch 189/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5161 - Accuracy: 0.8358
    Epoch 190/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5110 - Accuracy: 0.8364
    Epoch 191/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5233 - Accuracy: 0.8307
    Epoch 192/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5182 - Accuracy: 0.8330
    Epoch 193/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5228 - Accuracy: 0.8304
    Epoch 194/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5165 - Accuracy: 0.8341
    Epoch 195/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5171 - Accuracy: 0.8345
    Epoch 196/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5136 - Accuracy: 0.8372
    Epoch 197/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5067 - Accuracy: 0.8371
    Epoch 198/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5053 - Accuracy: 0.8372
    Epoch 199/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4995 - Accuracy: 0.8389
    Epoch 200/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5094 - Accuracy: 0.8366
    Epoch 201/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5059 - Accuracy: 0.8368
    Epoch 202/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5216 - Accuracy: 0.8353
    Epoch 203/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4970 - Accuracy: 0.8397
    Epoch 204/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5015 - Accuracy: 0.8419
    Epoch 205/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5021 - Accuracy: 0.8410
    Epoch 206/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5007 - Accuracy: 0.8393
    Epoch 207/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4978 - Accuracy: 0.8398
    Epoch 208/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4936 - Accuracy: 0.8438
    Epoch 209/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4836 - Accuracy: 0.8451
    Epoch 210/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.5078 - Accuracy: 0.8399
    Epoch 211/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4840 - Accuracy: 0.8465
    Epoch 212/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4837 - Accuracy: 0.8455
    Epoch 213/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4953 - Accuracy: 0.8519
    Epoch 214/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4706 - Accuracy: 0.8536
    Epoch 215/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4721 - Accuracy: 0.8541
    Epoch 216/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4638 - Accuracy: 0.8579
    Epoch 217/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4665 - Accuracy: 0.8567
    Epoch 218/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4667 - Accuracy: 0.8580
    Epoch 219/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4633 - Accuracy: 0.8573
    Epoch 220/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4502 - Accuracy: 0.8633
    Epoch 221/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4519 - Accuracy: 0.8671
    Epoch 222/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4363 - Accuracy: 0.8704
    Epoch 223/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4392 - Accuracy: 0.8760
    Epoch 224/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4281 - Accuracy: 0.8809
    Epoch 225/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4313 - Accuracy: 0.8784
    Epoch 226/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4169 - Accuracy: 0.8829
    Epoch 227/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4090 - Accuracy: 0.8851
    Epoch 228/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4086 - Accuracy: 0.8873
    Epoch 229/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4074 - Accuracy: 0.8866
    Epoch 230/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3963 - Accuracy: 0.8954
    Epoch 231/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.4028 - Accuracy: 0.8930
    Epoch 232/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3915 - Accuracy: 0.8945
    Epoch 233/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3908 - Accuracy: 0.8979
    Epoch 234/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3898 - Accuracy: 0.8947
    Epoch 235/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3835 - Accuracy: 0.8986
    Epoch 236/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3899 - Accuracy: 0.8960
    Epoch 237/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3865 - Accuracy: 0.8986
    Epoch 238/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3929 - Accuracy: 0.8974
    Epoch 239/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3805 - Accuracy: 0.9001
    Epoch 240/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3877 - Accuracy: 0.8970
    Epoch 241/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3754 - Accuracy: 0.9027
    Epoch 242/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3729 - Accuracy: 0.9040
    Epoch 243/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3862 - Accuracy: 0.8992
    Epoch 244/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3814 - Accuracy: 0.8993
    Epoch 245/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3721 - Accuracy: 0.9008
    Epoch 246/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3769 - Accuracy: 0.9014
    Epoch 247/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3867 - Accuracy: 0.8992
    Epoch 248/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3724 - Accuracy: 0.9042
    Epoch 249/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3813 - Accuracy: 0.9007
    Epoch 250/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3745 - Accuracy: 0.9025
    Epoch 251/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3783 - Accuracy: 0.9005
    Epoch 252/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3684 - Accuracy: 0.9052
    Epoch 253/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3840 - Accuracy: 0.8978
    Epoch 254/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3769 - Accuracy: 0.9030
    Epoch 255/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3740 - Accuracy: 0.9043
    Epoch 256/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3725 - Accuracy: 0.9030
    Epoch 257/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3709 - Accuracy: 0.9027
    Epoch 258/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3668 - Accuracy: 0.9053
    Epoch 259/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3721 - Accuracy: 0.9019
    Epoch 260/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3714 - Accuracy: 0.9044
    Epoch 261/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3699 - Accuracy: 0.9018
    Epoch 262/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3642 - Accuracy: 0.9049
    Epoch 263/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3638 - Accuracy: 0.9067
    Epoch 264/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3657 - Accuracy: 0.9046
    Epoch 265/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3686 - Accuracy: 0.8994
    Epoch 266/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3702 - Accuracy: 0.9044
    Epoch 267/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3681 - Accuracy: 0.9050
    Epoch 268/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3592 - Accuracy: 0.9056
    Epoch 269/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3623 - Accuracy: 0.9057
    Epoch 270/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3660 - Accuracy: 0.9032
    Epoch 271/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3621 - Accuracy: 0.9044
    Epoch 272/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3698 - Accuracy: 0.9019
    Epoch 273/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3656 - Accuracy: 0.9039
    Epoch 274/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3657 - Accuracy: 0.9028
    Epoch 275/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3575 - Accuracy: 0.9049
    Epoch 276/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3606 - Accuracy: 0.9053
    Epoch 277/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3609 - Accuracy: 0.9044
    Epoch 278/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3621 - Accuracy: 0.9046
    Epoch 279/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3592 - Accuracy: 0.9051
    Epoch 280/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3640 - Accuracy: 0.9047
    Epoch 281/500
    500/500 [==============================] - 2s 5ms/step - loss: 0.3544 - Accuracy: 0.9062
    Epoch 282/500
    500/500 [==============================] - 2s 5ms/step - loss: 0.3636 - Accuracy: 0.9039
    Epoch 283/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3588 - Accuracy: 0.9067
    Epoch 284/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3610 - Accuracy: 0.9038
    Epoch 285/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3586 - Accuracy: 0.9069
    Epoch 286/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3560 - Accuracy: 0.9054
    Epoch 287/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3538 - Accuracy: 0.9081
    Epoch 288/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3512 - Accuracy: 0.9079
    Epoch 289/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3524 - Accuracy: 0.9085
    Epoch 290/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3558 - Accuracy: 0.9049
    Epoch 291/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3590 - Accuracy: 0.9049
    Epoch 292/500
    500/500 [==============================] - 2s 5ms/step - loss: 0.3573 - Accuracy: 0.9057
    Epoch 293/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3555 - Accuracy: 0.9056
    Epoch 294/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3570 - Accuracy: 0.9056
    Epoch 295/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3547 - Accuracy: 0.9066
    Epoch 296/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3549 - Accuracy: 0.9058
    Epoch 297/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3495 - Accuracy: 0.9081
    Epoch 298/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3513 - Accuracy: 0.9068
    Epoch 299/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3449 - Accuracy: 0.9128
    Epoch 300/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3501 - Accuracy: 0.9070
    Epoch 301/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3500 - Accuracy: 0.9062
    Epoch 302/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3474 - Accuracy: 0.9080
    Epoch 303/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3507 - Accuracy: 0.9056
    Epoch 304/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3493 - Accuracy: 0.9053
    Epoch 305/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3561 - Accuracy: 0.9035
    Epoch 306/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3504 - Accuracy: 0.9049
    Epoch 307/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3484 - Accuracy: 0.9083
    Epoch 308/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3510 - Accuracy: 0.9064
    Epoch 309/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3454 - Accuracy: 0.9089
    Epoch 310/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3565 - Accuracy: 0.9065
    Epoch 311/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3414 - Accuracy: 0.9077
    Epoch 312/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3470 - Accuracy: 0.9086
    Epoch 313/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3450 - Accuracy: 0.9085
    Epoch 314/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3465 - Accuracy: 0.9045
    Epoch 315/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3485 - Accuracy: 0.9084
    Epoch 316/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3433 - Accuracy: 0.9083
    Epoch 317/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3496 - Accuracy: 0.9059
    Epoch 318/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3447 - Accuracy: 0.9093
    Epoch 319/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3384 - Accuracy: 0.9102
    Epoch 320/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3414 - Accuracy: 0.9081
    Epoch 321/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3442 - Accuracy: 0.9091
    Epoch 322/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3506 - Accuracy: 0.9065
    Epoch 323/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3419 - Accuracy: 0.9066
    Epoch 324/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3437 - Accuracy: 0.9079
    Epoch 325/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3421 - Accuracy: 0.9087
    Epoch 326/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3438 - Accuracy: 0.9079
    Epoch 327/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3436 - Accuracy: 0.9080
    Epoch 328/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3394 - Accuracy: 0.9082
    Epoch 329/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3415 - Accuracy: 0.9077
    Epoch 330/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3425 - Accuracy: 0.9066
    Epoch 331/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3369 - Accuracy: 0.9091
    Epoch 332/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3382 - Accuracy: 0.9094
    Epoch 333/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3407 - Accuracy: 0.9078
    Epoch 334/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3396 - Accuracy: 0.9083
    Epoch 335/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3395 - Accuracy: 0.9095
    Epoch 336/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3403 - Accuracy: 0.9092
    Epoch 337/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3359 - Accuracy: 0.9095
    Epoch 338/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3373 - Accuracy: 0.9086
    Epoch 339/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3412 - Accuracy: 0.9068
    Epoch 340/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3372 - Accuracy: 0.9067
    Epoch 341/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3345 - Accuracy: 0.9093
    Epoch 342/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3340 - Accuracy: 0.9081
    Epoch 343/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3345 - Accuracy: 0.9101
    Epoch 344/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3371 - Accuracy: 0.9071
    Epoch 345/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3390 - Accuracy: 0.9073
    Epoch 346/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3350 - Accuracy: 0.9079
    Epoch 347/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3375 - Accuracy: 0.9074
    Epoch 348/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3353 - Accuracy: 0.9091
    Epoch 349/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3311 - Accuracy: 0.9096
    Epoch 350/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3379 - Accuracy: 0.9060
    Epoch 351/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3332 - Accuracy: 0.9079
    Epoch 352/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3359 - Accuracy: 0.9079
    Epoch 353/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3333 - Accuracy: 0.9107
    Epoch 354/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3315 - Accuracy: 0.9093
    Epoch 355/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3308 - Accuracy: 0.9083
    Epoch 356/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3310 - Accuracy: 0.9092
    Epoch 357/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3311 - Accuracy: 0.9086
    Epoch 358/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3342 - Accuracy: 0.9076
    Epoch 359/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3341 - Accuracy: 0.9066
    Epoch 360/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3303 - Accuracy: 0.9097
    Epoch 361/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3331 - Accuracy: 0.9080
    Epoch 362/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3297 - Accuracy: 0.9082
    Epoch 363/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3300 - Accuracy: 0.9094
    Epoch 364/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3334 - Accuracy: 0.9103
    Epoch 365/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3304 - Accuracy: 0.9080
    Epoch 366/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3321 - Accuracy: 0.9090
    Epoch 367/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3288 - Accuracy: 0.9093
    Epoch 368/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3273 - Accuracy: 0.9079
    Epoch 369/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3261 - Accuracy: 0.9093
    Epoch 370/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3283 - Accuracy: 0.9097
    Epoch 371/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3274 - Accuracy: 0.9090
    Epoch 372/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3283 - Accuracy: 0.9079
    Epoch 373/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3262 - Accuracy: 0.9103
    Epoch 374/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3291 - Accuracy: 0.9087
    Epoch 375/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3297 - Accuracy: 0.9076
    Epoch 376/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3259 - Accuracy: 0.9087
    Epoch 377/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3267 - Accuracy: 0.9089
    Epoch 378/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3285 - Accuracy: 0.9089
    Epoch 379/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3268 - Accuracy: 0.9094
    Epoch 380/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3268 - Accuracy: 0.9086
    Epoch 381/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3273 - Accuracy: 0.9091
    Epoch 382/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3225 - Accuracy: 0.9107
    Epoch 383/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3202 - Accuracy: 0.9110
    Epoch 384/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3271 - Accuracy: 0.9094
    Epoch 385/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3271 - Accuracy: 0.9078
    Epoch 386/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3241 - Accuracy: 0.9101
    Epoch 387/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3248 - Accuracy: 0.9100
    Epoch 388/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3238 - Accuracy: 0.9099
    Epoch 389/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3261 - Accuracy: 0.9082
    Epoch 390/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3254 - Accuracy: 0.9081
    Epoch 391/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3236 - Accuracy: 0.9085
    Epoch 392/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3238 - Accuracy: 0.9087
    Epoch 393/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3223 - Accuracy: 0.9093
    Epoch 394/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3198 - Accuracy: 0.9097
    Epoch 395/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3232 - Accuracy: 0.9103
    Epoch 396/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3242 - Accuracy: 0.9085
    Epoch 397/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3226 - Accuracy: 0.9088
    Epoch 398/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3188 - Accuracy: 0.9098
    Epoch 399/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3224 - Accuracy: 0.9083
    Epoch 400/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3232 - Accuracy: 0.9080
    Epoch 401/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3221 - Accuracy: 0.9098
    Epoch 402/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3205 - Accuracy: 0.9095
    Epoch 403/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3196 - Accuracy: 0.9102
    Epoch 404/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3169 - Accuracy: 0.9114
    Epoch 405/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3203 - Accuracy: 0.9084
    Epoch 406/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3204 - Accuracy: 0.9087
    Epoch 407/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3190 - Accuracy: 0.9091
    Epoch 408/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3176 - Accuracy: 0.9099
    Epoch 409/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3175 - Accuracy: 0.9109
    Epoch 410/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3149 - Accuracy: 0.9115
    Epoch 411/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3194 - Accuracy: 0.9102
    Epoch 412/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3212 - Accuracy: 0.9088
    Epoch 413/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3182 - Accuracy: 0.9090
    Epoch 414/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3149 - Accuracy: 0.9095
    Epoch 415/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3173 - Accuracy: 0.9121
    Epoch 416/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3145 - Accuracy: 0.9110
    Epoch 417/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3163 - Accuracy: 0.9093
    Epoch 418/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3125 - Accuracy: 0.9119
    Epoch 419/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3167 - Accuracy: 0.9094
    Epoch 420/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3148 - Accuracy: 0.9097
    Epoch 421/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3177 - Accuracy: 0.9081
    Epoch 422/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3184 - Accuracy: 0.9091
    Epoch 423/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3153 - Accuracy: 0.9111
    Epoch 424/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3143 - Accuracy: 0.9117
    Epoch 425/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3139 - Accuracy: 0.9111
    Epoch 426/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3147 - Accuracy: 0.9093
    Epoch 427/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3144 - Accuracy: 0.9087
    Epoch 428/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3123 - Accuracy: 0.9128
    Epoch 429/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3132 - Accuracy: 0.9107
    Epoch 430/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3130 - Accuracy: 0.9107
    Epoch 431/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3150 - Accuracy: 0.9089
    Epoch 432/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3137 - Accuracy: 0.9097
    Epoch 433/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3152 - Accuracy: 0.9076
    Epoch 434/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3160 - Accuracy: 0.9094
    Epoch 435/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3125 - Accuracy: 0.9093
    Epoch 436/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3152 - Accuracy: 0.9064
    Epoch 437/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3108 - Accuracy: 0.9124
    Epoch 438/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3124 - Accuracy: 0.9097
    Epoch 439/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3111 - Accuracy: 0.9103
    Epoch 440/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3110 - Accuracy: 0.9108
    Epoch 441/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3105 - Accuracy: 0.9096
    Epoch 442/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3124 - Accuracy: 0.9105
    Epoch 443/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3118 - Accuracy: 0.9079
    Epoch 444/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3126 - Accuracy: 0.9091
    Epoch 445/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3100 - Accuracy: 0.9107
    Epoch 446/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3093 - Accuracy: 0.9093
    Epoch 447/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3122 - Accuracy: 0.9089
    Epoch 448/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3149 - Accuracy: 0.9098
    Epoch 449/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3128 - Accuracy: 0.9086
    Epoch 450/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3097 - Accuracy: 0.9120
    Epoch 451/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3086 - Accuracy: 0.9097
    Epoch 452/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3060 - Accuracy: 0.9124
    Epoch 453/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3085 - Accuracy: 0.9106
    Epoch 454/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3090 - Accuracy: 0.9103
    Epoch 455/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3071 - Accuracy: 0.9110
    Epoch 456/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3077 - Accuracy: 0.9112
    Epoch 457/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3080 - Accuracy: 0.9106
    Epoch 458/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3081 - Accuracy: 0.9104
    Epoch 459/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3080 - Accuracy: 0.9112
    Epoch 460/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3072 - Accuracy: 0.9109
    Epoch 461/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3080 - Accuracy: 0.9093
    Epoch 462/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3076 - Accuracy: 0.9100
    Epoch 463/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3075 - Accuracy: 0.9103
    Epoch 464/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3084 - Accuracy: 0.9103
    Epoch 465/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3079 - Accuracy: 0.9103
    Epoch 466/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3043 - Accuracy: 0.9101
    Epoch 467/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3072 - Accuracy: 0.9103
    Epoch 468/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3068 - Accuracy: 0.9095
    Epoch 469/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3058 - Accuracy: 0.9080
    Epoch 470/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3065 - Accuracy: 0.9125
    Epoch 471/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3053 - Accuracy: 0.9122
    Epoch 472/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3050 - Accuracy: 0.9119
    Epoch 473/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3042 - Accuracy: 0.9120
    Epoch 474/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3068 - Accuracy: 0.9103
    Epoch 475/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3060 - Accuracy: 0.9107
    Epoch 476/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3071 - Accuracy: 0.9119
    Epoch 477/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3043 - Accuracy: 0.9122
    Epoch 478/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3061 - Accuracy: 0.9104
    Epoch 479/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3043 - Accuracy: 0.9129
    Epoch 480/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3056 - Accuracy: 0.9101
    Epoch 481/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3062 - Accuracy: 0.9107
    Epoch 482/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3043 - Accuracy: 0.9137
    Epoch 483/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3031 - Accuracy: 0.9123
    Epoch 484/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3061 - Accuracy: 0.9107
    Epoch 485/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3043 - Accuracy: 0.9087
    Epoch 486/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3037 - Accuracy: 0.9103
    Epoch 487/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3039 - Accuracy: 0.9104
    Epoch 488/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3011 - Accuracy: 0.9121
    Epoch 489/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3041 - Accuracy: 0.9107
    Epoch 490/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3013 - Accuracy: 0.9109
    Epoch 491/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3042 - Accuracy: 0.9105
    Epoch 492/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3029 - Accuracy: 0.9116
    Epoch 493/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3013 - Accuracy: 0.9106
    Epoch 494/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3007 - Accuracy: 0.9122
    Epoch 495/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3017 - Accuracy: 0.9100
    Epoch 496/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3030 - Accuracy: 0.9113
    Epoch 497/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3009 - Accuracy: 0.9123
    Epoch 498/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3023 - Accuracy: 0.9100
    Epoch 499/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3011 - Accuracy: 0.9124
    Epoch 500/500
    500/500 [==============================] - 2s 4ms/step - loss: 0.3005 - Accuracy: 0.9113





    <keras.src.callbacks.History at 0x7fb8647d3100>



* TODO remove


```python
epistemic_model.save('./large_epitemic_model')
```

    WARNING:tensorflow:`_` is not a valid node name. Accepted names conform to Regex /re.compile('^[A-Za-z0-9.][A-Za-z0-9_.\\\\/>-]*$')/
    WARNING:tensorflow:`_` is not a valid node name. Accepted names conform to Regex /re.compile('^[A-Za-z0-9.][A-Za-z0-9_.\\\\/>-]*$')/
    WARNING:tensorflow:`_` is not a valid node name. Accepted names conform to Regex /re.compile('^[A-Za-z0-9.][A-Za-z0-9_.\\\\/>-]*$')/
    WARNING:tensorflow:`_` is not a valid node name. Accepted names conform to Regex /re.compile('^[A-Za-z0-9.][A-Za-z0-9_.\\\\/>-]*$')/
    WARNING:tensorflow:`_` is not a valid node name. Accepted names conform to Regex /re.compile('^[A-Za-z0-9.][A-Za-z0-9_.\\\\/>-]*$')/
    WARNING:tensorflow:`_` is not a valid node name. Accepted names conform to Regex /re.compile('^[A-Za-z0-9.][A-Za-z0-9_.\\\\/>-]*$')/
    INFO:tensorflow:Assets written to: ./large_epitemic_model/assets


    INFO:tensorflow:Assets written to: ./large_epitemic_model/assets



```python
test = tf.keras.models.load_model('./large_epitemic_model')
```

    WARNING:absl:`_` is not a valid tf.function parameter name. Sanitizing to `arg__`.
    WARNING:absl:`_` is not a valid tf.function parameter name. Sanitizing to `arg__`.
    WARNING:absl:`_` is not a valid tf.function parameter name. Sanitizing to `arg__`.
    WARNING:absl:`_` is not a valid tf.function parameter name. Sanitizing to `arg__`.
    WARNING:absl:`_` is not a valid tf.function parameter name. Sanitizing to `arg__`.



```python
def infer(X, y, model_, it=10): 
  y_preds = []
  for _ in range(it): 
    y_preds.append(model_(X))
  
  pred_mean = np.mean(y_preds, axis=0)
  pred_stdv = np.std(y_preds, axis=0)
  
  print('accuracy : ', accuracy_score(y, pred_mean.round()))

  return pred_mean, pred_stdv
```


```python
pred_mean, pred_stdv = infer(X_test, y_test, epistemic_model, 100) 

fig = plt.figure(figsize=(10, 10))
plt.subplot(4, 1, 1)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
plt.title('test set')

plt.subplot(4,2,3)
plt.scatter(X_test[:,0], X_test[:,1], c=pred_mean.round())
plt.title('predictions (bigger train set)')

plt.subplot(4,2,5)
plt.scatter(X_test[:,0], X_test[:,1], c=pred_mean)
plt.title('predictions (color gradient on prediction mean i.e sharpness)')

plt.subplot(4,2,7)
plt.scatter(X_test[:,0], X_test[:,1], c=pred_stdv)
plt.title('standard deviation on the inference, the closer to yellow => higher epistemic uncertainty')

plt.tight_layout()

```

    accuracy :  0.9135



    
![png](output_43_1.png)
    



```python

```


```python
small_epistemic_model = create_epistemic_model(multivariate_normal_prior, posterior)
small_epistemic_model.fit(X_train[:1000], y_train[:1000], batch_size=32, verbose=1, epochs=4000) 
```

    Model: "epistemic_BNN"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_6 (InputLayer)        [(None, 2)]               0         
                                                                     
     dense_variational_15 (Dens  (None, 200)               1200      
     eVariational)                                                   
                                                                     
     dense_variational_16 (Dens  (None, 200)               80400     
     eVariational)                                                   
                                                                     
     dense_variational_17 (Dens  (None, 1)                 402       
     eVariational)                                                   
                                                                     
    =================================================================
    Total params: 82002 (320.32 KB)
    Trainable params: 82002 (320.32 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    Epoch 1/3000
    32/32 [==============================] - 3s 4ms/step - loss: 24.3681 - Accuracy: 0.4490
    Epoch 2/3000
    32/32 [==============================] - 0s 4ms/step - loss: 22.7527 - Accuracy: 0.5040
    Epoch 3/3000
    32/32 [==============================] - 0s 4ms/step - loss: 24.1481 - Accuracy: 0.4570
    Epoch 4/3000
    32/32 [==============================] - 0s 4ms/step - loss: 18.9699 - Accuracy: 0.5060
    Epoch 5/3000
    32/32 [==============================] - 0s 4ms/step - loss: 17.9620 - Accuracy: 0.4820
    Epoch 6/3000
    32/32 [==============================] - 0s 4ms/step - loss: 17.5652 - Accuracy: 0.5020
    Epoch 7/3000
    32/32 [==============================] - 0s 4ms/step - loss: 22.2962 - Accuracy: 0.4770
    Epoch 8/3000
    32/32 [==============================] - 0s 4ms/step - loss: 15.8657 - Accuracy: 0.4670
    Epoch 9/3000
    32/32 [==============================] - 0s 4ms/step - loss: 16.7591 - Accuracy: 0.5070
    Epoch 10/3000
    32/32 [==============================] - 0s 4ms/step - loss: 17.7823 - Accuracy: 0.4560
    Epoch 11/3000
    32/32 [==============================] - 0s 4ms/step - loss: 19.4992 - Accuracy: 0.5060
    Epoch 12/3000
    32/32 [==============================] - 0s 4ms/step - loss: 21.3543 - Accuracy: 0.5030
    Epoch 13/3000
    32/32 [==============================] - 0s 4ms/step - loss: 16.7631 - Accuracy: 0.5310
    Epoch 14/3000
    32/32 [==============================] - 0s 4ms/step - loss: 17.3566 - Accuracy: 0.5430
    Epoch 15/3000
    32/32 [==============================] - 0s 4ms/step - loss: 16.8723 - Accuracy: 0.4870
    Epoch 16/3000
    32/32 [==============================] - 0s 4ms/step - loss: 17.8627 - Accuracy: 0.4870
    Epoch 17/3000
    32/32 [==============================] - 0s 4ms/step - loss: 15.0913 - Accuracy: 0.5020
    Epoch 18/3000
    32/32 [==============================] - 0s 4ms/step - loss: 20.3240 - Accuracy: 0.4720
    Epoch 19/3000
    32/32 [==============================] - 0s 4ms/step - loss: 15.7909 - Accuracy: 0.4960
    Epoch 20/3000
    32/32 [==============================] - 0s 4ms/step - loss: 16.3616 - Accuracy: 0.5040
    Epoch 21/3000
    32/32 [==============================] - 0s 4ms/step - loss: 12.6887 - Accuracy: 0.5540
    Epoch 22/3000
    32/32 [==============================] - 0s 4ms/step - loss: 12.4267 - Accuracy: 0.5530
    Epoch 23/3000
    32/32 [==============================] - 0s 4ms/step - loss: 17.2068 - Accuracy: 0.4790
    Epoch 24/3000
    32/32 [==============================] - 0s 4ms/step - loss: 15.3123 - Accuracy: 0.4820
    Epoch 25/3000
    32/32 [==============================] - 0s 4ms/step - loss: 16.6790 - Accuracy: 0.5040
    Epoch 26/3000
    32/32 [==============================] - 0s 4ms/step - loss: 15.2409 - Accuracy: 0.4610
    Epoch 27/3000
    32/32 [==============================] - 0s 4ms/step - loss: 15.0514 - Accuracy: 0.5230
    Epoch 28/3000
    32/32 [==============================] - 0s 4ms/step - loss: 13.0421 - Accuracy: 0.4770
    Epoch 29/3000
    32/32 [==============================] - 0s 4ms/step - loss: 11.9618 - Accuracy: 0.4860
    Epoch 30/3000
    32/32 [==============================] - 0s 4ms/step - loss: 16.0389 - Accuracy: 0.4630
    Epoch 31/3000
    32/32 [==============================] - 0s 4ms/step - loss: 10.9017 - Accuracy: 0.4880
    Epoch 32/3000
    32/32 [==============================] - 0s 4ms/step - loss: 14.6796 - Accuracy: 0.4730
    Epoch 33/3000
    32/32 [==============================] - 0s 4ms/step - loss: 16.6035 - Accuracy: 0.4520
    Epoch 34/3000
    32/32 [==============================] - 0s 4ms/step - loss: 11.1335 - Accuracy: 0.5250
    Epoch 35/3000
    32/32 [==============================] - 0s 4ms/step - loss: 12.3135 - Accuracy: 0.5010
    Epoch 36/3000
    32/32 [==============================] - 0s 4ms/step - loss: 16.1239 - Accuracy: 0.4730
    Epoch 37/3000
    32/32 [==============================] - 0s 4ms/step - loss: 11.9591 - Accuracy: 0.5110
    Epoch 38/3000
    32/32 [==============================] - 0s 4ms/step - loss: 15.0942 - Accuracy: 0.4650
    Epoch 39/3000
    32/32 [==============================] - 0s 4ms/step - loss: 13.7789 - Accuracy: 0.4800
    Epoch 40/3000
    32/32 [==============================] - 0s 4ms/step - loss: 12.8359 - Accuracy: 0.5050
    Epoch 41/3000
    32/32 [==============================] - 0s 4ms/step - loss: 12.7118 - Accuracy: 0.5270
    Epoch 42/3000
    32/32 [==============================] - 0s 4ms/step - loss: 11.4181 - Accuracy: 0.5230
    Epoch 43/3000
    32/32 [==============================] - 0s 4ms/step - loss: 11.1098 - Accuracy: 0.4720
    Epoch 44/3000
    32/32 [==============================] - 0s 4ms/step - loss: 11.8208 - Accuracy: 0.4870
    Epoch 45/3000
    32/32 [==============================] - 0s 4ms/step - loss: 10.2405 - Accuracy: 0.5240
    Epoch 46/3000
    32/32 [==============================] - 0s 4ms/step - loss: 9.2029 - Accuracy: 0.5330
    Epoch 47/3000
    32/32 [==============================] - 0s 4ms/step - loss: 9.3763 - Accuracy: 0.5130
    Epoch 48/3000
    32/32 [==============================] - 0s 4ms/step - loss: 9.9076 - Accuracy: 0.5050
    Epoch 49/3000
    32/32 [==============================] - 0s 4ms/step - loss: 11.5348 - Accuracy: 0.4950
    Epoch 50/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.6204 - Accuracy: 0.5270
    Epoch 51/3000
    32/32 [==============================] - 0s 4ms/step - loss: 11.9713 - Accuracy: 0.4660
    Epoch 52/3000
    32/32 [==============================] - 0s 4ms/step - loss: 12.2992 - Accuracy: 0.5040
    Epoch 53/3000
    32/32 [==============================] - 0s 4ms/step - loss: 9.7860 - Accuracy: 0.5320
    Epoch 54/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.2948 - Accuracy: 0.5010
    Epoch 55/3000
    32/32 [==============================] - 0s 4ms/step - loss: 10.2619 - Accuracy: 0.4610
    Epoch 56/3000
    32/32 [==============================] - 0s 4ms/step - loss: 9.3545 - Accuracy: 0.5100
    Epoch 57/3000
    32/32 [==============================] - 0s 4ms/step - loss: 11.8206 - Accuracy: 0.4750
    Epoch 58/3000
    32/32 [==============================] - 0s 4ms/step - loss: 12.7491 - Accuracy: 0.4430
    Epoch 59/3000
    32/32 [==============================] - 0s 4ms/step - loss: 10.1546 - Accuracy: 0.4850
    Epoch 60/3000
    32/32 [==============================] - 0s 4ms/step - loss: 9.1599 - Accuracy: 0.5160
    Epoch 61/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.6339 - Accuracy: 0.5310
    Epoch 62/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.4200 - Accuracy: 0.4840
    Epoch 63/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.7856 - Accuracy: 0.4910
    Epoch 64/3000
    32/32 [==============================] - 0s 4ms/step - loss: 9.0730 - Accuracy: 0.4660
    Epoch 65/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.1352 - Accuracy: 0.4910
    Epoch 66/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.7392 - Accuracy: 0.5320
    Epoch 67/3000
    32/32 [==============================] - 0s 4ms/step - loss: 10.1542 - Accuracy: 0.4900
    Epoch 68/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.8260 - Accuracy: 0.5260
    Epoch 69/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.4882 - Accuracy: 0.4780
    Epoch 70/3000
    32/32 [==============================] - 0s 4ms/step - loss: 7.7527 - Accuracy: 0.5140
    Epoch 71/3000
    32/32 [==============================] - 0s 4ms/step - loss: 9.4830 - Accuracy: 0.4850
    Epoch 72/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.5103 - Accuracy: 0.5310
    Epoch 73/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.9314 - Accuracy: 0.5330
    Epoch 74/3000
    32/32 [==============================] - 0s 4ms/step - loss: 9.0514 - Accuracy: 0.5090
    Epoch 75/3000
    32/32 [==============================] - 0s 4ms/step - loss: 7.6972 - Accuracy: 0.5160
    Epoch 76/3000
    32/32 [==============================] - 0s 4ms/step - loss: 9.0543 - Accuracy: 0.4780
    Epoch 77/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.7032 - Accuracy: 0.4880
    Epoch 78/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.2172 - Accuracy: 0.5050
    Epoch 79/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.1621 - Accuracy: 0.4660
    Epoch 80/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.8909 - Accuracy: 0.5260
    Epoch 81/3000
    32/32 [==============================] - 0s 4ms/step - loss: 7.0365 - Accuracy: 0.5100
    Epoch 82/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.9839 - Accuracy: 0.4920
    Epoch 83/3000
    32/32 [==============================] - 0s 4ms/step - loss: 8.0912 - Accuracy: 0.4560
    Epoch 84/3000
    32/32 [==============================] - 0s 4ms/step - loss: 7.7079 - Accuracy: 0.4840
    Epoch 85/3000
    32/32 [==============================] - 0s 4ms/step - loss: 7.9591 - Accuracy: 0.4690
    Epoch 86/3000
    32/32 [==============================] - 0s 4ms/step - loss: 7.3306 - Accuracy: 0.4580
    Epoch 87/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.8024 - Accuracy: 0.4730
    Epoch 88/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.7852 - Accuracy: 0.4660
    Epoch 89/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.9075 - Accuracy: 0.5110
    Epoch 90/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.6531 - Accuracy: 0.5530
    Epoch 91/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.7756 - Accuracy: 0.4960
    Epoch 92/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.9118 - Accuracy: 0.5180
    Epoch 93/3000
    32/32 [==============================] - 0s 4ms/step - loss: 7.3522 - Accuracy: 0.5530
    Epoch 94/3000
    32/32 [==============================] - 0s 4ms/step - loss: 7.4134 - Accuracy: 0.4780
    Epoch 95/3000
    32/32 [==============================] - 0s 4ms/step - loss: 7.0518 - Accuracy: 0.4800
    Epoch 96/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.2887 - Accuracy: 0.5390
    Epoch 97/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.3640 - Accuracy: 0.5140
    Epoch 98/3000
    32/32 [==============================] - 0s 4ms/step - loss: 7.0284 - Accuracy: 0.4720
    Epoch 99/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.6376 - Accuracy: 0.5310
    Epoch 100/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.5386 - Accuracy: 0.4750
    Epoch 101/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.0714 - Accuracy: 0.5670
    Epoch 102/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.1408 - Accuracy: 0.5290
    Epoch 103/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.9049 - Accuracy: 0.4970
    Epoch 104/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.7164 - Accuracy: 0.5390
    Epoch 105/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.2672 - Accuracy: 0.4660
    Epoch 106/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.5430 - Accuracy: 0.4890
    Epoch 107/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.1097 - Accuracy: 0.4980
    Epoch 108/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.5226 - Accuracy: 0.5150
    Epoch 109/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.4441 - Accuracy: 0.4700
    Epoch 110/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.1138 - Accuracy: 0.4530
    Epoch 111/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.0705 - Accuracy: 0.5260
    Epoch 112/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.1994 - Accuracy: 0.5250
    Epoch 113/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.9702 - Accuracy: 0.5190
    Epoch 114/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.6708 - Accuracy: 0.5370
    Epoch 115/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.9626 - Accuracy: 0.4610
    Epoch 116/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.8271 - Accuracy: 0.5140
    Epoch 117/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.9022 - Accuracy: 0.5010
    Epoch 118/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.7496 - Accuracy: 0.4910
    Epoch 119/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.2607 - Accuracy: 0.5190
    Epoch 120/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.8711 - Accuracy: 0.4780
    Epoch 121/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.9130 - Accuracy: 0.4630
    Epoch 122/3000
    32/32 [==============================] - 0s 4ms/step - loss: 5.3205 - Accuracy: 0.4790
    Epoch 123/3000
    32/32 [==============================] - 0s 4ms/step - loss: 6.1213 - Accuracy: 0.4520
    Epoch 124/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.6821 - Accuracy: 0.4800
    Epoch 125/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.9208 - Accuracy: 0.5170
    Epoch 126/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.9322 - Accuracy: 0.5280
    Epoch 127/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.9654 - Accuracy: 0.5320
    Epoch 128/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.5071 - Accuracy: 0.4920
    Epoch 129/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.0295 - Accuracy: 0.5440
    Epoch 130/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.1607 - Accuracy: 0.5430
    Epoch 131/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.7968 - Accuracy: 0.5110
    Epoch 132/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.9803 - Accuracy: 0.5100
    Epoch 133/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.7937 - Accuracy: 0.5200
    Epoch 134/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.1638 - Accuracy: 0.4770
    Epoch 135/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.1670 - Accuracy: 0.5000
    Epoch 136/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.1633 - Accuracy: 0.5450
    Epoch 137/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.8928 - Accuracy: 0.5230
    Epoch 138/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.0802 - Accuracy: 0.4640
    Epoch 139/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.0157 - Accuracy: 0.5290
    Epoch 140/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.5742 - Accuracy: 0.5230
    Epoch 141/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.6890 - Accuracy: 0.4810
    Epoch 142/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.5206 - Accuracy: 0.5210
    Epoch 143/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.7832 - Accuracy: 0.4710
    Epoch 144/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.8900 - Accuracy: 0.4620
    Epoch 145/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.7798 - Accuracy: 0.5050
    Epoch 146/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.3359 - Accuracy: 0.5350
    Epoch 147/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.1131 - Accuracy: 0.5090
    Epoch 148/3000
    32/32 [==============================] - 0s 4ms/step - loss: 4.2873 - Accuracy: 0.4310
    Epoch 149/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.6114 - Accuracy: 0.4920
    Epoch 150/3000
    32/32 [==============================] - 0s 5ms/step - loss: 3.8422 - Accuracy: 0.4660
    Epoch 151/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.0426 - Accuracy: 0.5110
    Epoch 152/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.8262 - Accuracy: 0.4870
    Epoch 153/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.3370 - Accuracy: 0.5360
    Epoch 154/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.9095 - Accuracy: 0.4920
    Epoch 155/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.9513 - Accuracy: 0.5290
    Epoch 156/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.8973 - Accuracy: 0.5420
    Epoch 157/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.6580 - Accuracy: 0.5120
    Epoch 158/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.5587 - Accuracy: 0.5250
    Epoch 159/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.1948 - Accuracy: 0.5530
    Epoch 160/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.9325 - Accuracy: 0.5050
    Epoch 161/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.3202 - Accuracy: 0.4730
    Epoch 162/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.0023 - Accuracy: 0.4650
    Epoch 163/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.8955 - Accuracy: 0.4410
    Epoch 164/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.2425 - Accuracy: 0.4750
    Epoch 165/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.2445 - Accuracy: 0.4820
    Epoch 166/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.3287 - Accuracy: 0.5000
    Epoch 167/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.6429 - Accuracy: 0.5620
    Epoch 168/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.6861 - Accuracy: 0.5040
    Epoch 169/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.5916 - Accuracy: 0.5170
    Epoch 170/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.9966 - Accuracy: 0.5130
    Epoch 171/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.7146 - Accuracy: 0.5200
    Epoch 172/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.4374 - Accuracy: 0.4450
    Epoch 173/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.7568 - Accuracy: 0.4800
    Epoch 174/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.6526 - Accuracy: 0.5070
    Epoch 175/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.6223 - Accuracy: 0.5050
    Epoch 176/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.6516 - Accuracy: 0.5290
    Epoch 177/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.4910 - Accuracy: 0.5070
    Epoch 178/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.0989 - Accuracy: 0.5060
    Epoch 179/3000
    32/32 [==============================] - 0s 4ms/step - loss: 3.0638 - Accuracy: 0.4640
    Epoch 180/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.7286 - Accuracy: 0.4950
    Epoch 181/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.2779 - Accuracy: 0.5310
    Epoch 182/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.1638 - Accuracy: 0.5550
    Epoch 183/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.9102 - Accuracy: 0.4750
    Epoch 184/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.8655 - Accuracy: 0.4920
    Epoch 185/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.5311 - Accuracy: 0.4790
    Epoch 186/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.6587 - Accuracy: 0.4410
    Epoch 187/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.0358 - Accuracy: 0.5410
    Epoch 188/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.3267 - Accuracy: 0.5290
    Epoch 189/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.0897 - Accuracy: 0.5120
    Epoch 190/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.1751 - Accuracy: 0.5320
    Epoch 191/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.3529 - Accuracy: 0.4930
    Epoch 192/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.4133 - Accuracy: 0.4710
    Epoch 193/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.4018 - Accuracy: 0.4600
    Epoch 194/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.4274 - Accuracy: 0.4950
    Epoch 195/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.0865 - Accuracy: 0.5210
    Epoch 196/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.2509 - Accuracy: 0.4930
    Epoch 197/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.2793 - Accuracy: 0.5140
    Epoch 198/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.1226 - Accuracy: 0.4770
    Epoch 199/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.9962 - Accuracy: 0.5440
    Epoch 200/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.1254 - Accuracy: 0.5270
    Epoch 201/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.1258 - Accuracy: 0.5010
    Epoch 202/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.0161 - Accuracy: 0.5120
    Epoch 203/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.0714 - Accuracy: 0.5170
    Epoch 204/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.3340 - Accuracy: 0.4120
    Epoch 205/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.7725 - Accuracy: 0.5020
    Epoch 206/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.1773 - Accuracy: 0.5180
    Epoch 207/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.8626 - Accuracy: 0.4940
    Epoch 208/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.1549 - Accuracy: 0.5040
    Epoch 209/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4434 - Accuracy: 0.5630
    Epoch 210/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.0767 - Accuracy: 0.4990
    Epoch 211/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.9173 - Accuracy: 0.5230
    Epoch 212/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.9696 - Accuracy: 0.5160
    Epoch 213/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.8362 - Accuracy: 0.5020
    Epoch 214/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.7689 - Accuracy: 0.5250
    Epoch 215/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.8421 - Accuracy: 0.4930
    Epoch 216/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.7954 - Accuracy: 0.5190
    Epoch 217/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.9688 - Accuracy: 0.4550
    Epoch 218/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.9033 - Accuracy: 0.4610
    Epoch 219/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.8554 - Accuracy: 0.5260
    Epoch 220/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.9330 - Accuracy: 0.4920
    Epoch 221/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.0573 - Accuracy: 0.5100
    Epoch 222/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.0538 - Accuracy: 0.4610
    Epoch 223/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.7814 - Accuracy: 0.5040
    Epoch 224/3000
    32/32 [==============================] - 0s 4ms/step - loss: 2.0178 - Accuracy: 0.4510
    Epoch 225/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.8595 - Accuracy: 0.4940
    Epoch 226/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.6034 - Accuracy: 0.5570
    Epoch 227/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.6907 - Accuracy: 0.5160
    Epoch 228/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.6630 - Accuracy: 0.5040
    Epoch 229/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5924 - Accuracy: 0.5700
    Epoch 230/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5749 - Accuracy: 0.5490
    Epoch 231/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.6590 - Accuracy: 0.4490
    Epoch 232/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.7912 - Accuracy: 0.4590
    Epoch 233/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.6217 - Accuracy: 0.4600
    Epoch 234/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.7839 - Accuracy: 0.4990
    Epoch 235/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.9468 - Accuracy: 0.5210
    Epoch 236/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5341 - Accuracy: 0.5040
    Epoch 237/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5315 - Accuracy: 0.5130
    Epoch 238/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4488 - Accuracy: 0.4560
    Epoch 239/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5538 - Accuracy: 0.5060
    Epoch 240/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.7094 - Accuracy: 0.4780
    Epoch 241/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.6103 - Accuracy: 0.5320
    Epoch 242/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.6671 - Accuracy: 0.4450
    Epoch 243/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4969 - Accuracy: 0.5270
    Epoch 244/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5424 - Accuracy: 0.5230
    Epoch 245/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.7602 - Accuracy: 0.4790
    Epoch 246/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5552 - Accuracy: 0.5040
    Epoch 247/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5118 - Accuracy: 0.4940
    Epoch 248/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4379 - Accuracy: 0.5380
    Epoch 249/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4905 - Accuracy: 0.5330
    Epoch 250/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4428 - Accuracy: 0.5300
    Epoch 251/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5172 - Accuracy: 0.5010
    Epoch 252/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.6020 - Accuracy: 0.5030
    Epoch 253/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.7723 - Accuracy: 0.4560
    Epoch 254/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.6407 - Accuracy: 0.4810
    Epoch 255/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5240 - Accuracy: 0.4860
    Epoch 256/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3343 - Accuracy: 0.5230
    Epoch 257/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5275 - Accuracy: 0.5100
    Epoch 258/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4599 - Accuracy: 0.4890
    Epoch 259/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4855 - Accuracy: 0.4470
    Epoch 260/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5332 - Accuracy: 0.5150
    Epoch 261/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4520 - Accuracy: 0.4980
    Epoch 262/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5538 - Accuracy: 0.4980
    Epoch 263/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4813 - Accuracy: 0.3980
    Epoch 264/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3038 - Accuracy: 0.5170
    Epoch 265/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2378 - Accuracy: 0.5020
    Epoch 266/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2953 - Accuracy: 0.5290
    Epoch 267/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4992 - Accuracy: 0.4620
    Epoch 268/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2599 - Accuracy: 0.5400
    Epoch 269/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4107 - Accuracy: 0.4840
    Epoch 270/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5565 - Accuracy: 0.4710
    Epoch 271/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3120 - Accuracy: 0.5480
    Epoch 272/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5311 - Accuracy: 0.4810
    Epoch 273/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3865 - Accuracy: 0.5030
    Epoch 274/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3220 - Accuracy: 0.4820
    Epoch 275/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4702 - Accuracy: 0.4720
    Epoch 276/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.5336 - Accuracy: 0.4540
    Epoch 277/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2054 - Accuracy: 0.5450
    Epoch 278/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4498 - Accuracy: 0.4930
    Epoch 279/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4497 - Accuracy: 0.4850
    Epoch 280/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3096 - Accuracy: 0.4650
    Epoch 281/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4204 - Accuracy: 0.5120
    Epoch 282/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3830 - Accuracy: 0.5290
    Epoch 283/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4269 - Accuracy: 0.4820
    Epoch 284/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3439 - Accuracy: 0.5150
    Epoch 285/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3690 - Accuracy: 0.4920
    Epoch 286/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2324 - Accuracy: 0.5090
    Epoch 287/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1923 - Accuracy: 0.5700
    Epoch 288/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2570 - Accuracy: 0.5250
    Epoch 289/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4186 - Accuracy: 0.4820
    Epoch 290/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2809 - Accuracy: 0.4840
    Epoch 291/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3382 - Accuracy: 0.4750
    Epoch 292/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3083 - Accuracy: 0.4680
    Epoch 293/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4071 - Accuracy: 0.4220
    Epoch 294/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2124 - Accuracy: 0.5390
    Epoch 295/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1899 - Accuracy: 0.5260
    Epoch 296/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2226 - Accuracy: 0.5070
    Epoch 297/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3012 - Accuracy: 0.4830
    Epoch 298/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2079 - Accuracy: 0.5140
    Epoch 299/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3054 - Accuracy: 0.5180
    Epoch 300/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2103 - Accuracy: 0.4960
    Epoch 301/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2887 - Accuracy: 0.4770
    Epoch 302/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2768 - Accuracy: 0.4580
    Epoch 303/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3169 - Accuracy: 0.4980
    Epoch 304/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2353 - Accuracy: 0.5110
    Epoch 305/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2428 - Accuracy: 0.5180
    Epoch 306/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.4223 - Accuracy: 0.4800
    Epoch 307/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2226 - Accuracy: 0.5130
    Epoch 308/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1893 - Accuracy: 0.5040
    Epoch 309/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3332 - Accuracy: 0.4530
    Epoch 310/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3413 - Accuracy: 0.4620
    Epoch 311/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1936 - Accuracy: 0.5010
    Epoch 312/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2011 - Accuracy: 0.5300
    Epoch 313/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1559 - Accuracy: 0.4940
    Epoch 314/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1741 - Accuracy: 0.4970
    Epoch 315/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2033 - Accuracy: 0.4950
    Epoch 316/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3757 - Accuracy: 0.5000
    Epoch 317/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1125 - Accuracy: 0.5490
    Epoch 318/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2714 - Accuracy: 0.5160
    Epoch 319/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2593 - Accuracy: 0.5220
    Epoch 320/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1689 - Accuracy: 0.5120
    Epoch 321/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.3020 - Accuracy: 0.4680
    Epoch 322/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2978 - Accuracy: 0.4960
    Epoch 323/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1682 - Accuracy: 0.4630
    Epoch 324/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1588 - Accuracy: 0.5200
    Epoch 325/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1530 - Accuracy: 0.5480
    Epoch 326/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1618 - Accuracy: 0.4980
    Epoch 327/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1252 - Accuracy: 0.4970
    Epoch 328/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2833 - Accuracy: 0.4950
    Epoch 329/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2245 - Accuracy: 0.5170
    Epoch 330/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2120 - Accuracy: 0.5160
    Epoch 331/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1869 - Accuracy: 0.4660
    Epoch 332/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1542 - Accuracy: 0.5400
    Epoch 333/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1794 - Accuracy: 0.5260
    Epoch 334/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1272 - Accuracy: 0.4870
    Epoch 335/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1848 - Accuracy: 0.5010
    Epoch 336/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1774 - Accuracy: 0.4980
    Epoch 337/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1009 - Accuracy: 0.5240
    Epoch 338/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1462 - Accuracy: 0.4930
    Epoch 339/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1611 - Accuracy: 0.4990
    Epoch 340/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2151 - Accuracy: 0.4960
    Epoch 341/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1765 - Accuracy: 0.4850
    Epoch 342/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1330 - Accuracy: 0.5150
    Epoch 343/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1759 - Accuracy: 0.4910
    Epoch 344/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2139 - Accuracy: 0.4920
    Epoch 345/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1671 - Accuracy: 0.5090
    Epoch 346/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1724 - Accuracy: 0.4880
    Epoch 347/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1970 - Accuracy: 0.5030
    Epoch 348/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2253 - Accuracy: 0.4260
    Epoch 349/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1417 - Accuracy: 0.5160
    Epoch 350/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2138 - Accuracy: 0.4660
    Epoch 351/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1435 - Accuracy: 0.4870
    Epoch 352/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1758 - Accuracy: 0.5340
    Epoch 353/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1452 - Accuracy: 0.4760
    Epoch 354/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1824 - Accuracy: 0.4880
    Epoch 355/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1025 - Accuracy: 0.5040
    Epoch 356/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1046 - Accuracy: 0.5270
    Epoch 357/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2320 - Accuracy: 0.5010
    Epoch 358/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1929 - Accuracy: 0.5390
    Epoch 359/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1713 - Accuracy: 0.4440
    Epoch 360/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1542 - Accuracy: 0.4830
    Epoch 361/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2811 - Accuracy: 0.5000
    Epoch 362/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0991 - Accuracy: 0.5050
    Epoch 363/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1690 - Accuracy: 0.4600
    Epoch 364/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1602 - Accuracy: 0.4830
    Epoch 365/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1181 - Accuracy: 0.5520
    Epoch 366/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1595 - Accuracy: 0.5280
    Epoch 367/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1540 - Accuracy: 0.4700
    Epoch 368/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1237 - Accuracy: 0.5290
    Epoch 369/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1445 - Accuracy: 0.5310
    Epoch 370/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1147 - Accuracy: 0.4890
    Epoch 371/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1311 - Accuracy: 0.5330
    Epoch 372/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1392 - Accuracy: 0.5020
    Epoch 373/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1571 - Accuracy: 0.4670
    Epoch 374/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1635 - Accuracy: 0.4970
    Epoch 375/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0913 - Accuracy: 0.4830
    Epoch 376/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1816 - Accuracy: 0.5040
    Epoch 377/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1780 - Accuracy: 0.4890
    Epoch 378/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1336 - Accuracy: 0.5000
    Epoch 379/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1587 - Accuracy: 0.4980
    Epoch 380/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1990 - Accuracy: 0.4790
    Epoch 381/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1894 - Accuracy: 0.4830
    Epoch 382/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1397 - Accuracy: 0.5070
    Epoch 383/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1188 - Accuracy: 0.5160
    Epoch 384/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1827 - Accuracy: 0.5190
    Epoch 385/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1341 - Accuracy: 0.5070
    Epoch 386/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0888 - Accuracy: 0.5000
    Epoch 387/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1898 - Accuracy: 0.4840
    Epoch 388/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0856 - Accuracy: 0.5490
    Epoch 389/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1962 - Accuracy: 0.4920
    Epoch 390/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1004 - Accuracy: 0.5270
    Epoch 391/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.2023 - Accuracy: 0.4760
    Epoch 392/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1812 - Accuracy: 0.4510
    Epoch 393/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0871 - Accuracy: 0.5080
    Epoch 394/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1232 - Accuracy: 0.5300
    Epoch 395/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0982 - Accuracy: 0.5130
    Epoch 396/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1505 - Accuracy: 0.4960
    Epoch 397/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1236 - Accuracy: 0.4780
    Epoch 398/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0963 - Accuracy: 0.5110
    Epoch 399/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1909 - Accuracy: 0.4660
    Epoch 400/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1111 - Accuracy: 0.4940
    Epoch 401/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1025 - Accuracy: 0.5030
    Epoch 402/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0997 - Accuracy: 0.5190
    Epoch 403/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1837 - Accuracy: 0.4700
    Epoch 404/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1372 - Accuracy: 0.5150
    Epoch 405/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1410 - Accuracy: 0.4960
    Epoch 406/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1133 - Accuracy: 0.5430
    Epoch 407/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1142 - Accuracy: 0.4820
    Epoch 408/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1002 - Accuracy: 0.5020
    Epoch 409/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1461 - Accuracy: 0.4960
    Epoch 410/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1170 - Accuracy: 0.5030
    Epoch 411/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1039 - Accuracy: 0.5440
    Epoch 412/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1276 - Accuracy: 0.4850
    Epoch 413/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1455 - Accuracy: 0.5070
    Epoch 414/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0575 - Accuracy: 0.5400
    Epoch 415/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1205 - Accuracy: 0.5000
    Epoch 416/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1413 - Accuracy: 0.5020
    Epoch 417/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0972 - Accuracy: 0.4930
    Epoch 418/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0816 - Accuracy: 0.5290
    Epoch 419/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0903 - Accuracy: 0.4980
    Epoch 420/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1645 - Accuracy: 0.4470
    Epoch 421/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1252 - Accuracy: 0.5350
    Epoch 422/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1000 - Accuracy: 0.5220
    Epoch 423/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1045 - Accuracy: 0.4960
    Epoch 424/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1249 - Accuracy: 0.4910
    Epoch 425/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0983 - Accuracy: 0.4920
    Epoch 426/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1153 - Accuracy: 0.5130
    Epoch 427/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0924 - Accuracy: 0.4740
    Epoch 428/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1032 - Accuracy: 0.4990
    Epoch 429/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1030 - Accuracy: 0.5230
    Epoch 430/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0536 - Accuracy: 0.5400
    Epoch 431/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1141 - Accuracy: 0.4600
    Epoch 432/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0951 - Accuracy: 0.5350
    Epoch 433/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1265 - Accuracy: 0.4870
    Epoch 434/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1066 - Accuracy: 0.4970
    Epoch 435/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0782 - Accuracy: 0.5060
    Epoch 436/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0623 - Accuracy: 0.4950
    Epoch 437/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1188 - Accuracy: 0.5270
    Epoch 438/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0943 - Accuracy: 0.4870
    Epoch 439/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0731 - Accuracy: 0.4790
    Epoch 440/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1502 - Accuracy: 0.4830
    Epoch 441/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0987 - Accuracy: 0.5240
    Epoch 442/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1344 - Accuracy: 0.4840
    Epoch 443/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0834 - Accuracy: 0.4990
    Epoch 444/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0726 - Accuracy: 0.5260
    Epoch 445/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1007 - Accuracy: 0.5160
    Epoch 446/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0687 - Accuracy: 0.5030
    Epoch 447/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1573 - Accuracy: 0.5130
    Epoch 448/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1595 - Accuracy: 0.5260
    Epoch 449/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0985 - Accuracy: 0.5150
    Epoch 450/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0549 - Accuracy: 0.5120
    Epoch 451/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0890 - Accuracy: 0.4810
    Epoch 452/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0783 - Accuracy: 0.4880
    Epoch 453/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0824 - Accuracy: 0.4750
    Epoch 454/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0863 - Accuracy: 0.4940
    Epoch 455/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0631 - Accuracy: 0.5340
    Epoch 456/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0936 - Accuracy: 0.4760
    Epoch 457/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0569 - Accuracy: 0.5020
    Epoch 458/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1034 - Accuracy: 0.4910
    Epoch 459/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0604 - Accuracy: 0.5490
    Epoch 460/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1322 - Accuracy: 0.5180
    Epoch 461/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0461 - Accuracy: 0.5350
    Epoch 462/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0748 - Accuracy: 0.5220
    Epoch 463/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1057 - Accuracy: 0.4950
    Epoch 464/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0415 - Accuracy: 0.5220
    Epoch 465/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0919 - Accuracy: 0.5410
    Epoch 466/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0929 - Accuracy: 0.4970
    Epoch 467/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0633 - Accuracy: 0.5000
    Epoch 468/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1330 - Accuracy: 0.5000
    Epoch 469/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0608 - Accuracy: 0.4940
    Epoch 470/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0755 - Accuracy: 0.5090
    Epoch 471/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0904 - Accuracy: 0.5360
    Epoch 472/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0695 - Accuracy: 0.5330
    Epoch 473/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0345 - Accuracy: 0.5080
    Epoch 474/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0795 - Accuracy: 0.5020
    Epoch 475/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0644 - Accuracy: 0.5060
    Epoch 476/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0569 - Accuracy: 0.5140
    Epoch 477/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0821 - Accuracy: 0.4830
    Epoch 478/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0656 - Accuracy: 0.5040
    Epoch 479/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0879 - Accuracy: 0.4760
    Epoch 480/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1840 - Accuracy: 0.4890
    Epoch 481/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1774 - Accuracy: 0.4730
    Epoch 482/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0520 - Accuracy: 0.4750
    Epoch 483/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0610 - Accuracy: 0.4810
    Epoch 484/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0877 - Accuracy: 0.4930
    Epoch 485/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0460 - Accuracy: 0.5280
    Epoch 486/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0390 - Accuracy: 0.4750
    Epoch 487/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0747 - Accuracy: 0.4850
    Epoch 488/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0693 - Accuracy: 0.5050
    Epoch 489/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0497 - Accuracy: 0.5250
    Epoch 490/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0574 - Accuracy: 0.5190
    Epoch 491/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1626 - Accuracy: 0.5270
    Epoch 492/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0653 - Accuracy: 0.4810
    Epoch 493/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0407 - Accuracy: 0.5240
    Epoch 494/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0908 - Accuracy: 0.4840
    Epoch 495/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1181 - Accuracy: 0.4810
    Epoch 496/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0755 - Accuracy: 0.4580
    Epoch 497/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0544 - Accuracy: 0.4710
    Epoch 498/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0841 - Accuracy: 0.5210
    Epoch 499/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0572 - Accuracy: 0.5010
    Epoch 500/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0759 - Accuracy: 0.5230
    Epoch 501/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0118 - Accuracy: 0.5350
    Epoch 502/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0411 - Accuracy: 0.5390
    Epoch 503/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0901 - Accuracy: 0.4830
    Epoch 504/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1054 - Accuracy: 0.5120
    Epoch 505/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0872 - Accuracy: 0.5090
    Epoch 506/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0612 - Accuracy: 0.4830
    Epoch 507/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0645 - Accuracy: 0.4830
    Epoch 508/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1174 - Accuracy: 0.4630
    Epoch 509/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0445 - Accuracy: 0.4980
    Epoch 510/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0571 - Accuracy: 0.5220
    Epoch 511/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0374 - Accuracy: 0.4910
    Epoch 512/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0336 - Accuracy: 0.4810
    Epoch 513/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0571 - Accuracy: 0.4950
    Epoch 514/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0183 - Accuracy: 0.5360
    Epoch 515/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0457 - Accuracy: 0.4750
    Epoch 516/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1000 - Accuracy: 0.4960
    Epoch 517/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0541 - Accuracy: 0.4740
    Epoch 518/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1088 - Accuracy: 0.5000
    Epoch 519/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9995 - Accuracy: 0.5330
    Epoch 520/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1161 - Accuracy: 0.4910
    Epoch 521/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1013 - Accuracy: 0.4990
    Epoch 522/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0401 - Accuracy: 0.5200
    Epoch 523/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0163 - Accuracy: 0.4810
    Epoch 524/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0120 - Accuracy: 0.5010
    Epoch 525/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0793 - Accuracy: 0.5190
    Epoch 526/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0528 - Accuracy: 0.5500
    Epoch 527/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0141 - Accuracy: 0.4970
    Epoch 528/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0460 - Accuracy: 0.5000
    Epoch 529/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0219 - Accuracy: 0.5010
    Epoch 530/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0525 - Accuracy: 0.5220
    Epoch 531/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0529 - Accuracy: 0.5270
    Epoch 532/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1077 - Accuracy: 0.4950
    Epoch 533/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0211 - Accuracy: 0.5000
    Epoch 534/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0022 - Accuracy: 0.5310
    Epoch 535/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0723 - Accuracy: 0.5020
    Epoch 536/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0438 - Accuracy: 0.4990
    Epoch 537/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0498 - Accuracy: 0.5270
    Epoch 538/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9824 - Accuracy: 0.5400
    Epoch 539/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0156 - Accuracy: 0.5090
    Epoch 540/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0104 - Accuracy: 0.5120
    Epoch 541/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1261 - Accuracy: 0.4960
    Epoch 542/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0787 - Accuracy: 0.4830
    Epoch 543/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0174 - Accuracy: 0.5280
    Epoch 544/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0630 - Accuracy: 0.4990
    Epoch 545/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0320 - Accuracy: 0.4750
    Epoch 546/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0451 - Accuracy: 0.5360
    Epoch 547/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0546 - Accuracy: 0.5070
    Epoch 548/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0797 - Accuracy: 0.4950
    Epoch 549/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0904 - Accuracy: 0.4820
    Epoch 550/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0669 - Accuracy: 0.5050
    Epoch 551/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0103 - Accuracy: 0.4940
    Epoch 552/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9963 - Accuracy: 0.5290
    Epoch 553/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0284 - Accuracy: 0.4980
    Epoch 554/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0295 - Accuracy: 0.5120
    Epoch 555/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0216 - Accuracy: 0.4960
    Epoch 556/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9974 - Accuracy: 0.4980
    Epoch 557/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0188 - Accuracy: 0.5060
    Epoch 558/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0158 - Accuracy: 0.5150
    Epoch 559/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0651 - Accuracy: 0.5450
    Epoch 560/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0287 - Accuracy: 0.5350
    Epoch 561/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0442 - Accuracy: 0.4700
    Epoch 562/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0999 - Accuracy: 0.5040
    Epoch 563/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0149 - Accuracy: 0.5180
    Epoch 564/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9947 - Accuracy: 0.5160
    Epoch 565/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9940 - Accuracy: 0.5080
    Epoch 566/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1641 - Accuracy: 0.4780
    Epoch 567/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0290 - Accuracy: 0.5460
    Epoch 568/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0229 - Accuracy: 0.5060
    Epoch 569/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9729 - Accuracy: 0.5100
    Epoch 570/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0366 - Accuracy: 0.5320
    Epoch 571/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0229 - Accuracy: 0.5100
    Epoch 572/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0191 - Accuracy: 0.5330
    Epoch 573/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9854 - Accuracy: 0.5430
    Epoch 574/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0176 - Accuracy: 0.4870
    Epoch 575/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0246 - Accuracy: 0.5070
    Epoch 576/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0443 - Accuracy: 0.5420
    Epoch 577/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0198 - Accuracy: 0.4780
    Epoch 578/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9939 - Accuracy: 0.5240
    Epoch 579/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0129 - Accuracy: 0.4810
    Epoch 580/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0145 - Accuracy: 0.5290
    Epoch 581/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0650 - Accuracy: 0.4980
    Epoch 582/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0114 - Accuracy: 0.5020
    Epoch 583/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0220 - Accuracy: 0.4990
    Epoch 584/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0052 - Accuracy: 0.4970
    Epoch 585/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0898 - Accuracy: 0.4980
    Epoch 586/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0070 - Accuracy: 0.5020
    Epoch 587/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0236 - Accuracy: 0.5040
    Epoch 588/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0189 - Accuracy: 0.4970
    Epoch 589/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0181 - Accuracy: 0.5090
    Epoch 590/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0555 - Accuracy: 0.4720
    Epoch 591/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0143 - Accuracy: 0.4960
    Epoch 592/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9815 - Accuracy: 0.5300
    Epoch 593/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0082 - Accuracy: 0.5090
    Epoch 594/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0014 - Accuracy: 0.5070
    Epoch 595/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0007 - Accuracy: 0.5380
    Epoch 596/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0543 - Accuracy: 0.4950
    Epoch 597/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9855 - Accuracy: 0.5150
    Epoch 598/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9800 - Accuracy: 0.5290
    Epoch 599/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9885 - Accuracy: 0.4930
    Epoch 600/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0969 - Accuracy: 0.4780
    Epoch 601/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0484 - Accuracy: 0.4860
    Epoch 602/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9538 - Accuracy: 0.5440
    Epoch 603/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9933 - Accuracy: 0.4940
    Epoch 604/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0892 - Accuracy: 0.5140
    Epoch 605/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0193 - Accuracy: 0.5080
    Epoch 606/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0263 - Accuracy: 0.4940
    Epoch 607/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0116 - Accuracy: 0.5140
    Epoch 608/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9971 - Accuracy: 0.5030
    Epoch 609/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0077 - Accuracy: 0.4870
    Epoch 610/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9786 - Accuracy: 0.5340
    Epoch 611/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0291 - Accuracy: 0.5090
    Epoch 612/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1716 - Accuracy: 0.4960
    Epoch 613/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0134 - Accuracy: 0.4830
    Epoch 614/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0047 - Accuracy: 0.5300
    Epoch 615/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0114 - Accuracy: 0.4880
    Epoch 616/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9853 - Accuracy: 0.5280
    Epoch 617/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0459 - Accuracy: 0.5010
    Epoch 618/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9593 - Accuracy: 0.4960
    Epoch 619/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1627 - Accuracy: 0.4790
    Epoch 620/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9997 - Accuracy: 0.5130
    Epoch 621/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9493 - Accuracy: 0.5010
    Epoch 622/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0016 - Accuracy: 0.5230
    Epoch 623/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0087 - Accuracy: 0.5260
    Epoch 624/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0883 - Accuracy: 0.4890
    Epoch 625/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9824 - Accuracy: 0.5430
    Epoch 626/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9700 - Accuracy: 0.5010
    Epoch 627/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0502 - Accuracy: 0.4730
    Epoch 628/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9491 - Accuracy: 0.4900
    Epoch 629/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0434 - Accuracy: 0.5130
    Epoch 630/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9775 - Accuracy: 0.5290
    Epoch 631/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9724 - Accuracy: 0.5370
    Epoch 632/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9782 - Accuracy: 0.5320
    Epoch 633/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0344 - Accuracy: 0.4800
    Epoch 634/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0139 - Accuracy: 0.4950
    Epoch 635/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9743 - Accuracy: 0.5020
    Epoch 636/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0176 - Accuracy: 0.4910
    Epoch 637/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0413 - Accuracy: 0.4820
    Epoch 638/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9729 - Accuracy: 0.5000
    Epoch 639/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9701 - Accuracy: 0.4910
    Epoch 640/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9610 - Accuracy: 0.5310
    Epoch 641/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9990 - Accuracy: 0.5030
    Epoch 642/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0374 - Accuracy: 0.4910
    Epoch 643/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0594 - Accuracy: 0.4890
    Epoch 644/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9702 - Accuracy: 0.5190
    Epoch 645/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9952 - Accuracy: 0.5290
    Epoch 646/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9573 - Accuracy: 0.4800
    Epoch 647/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9814 - Accuracy: 0.5240
    Epoch 648/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9859 - Accuracy: 0.5120
    Epoch 649/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9805 - Accuracy: 0.5430
    Epoch 650/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0566 - Accuracy: 0.4710
    Epoch 651/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9537 - Accuracy: 0.5190
    Epoch 652/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9868 - Accuracy: 0.4990
    Epoch 653/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0026 - Accuracy: 0.5230
    Epoch 654/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9918 - Accuracy: 0.4960
    Epoch 655/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9709 - Accuracy: 0.5070
    Epoch 656/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1245 - Accuracy: 0.5220
    Epoch 657/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9953 - Accuracy: 0.5060
    Epoch 658/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1897 - Accuracy: 0.5120
    Epoch 659/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0193 - Accuracy: 0.4510
    Epoch 660/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0211 - Accuracy: 0.5360
    Epoch 661/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9489 - Accuracy: 0.5240
    Epoch 662/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0269 - Accuracy: 0.4770
    Epoch 663/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9505 - Accuracy: 0.5320
    Epoch 664/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9585 - Accuracy: 0.5380
    Epoch 665/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0297 - Accuracy: 0.4770
    Epoch 666/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0313 - Accuracy: 0.4990
    Epoch 667/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0461 - Accuracy: 0.5080
    Epoch 668/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9758 - Accuracy: 0.5060
    Epoch 669/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9643 - Accuracy: 0.5280
    Epoch 670/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9713 - Accuracy: 0.5130
    Epoch 671/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9609 - Accuracy: 0.4790
    Epoch 672/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9929 - Accuracy: 0.5040
    Epoch 673/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9668 - Accuracy: 0.5050
    Epoch 674/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9672 - Accuracy: 0.4810
    Epoch 675/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9730 - Accuracy: 0.5130
    Epoch 676/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9916 - Accuracy: 0.4930
    Epoch 677/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9476 - Accuracy: 0.5490
    Epoch 678/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9537 - Accuracy: 0.5100
    Epoch 679/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9937 - Accuracy: 0.5160
    Epoch 680/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9481 - Accuracy: 0.4980
    Epoch 681/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0107 - Accuracy: 0.5250
    Epoch 682/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0179 - Accuracy: 0.4630
    Epoch 683/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9689 - Accuracy: 0.5150
    Epoch 684/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0304 - Accuracy: 0.4890
    Epoch 685/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9776 - Accuracy: 0.5260
    Epoch 686/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0144 - Accuracy: 0.5040
    Epoch 687/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9843 - Accuracy: 0.4660
    Epoch 688/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9636 - Accuracy: 0.4770
    Epoch 689/3000
    32/32 [==============================] - 0s 5ms/step - loss: 0.9945 - Accuracy: 0.5020
    Epoch 690/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0127 - Accuracy: 0.4720
    Epoch 691/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9595 - Accuracy: 0.4600
    Epoch 692/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9874 - Accuracy: 0.4850
    Epoch 693/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0185 - Accuracy: 0.5060
    Epoch 694/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9260 - Accuracy: 0.5550
    Epoch 695/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0197 - Accuracy: 0.5110
    Epoch 696/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0538 - Accuracy: 0.5180
    Epoch 697/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9748 - Accuracy: 0.5140
    Epoch 698/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9183 - Accuracy: 0.5180
    Epoch 699/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9554 - Accuracy: 0.5090
    Epoch 700/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9496 - Accuracy: 0.5160
    Epoch 701/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0273 - Accuracy: 0.5320
    Epoch 702/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9532 - Accuracy: 0.4710
    Epoch 703/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0143 - Accuracy: 0.5160
    Epoch 704/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9994 - Accuracy: 0.5230
    Epoch 705/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9601 - Accuracy: 0.5200
    Epoch 706/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9495 - Accuracy: 0.4920
    Epoch 707/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0717 - Accuracy: 0.4920
    Epoch 708/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0182 - Accuracy: 0.4940
    Epoch 709/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9641 - Accuracy: 0.4890
    Epoch 710/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0028 - Accuracy: 0.4650
    Epoch 711/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9759 - Accuracy: 0.5010
    Epoch 712/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9752 - Accuracy: 0.4880
    Epoch 713/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9876 - Accuracy: 0.5000
    Epoch 714/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0096 - Accuracy: 0.4960
    Epoch 715/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9682 - Accuracy: 0.5160
    Epoch 716/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9812 - Accuracy: 0.4950
    Epoch 717/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9989 - Accuracy: 0.5320
    Epoch 718/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9708 - Accuracy: 0.4910
    Epoch 719/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9582 - Accuracy: 0.5170
    Epoch 720/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9624 - Accuracy: 0.5110
    Epoch 721/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9387 - Accuracy: 0.5290
    Epoch 722/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9340 - Accuracy: 0.5540
    Epoch 723/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9176 - Accuracy: 0.5190
    Epoch 724/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9634 - Accuracy: 0.5510
    Epoch 725/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9339 - Accuracy: 0.4980
    Epoch 726/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9504 - Accuracy: 0.5050
    Epoch 727/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9635 - Accuracy: 0.4750
    Epoch 728/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9555 - Accuracy: 0.5080
    Epoch 729/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9347 - Accuracy: 0.5250
    Epoch 730/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9804 - Accuracy: 0.5050
    Epoch 731/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9824 - Accuracy: 0.5040
    Epoch 732/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9971 - Accuracy: 0.5090
    Epoch 733/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9545 - Accuracy: 0.4890
    Epoch 734/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9086 - Accuracy: 0.5550
    Epoch 735/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9448 - Accuracy: 0.4940
    Epoch 736/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0045 - Accuracy: 0.4760
    Epoch 737/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0044 - Accuracy: 0.5030
    Epoch 738/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9498 - Accuracy: 0.5040
    Epoch 739/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9375 - Accuracy: 0.5290
    Epoch 740/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0106 - Accuracy: 0.5130
    Epoch 741/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9903 - Accuracy: 0.5260
    Epoch 742/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9848 - Accuracy: 0.4970
    Epoch 743/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9333 - Accuracy: 0.4900
    Epoch 744/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9058 - Accuracy: 0.5400
    Epoch 745/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9383 - Accuracy: 0.5150
    Epoch 746/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9268 - Accuracy: 0.5280
    Epoch 747/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9494 - Accuracy: 0.5230
    Epoch 748/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9795 - Accuracy: 0.4920
    Epoch 749/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9208 - Accuracy: 0.5310
    Epoch 750/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9479 - Accuracy: 0.5330
    Epoch 751/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9253 - Accuracy: 0.5450
    Epoch 752/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9982 - Accuracy: 0.4820
    Epoch 753/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9310 - Accuracy: 0.5300
    Epoch 754/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9546 - Accuracy: 0.5120
    Epoch 755/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9403 - Accuracy: 0.5320
    Epoch 756/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9972 - Accuracy: 0.5100
    Epoch 757/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9903 - Accuracy: 0.5110
    Epoch 758/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9396 - Accuracy: 0.5140
    Epoch 759/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9542 - Accuracy: 0.4970
    Epoch 760/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9224 - Accuracy: 0.5230
    Epoch 761/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9532 - Accuracy: 0.5140
    Epoch 762/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9339 - Accuracy: 0.4970
    Epoch 763/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9431 - Accuracy: 0.5300
    Epoch 764/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9052 - Accuracy: 0.5110
    Epoch 765/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9746 - Accuracy: 0.5100
    Epoch 766/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0634 - Accuracy: 0.4940
    Epoch 767/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9252 - Accuracy: 0.5290
    Epoch 768/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9522 - Accuracy: 0.5220
    Epoch 769/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8939 - Accuracy: 0.5400
    Epoch 770/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9026 - Accuracy: 0.5250
    Epoch 771/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9672 - Accuracy: 0.5330
    Epoch 772/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9156 - Accuracy: 0.5000
    Epoch 773/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9261 - Accuracy: 0.5100
    Epoch 774/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9640 - Accuracy: 0.4780
    Epoch 775/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0087 - Accuracy: 0.5090
    Epoch 776/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9190 - Accuracy: 0.5020
    Epoch 777/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9087 - Accuracy: 0.5050
    Epoch 778/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9773 - Accuracy: 0.5220
    Epoch 779/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9481 - Accuracy: 0.5220
    Epoch 780/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9314 - Accuracy: 0.5160
    Epoch 781/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9239 - Accuracy: 0.5000
    Epoch 782/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9536 - Accuracy: 0.5150
    Epoch 783/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9289 - Accuracy: 0.5010
    Epoch 784/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9201 - Accuracy: 0.5120
    Epoch 785/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9488 - Accuracy: 0.4800
    Epoch 786/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9621 - Accuracy: 0.4800
    Epoch 787/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9290 - Accuracy: 0.5010
    Epoch 788/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9423 - Accuracy: 0.5220
    Epoch 789/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9731 - Accuracy: 0.5050
    Epoch 790/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9162 - Accuracy: 0.4940
    Epoch 791/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9406 - Accuracy: 0.5210
    Epoch 792/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9198 - Accuracy: 0.5230
    Epoch 793/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9144 - Accuracy: 0.5260
    Epoch 794/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9025 - Accuracy: 0.5240
    Epoch 795/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8919 - Accuracy: 0.5580
    Epoch 796/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9687 - Accuracy: 0.4750
    Epoch 797/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9186 - Accuracy: 0.5090
    Epoch 798/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9178 - Accuracy: 0.5310
    Epoch 799/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0007 - Accuracy: 0.5040
    Epoch 800/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9184 - Accuracy: 0.5260
    Epoch 801/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9183 - Accuracy: 0.5370
    Epoch 802/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9942 - Accuracy: 0.5130
    Epoch 803/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0550 - Accuracy: 0.4860
    Epoch 804/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0855 - Accuracy: 0.4780
    Epoch 805/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9376 - Accuracy: 0.5150
    Epoch 806/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9065 - Accuracy: 0.5200
    Epoch 807/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9376 - Accuracy: 0.5050
    Epoch 808/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9038 - Accuracy: 0.5120
    Epoch 809/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8876 - Accuracy: 0.5520
    Epoch 810/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0658 - Accuracy: 0.5150
    Epoch 811/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9178 - Accuracy: 0.5200
    Epoch 812/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9880 - Accuracy: 0.5030
    Epoch 813/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9536 - Accuracy: 0.4770
    Epoch 814/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0099 - Accuracy: 0.5090
    Epoch 815/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8977 - Accuracy: 0.5320
    Epoch 816/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0205 - Accuracy: 0.4580
    Epoch 817/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9323 - Accuracy: 0.4990
    Epoch 818/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0732 - Accuracy: 0.4650
    Epoch 819/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9385 - Accuracy: 0.4860
    Epoch 820/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9256 - Accuracy: 0.4950
    Epoch 821/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9722 - Accuracy: 0.5240
    Epoch 822/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9626 - Accuracy: 0.4790
    Epoch 823/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8948 - Accuracy: 0.4960
    Epoch 824/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9511 - Accuracy: 0.4900
    Epoch 825/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9490 - Accuracy: 0.5210
    Epoch 826/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9145 - Accuracy: 0.5260
    Epoch 827/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9233 - Accuracy: 0.4840
    Epoch 828/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9516 - Accuracy: 0.5170
    Epoch 829/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9091 - Accuracy: 0.4890
    Epoch 830/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9121 - Accuracy: 0.5160
    Epoch 831/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9237 - Accuracy: 0.4660
    Epoch 832/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8979 - Accuracy: 0.4950
    Epoch 833/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0277 - Accuracy: 0.4950
    Epoch 834/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9029 - Accuracy: 0.5100
    Epoch 835/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9716 - Accuracy: 0.5000
    Epoch 836/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0505 - Accuracy: 0.5090
    Epoch 837/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9228 - Accuracy: 0.5010
    Epoch 838/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9229 - Accuracy: 0.5010
    Epoch 839/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9095 - Accuracy: 0.4890
    Epoch 840/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9667 - Accuracy: 0.4790
    Epoch 841/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9846 - Accuracy: 0.4720
    Epoch 842/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8842 - Accuracy: 0.5230
    Epoch 843/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8951 - Accuracy: 0.5060
    Epoch 844/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9155 - Accuracy: 0.4920
    Epoch 845/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9534 - Accuracy: 0.4490
    Epoch 846/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9338 - Accuracy: 0.4670
    Epoch 847/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9557 - Accuracy: 0.4770
    Epoch 848/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9271 - Accuracy: 0.4890
    Epoch 849/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9390 - Accuracy: 0.4980
    Epoch 850/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8952 - Accuracy: 0.5050
    Epoch 851/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8966 - Accuracy: 0.4910
    Epoch 852/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9561 - Accuracy: 0.4980
    Epoch 853/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9408 - Accuracy: 0.4900
    Epoch 854/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9149 - Accuracy: 0.5080
    Epoch 855/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9214 - Accuracy: 0.4890
    Epoch 856/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9619 - Accuracy: 0.5170
    Epoch 857/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9290 - Accuracy: 0.5090
    Epoch 858/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9515 - Accuracy: 0.4920
    Epoch 859/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8902 - Accuracy: 0.5120
    Epoch 860/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9109 - Accuracy: 0.5450
    Epoch 861/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9092 - Accuracy: 0.5160
    Epoch 862/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9256 - Accuracy: 0.5000
    Epoch 863/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9686 - Accuracy: 0.5280
    Epoch 864/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9146 - Accuracy: 0.4710
    Epoch 865/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9503 - Accuracy: 0.5090
    Epoch 866/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8980 - Accuracy: 0.5120
    Epoch 867/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9144 - Accuracy: 0.5250
    Epoch 868/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9396 - Accuracy: 0.4530
    Epoch 869/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9120 - Accuracy: 0.5150
    Epoch 870/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1152 - Accuracy: 0.5040
    Epoch 871/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0112 - Accuracy: 0.4860
    Epoch 872/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9257 - Accuracy: 0.5020
    Epoch 873/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9315 - Accuracy: 0.4940
    Epoch 874/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9065 - Accuracy: 0.5210
    Epoch 875/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9176 - Accuracy: 0.5310
    Epoch 876/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9047 - Accuracy: 0.4850
    Epoch 877/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9514 - Accuracy: 0.4930
    Epoch 878/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9695 - Accuracy: 0.4990
    Epoch 879/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8843 - Accuracy: 0.5170
    Epoch 880/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8947 - Accuracy: 0.5030
    Epoch 881/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9330 - Accuracy: 0.4910
    Epoch 882/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9937 - Accuracy: 0.4900
    Epoch 883/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9784 - Accuracy: 0.5000
    Epoch 884/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9603 - Accuracy: 0.5160
    Epoch 885/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8634 - Accuracy: 0.5400
    Epoch 886/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8998 - Accuracy: 0.4980
    Epoch 887/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9317 - Accuracy: 0.5260
    Epoch 888/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9302 - Accuracy: 0.4980
    Epoch 889/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8677 - Accuracy: 0.5280
    Epoch 890/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8985 - Accuracy: 0.4860
    Epoch 891/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8836 - Accuracy: 0.5400
    Epoch 892/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9316 - Accuracy: 0.5210
    Epoch 893/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9209 - Accuracy: 0.4680
    Epoch 894/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9392 - Accuracy: 0.4760
    Epoch 895/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8847 - Accuracy: 0.5270
    Epoch 896/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8866 - Accuracy: 0.5340
    Epoch 897/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8802 - Accuracy: 0.5060
    Epoch 898/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9226 - Accuracy: 0.5140
    Epoch 899/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8990 - Accuracy: 0.5110
    Epoch 900/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9754 - Accuracy: 0.4730
    Epoch 901/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9059 - Accuracy: 0.4760
    Epoch 902/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8950 - Accuracy: 0.5360
    Epoch 903/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9020 - Accuracy: 0.5020
    Epoch 904/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9005 - Accuracy: 0.5020
    Epoch 905/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9087 - Accuracy: 0.5210
    Epoch 906/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8987 - Accuracy: 0.5030
    Epoch 907/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9082 - Accuracy: 0.4740
    Epoch 908/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9521 - Accuracy: 0.4910
    Epoch 909/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8826 - Accuracy: 0.5150
    Epoch 910/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9584 - Accuracy: 0.5260
    Epoch 911/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9280 - Accuracy: 0.5290
    Epoch 912/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8659 - Accuracy: 0.5520
    Epoch 913/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9445 - Accuracy: 0.4790
    Epoch 914/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9334 - Accuracy: 0.4890
    Epoch 915/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9788 - Accuracy: 0.4850
    Epoch 916/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9195 - Accuracy: 0.5060
    Epoch 917/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8855 - Accuracy: 0.5160
    Epoch 918/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9447 - Accuracy: 0.5040
    Epoch 919/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9346 - Accuracy: 0.4840
    Epoch 920/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8907 - Accuracy: 0.5190
    Epoch 921/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9073 - Accuracy: 0.5200
    Epoch 922/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9298 - Accuracy: 0.4690
    Epoch 923/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9073 - Accuracy: 0.4980
    Epoch 924/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9287 - Accuracy: 0.4800
    Epoch 925/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9516 - Accuracy: 0.5080
    Epoch 926/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9530 - Accuracy: 0.5050
    Epoch 927/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8897 - Accuracy: 0.5010
    Epoch 928/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8834 - Accuracy: 0.5470
    Epoch 929/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8955 - Accuracy: 0.5140
    Epoch 930/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8826 - Accuracy: 0.4960
    Epoch 931/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9045 - Accuracy: 0.5100
    Epoch 932/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8927 - Accuracy: 0.5000
    Epoch 933/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8746 - Accuracy: 0.5240
    Epoch 934/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9008 - Accuracy: 0.5000
    Epoch 935/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9321 - Accuracy: 0.5270
    Epoch 936/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1385 - Accuracy: 0.4400
    Epoch 937/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8604 - Accuracy: 0.5220
    Epoch 938/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9132 - Accuracy: 0.5150
    Epoch 939/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9278 - Accuracy: 0.4910
    Epoch 940/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8668 - Accuracy: 0.5290
    Epoch 941/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9305 - Accuracy: 0.5570
    Epoch 942/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9659 - Accuracy: 0.5150
    Epoch 943/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8890 - Accuracy: 0.4730
    Epoch 944/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8848 - Accuracy: 0.5060
    Epoch 945/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9428 - Accuracy: 0.5410
    Epoch 946/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9047 - Accuracy: 0.4880
    Epoch 947/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8952 - Accuracy: 0.4940
    Epoch 948/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9193 - Accuracy: 0.4960
    Epoch 949/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9266 - Accuracy: 0.4820
    Epoch 950/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0484 - Accuracy: 0.5000
    Epoch 951/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9014 - Accuracy: 0.5090
    Epoch 952/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8634 - Accuracy: 0.5130
    Epoch 953/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8767 - Accuracy: 0.5130
    Epoch 954/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8664 - Accuracy: 0.5360
    Epoch 955/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8987 - Accuracy: 0.5170
    Epoch 956/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8986 - Accuracy: 0.4970
    Epoch 957/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0565 - Accuracy: 0.4930
    Epoch 958/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9058 - Accuracy: 0.4980
    Epoch 959/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9041 - Accuracy: 0.4970
    Epoch 960/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9067 - Accuracy: 0.4850
    Epoch 961/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8729 - Accuracy: 0.5000
    Epoch 962/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8985 - Accuracy: 0.4920
    Epoch 963/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8859 - Accuracy: 0.5050
    Epoch 964/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8921 - Accuracy: 0.5150
    Epoch 965/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8786 - Accuracy: 0.5320
    Epoch 966/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8914 - Accuracy: 0.4800
    Epoch 967/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9097 - Accuracy: 0.5160
    Epoch 968/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8935 - Accuracy: 0.5030
    Epoch 969/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8911 - Accuracy: 0.4980
    Epoch 970/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1395 - Accuracy: 0.4780
    Epoch 971/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8912 - Accuracy: 0.5070
    Epoch 972/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9315 - Accuracy: 0.5250
    Epoch 973/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8918 - Accuracy: 0.5460
    Epoch 974/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8708 - Accuracy: 0.4930
    Epoch 975/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9168 - Accuracy: 0.4900
    Epoch 976/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8640 - Accuracy: 0.5180
    Epoch 977/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9143 - Accuracy: 0.5390
    Epoch 978/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8737 - Accuracy: 0.5060
    Epoch 979/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9060 - Accuracy: 0.4730
    Epoch 980/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8732 - Accuracy: 0.5430
    Epoch 981/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9011 - Accuracy: 0.5390
    Epoch 982/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8637 - Accuracy: 0.5060
    Epoch 983/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8800 - Accuracy: 0.5040
    Epoch 984/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8626 - Accuracy: 0.5150
    Epoch 985/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8883 - Accuracy: 0.5340
    Epoch 986/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9517 - Accuracy: 0.4960
    Epoch 987/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8770 - Accuracy: 0.5130
    Epoch 988/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8787 - Accuracy: 0.4900
    Epoch 989/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8441 - Accuracy: 0.5440
    Epoch 990/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8728 - Accuracy: 0.5530
    Epoch 991/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9555 - Accuracy: 0.5240
    Epoch 992/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8721 - Accuracy: 0.5380
    Epoch 993/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9067 - Accuracy: 0.5080
    Epoch 994/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8849 - Accuracy: 0.5090
    Epoch 995/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9122 - Accuracy: 0.4990
    Epoch 996/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8686 - Accuracy: 0.5010
    Epoch 997/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9027 - Accuracy: 0.4790
    Epoch 998/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9779 - Accuracy: 0.5060
    Epoch 999/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8861 - Accuracy: 0.4710
    Epoch 1000/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8679 - Accuracy: 0.5090
    Epoch 1001/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8687 - Accuracy: 0.4950
    Epoch 1002/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0670 - Accuracy: 0.5060
    Epoch 1003/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9117 - Accuracy: 0.4890
    Epoch 1004/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9354 - Accuracy: 0.4790
    Epoch 1005/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8841 - Accuracy: 0.4860
    Epoch 1006/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8884 - Accuracy: 0.4930
    Epoch 1007/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9517 - Accuracy: 0.5020
    Epoch 1008/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9043 - Accuracy: 0.4820
    Epoch 1009/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9492 - Accuracy: 0.5070
    Epoch 1010/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9594 - Accuracy: 0.5270
    Epoch 1011/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8864 - Accuracy: 0.5170
    Epoch 1012/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8563 - Accuracy: 0.5220
    Epoch 1013/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8712 - Accuracy: 0.5120
    Epoch 1014/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9186 - Accuracy: 0.5060
    Epoch 1015/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8940 - Accuracy: 0.5160
    Epoch 1016/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8628 - Accuracy: 0.5130
    Epoch 1017/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8734 - Accuracy: 0.5060
    Epoch 1018/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9164 - Accuracy: 0.4870
    Epoch 1019/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8874 - Accuracy: 0.5210
    Epoch 1020/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8908 - Accuracy: 0.5270
    Epoch 1021/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8985 - Accuracy: 0.5180
    Epoch 1022/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9002 - Accuracy: 0.4960
    Epoch 1023/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9098 - Accuracy: 0.4920
    Epoch 1024/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8849 - Accuracy: 0.5130
    Epoch 1025/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9346 - Accuracy: 0.5050
    Epoch 1026/3000
    32/32 [==============================] - 0s 5ms/step - loss: 0.8860 - Accuracy: 0.4840
    Epoch 1027/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8890 - Accuracy: 0.4940
    Epoch 1028/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8475 - Accuracy: 0.5240
    Epoch 1029/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8627 - Accuracy: 0.4860
    Epoch 1030/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8716 - Accuracy: 0.5110
    Epoch 1031/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8888 - Accuracy: 0.5160
    Epoch 1032/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8915 - Accuracy: 0.5140
    Epoch 1033/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8503 - Accuracy: 0.5260
    Epoch 1034/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8571 - Accuracy: 0.5460
    Epoch 1035/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8728 - Accuracy: 0.5100
    Epoch 1036/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8883 - Accuracy: 0.5020
    Epoch 1037/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9154 - Accuracy: 0.4930
    Epoch 1038/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8889 - Accuracy: 0.4760
    Epoch 1039/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9229 - Accuracy: 0.4610
    Epoch 1040/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8989 - Accuracy: 0.4960
    Epoch 1041/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9380 - Accuracy: 0.5270
    Epoch 1042/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9094 - Accuracy: 0.4750
    Epoch 1043/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8762 - Accuracy: 0.5120
    Epoch 1044/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8631 - Accuracy: 0.4890
    Epoch 1045/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8652 - Accuracy: 0.4940
    Epoch 1046/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8727 - Accuracy: 0.5050
    Epoch 1047/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9025 - Accuracy: 0.5060
    Epoch 1048/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9591 - Accuracy: 0.4960
    Epoch 1049/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9058 - Accuracy: 0.4670
    Epoch 1050/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9600 - Accuracy: 0.5330
    Epoch 1051/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9026 - Accuracy: 0.5160
    Epoch 1052/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9051 - Accuracy: 0.5170
    Epoch 1053/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8542 - Accuracy: 0.5070
    Epoch 1054/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8422 - Accuracy: 0.5450
    Epoch 1055/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9347 - Accuracy: 0.4790
    Epoch 1056/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8980 - Accuracy: 0.4980
    Epoch 1057/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8719 - Accuracy: 0.5070
    Epoch 1058/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8677 - Accuracy: 0.5210
    Epoch 1059/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8897 - Accuracy: 0.4780
    Epoch 1060/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8930 - Accuracy: 0.5200
    Epoch 1061/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8628 - Accuracy: 0.4900
    Epoch 1062/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8676 - Accuracy: 0.5110
    Epoch 1063/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9166 - Accuracy: 0.5100
    Epoch 1064/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8521 - Accuracy: 0.5310
    Epoch 1065/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9483 - Accuracy: 0.5080
    Epoch 1066/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8576 - Accuracy: 0.4810
    Epoch 1067/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8804 - Accuracy: 0.5290
    Epoch 1068/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8717 - Accuracy: 0.5290
    Epoch 1069/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9109 - Accuracy: 0.4940
    Epoch 1070/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8991 - Accuracy: 0.5030
    Epoch 1071/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8919 - Accuracy: 0.5220
    Epoch 1072/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8771 - Accuracy: 0.5000
    Epoch 1073/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8720 - Accuracy: 0.5100
    Epoch 1074/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8927 - Accuracy: 0.5230
    Epoch 1075/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9343 - Accuracy: 0.4780
    Epoch 1076/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8609 - Accuracy: 0.4810
    Epoch 1077/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8832 - Accuracy: 0.5070
    Epoch 1078/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9141 - Accuracy: 0.5360
    Epoch 1079/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8679 - Accuracy: 0.5340
    Epoch 1080/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8722 - Accuracy: 0.4930
    Epoch 1081/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8698 - Accuracy: 0.5180
    Epoch 1082/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9129 - Accuracy: 0.4810
    Epoch 1083/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9748 - Accuracy: 0.4680
    Epoch 1084/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8752 - Accuracy: 0.5010
    Epoch 1085/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9007 - Accuracy: 0.5190
    Epoch 1086/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9195 - Accuracy: 0.5090
    Epoch 1087/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9041 - Accuracy: 0.5250
    Epoch 1088/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9071 - Accuracy: 0.5080
    Epoch 1089/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8630 - Accuracy: 0.4890
    Epoch 1090/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8310 - Accuracy: 0.5000
    Epoch 1091/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9460 - Accuracy: 0.5100
    Epoch 1092/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8520 - Accuracy: 0.5170
    Epoch 1093/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8974 - Accuracy: 0.4950
    Epoch 1094/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8586 - Accuracy: 0.5120
    Epoch 1095/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8704 - Accuracy: 0.4740
    Epoch 1096/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9074 - Accuracy: 0.5130
    Epoch 1097/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9001 - Accuracy: 0.4850
    Epoch 1098/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9134 - Accuracy: 0.5030
    Epoch 1099/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8937 - Accuracy: 0.5240
    Epoch 1100/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9046 - Accuracy: 0.5050
    Epoch 1101/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8496 - Accuracy: 0.4930
    Epoch 1102/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8995 - Accuracy: 0.4480
    Epoch 1103/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8788 - Accuracy: 0.4810
    Epoch 1104/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8756 - Accuracy: 0.5220
    Epoch 1105/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8778 - Accuracy: 0.5220
    Epoch 1106/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9166 - Accuracy: 0.4940
    Epoch 1107/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8508 - Accuracy: 0.5130
    Epoch 1108/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9403 - Accuracy: 0.4920
    Epoch 1109/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8632 - Accuracy: 0.5110
    Epoch 1110/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8885 - Accuracy: 0.4900
    Epoch 1111/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9308 - Accuracy: 0.5120
    Epoch 1112/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8738 - Accuracy: 0.4880
    Epoch 1113/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9735 - Accuracy: 0.4980
    Epoch 1114/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9747 - Accuracy: 0.5490
    Epoch 1115/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8976 - Accuracy: 0.5290
    Epoch 1116/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9319 - Accuracy: 0.4960
    Epoch 1117/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8352 - Accuracy: 0.5110
    Epoch 1118/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8639 - Accuracy: 0.5400
    Epoch 1119/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8963 - Accuracy: 0.5190
    Epoch 1120/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8450 - Accuracy: 0.5050
    Epoch 1121/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9168 - Accuracy: 0.5060
    Epoch 1122/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8559 - Accuracy: 0.5250
    Epoch 1123/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9409 - Accuracy: 0.5100
    Epoch 1124/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8605 - Accuracy: 0.5290
    Epoch 1125/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8995 - Accuracy: 0.4950
    Epoch 1126/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8543 - Accuracy: 0.5370
    Epoch 1127/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8814 - Accuracy: 0.4980
    Epoch 1128/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9567 - Accuracy: 0.4810
    Epoch 1129/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8567 - Accuracy: 0.4940
    Epoch 1130/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8581 - Accuracy: 0.5030
    Epoch 1131/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8665 - Accuracy: 0.5070
    Epoch 1132/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8490 - Accuracy: 0.4990
    Epoch 1133/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8622 - Accuracy: 0.5080
    Epoch 1134/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8610 - Accuracy: 0.5130
    Epoch 1135/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8976 - Accuracy: 0.4920
    Epoch 1136/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8761 - Accuracy: 0.5070
    Epoch 1137/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9526 - Accuracy: 0.4850
    Epoch 1138/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9221 - Accuracy: 0.5050
    Epoch 1139/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8863 - Accuracy: 0.5300
    Epoch 1140/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8878 - Accuracy: 0.5050
    Epoch 1141/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8963 - Accuracy: 0.4970
    Epoch 1142/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0387 - Accuracy: 0.4930
    Epoch 1143/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9995 - Accuracy: 0.4900
    Epoch 1144/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8754 - Accuracy: 0.5030
    Epoch 1145/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9157 - Accuracy: 0.4860
    Epoch 1146/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8583 - Accuracy: 0.5030
    Epoch 1147/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8981 - Accuracy: 0.5140
    Epoch 1148/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9044 - Accuracy: 0.4870
    Epoch 1149/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9818 - Accuracy: 0.4900
    Epoch 1150/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8489 - Accuracy: 0.5530
    Epoch 1151/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9049 - Accuracy: 0.4870
    Epoch 1152/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8611 - Accuracy: 0.5280
    Epoch 1153/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8741 - Accuracy: 0.4900
    Epoch 1154/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8709 - Accuracy: 0.5240
    Epoch 1155/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9002 - Accuracy: 0.4900
    Epoch 1156/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9648 - Accuracy: 0.4910
    Epoch 1157/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8887 - Accuracy: 0.4960
    Epoch 1158/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8543 - Accuracy: 0.5120
    Epoch 1159/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8520 - Accuracy: 0.5110
    Epoch 1160/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8698 - Accuracy: 0.5050
    Epoch 1161/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8814 - Accuracy: 0.5120
    Epoch 1162/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8809 - Accuracy: 0.4880
    Epoch 1163/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9406 - Accuracy: 0.4960
    Epoch 1164/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9717 - Accuracy: 0.5030
    Epoch 1165/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8780 - Accuracy: 0.5250
    Epoch 1166/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9065 - Accuracy: 0.5110
    Epoch 1167/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8630 - Accuracy: 0.5350
    Epoch 1168/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8587 - Accuracy: 0.5170
    Epoch 1169/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8891 - Accuracy: 0.5150
    Epoch 1170/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8891 - Accuracy: 0.5140
    Epoch 1171/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8796 - Accuracy: 0.4830
    Epoch 1172/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8590 - Accuracy: 0.5130
    Epoch 1173/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8578 - Accuracy: 0.4860
    Epoch 1174/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9012 - Accuracy: 0.4670
    Epoch 1175/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8693 - Accuracy: 0.5030
    Epoch 1176/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8895 - Accuracy: 0.5270
    Epoch 1177/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8661 - Accuracy: 0.5170
    Epoch 1178/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8652 - Accuracy: 0.5060
    Epoch 1179/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8669 - Accuracy: 0.5350
    Epoch 1180/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8638 - Accuracy: 0.5130
    Epoch 1181/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1579 - Accuracy: 0.5310
    Epoch 1182/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8941 - Accuracy: 0.5180
    Epoch 1183/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8797 - Accuracy: 0.4900
    Epoch 1184/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9016 - Accuracy: 0.5020
    Epoch 1185/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1276 - Accuracy: 0.4750
    Epoch 1186/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8619 - Accuracy: 0.4980
    Epoch 1187/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8621 - Accuracy: 0.5130
    Epoch 1188/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8766 - Accuracy: 0.5000
    Epoch 1189/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8950 - Accuracy: 0.4970
    Epoch 1190/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8826 - Accuracy: 0.4970
    Epoch 1191/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8481 - Accuracy: 0.5260
    Epoch 1192/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8536 - Accuracy: 0.4830
    Epoch 1193/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9197 - Accuracy: 0.4980
    Epoch 1194/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8575 - Accuracy: 0.4810
    Epoch 1195/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8881 - Accuracy: 0.4920
    Epoch 1196/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8685 - Accuracy: 0.5240
    Epoch 1197/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9214 - Accuracy: 0.5360
    Epoch 1198/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8403 - Accuracy: 0.5060
    Epoch 1199/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9151 - Accuracy: 0.5230
    Epoch 1200/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8649 - Accuracy: 0.4910
    Epoch 1201/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8477 - Accuracy: 0.4890
    Epoch 1202/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8604 - Accuracy: 0.4950
    Epoch 1203/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8864 - Accuracy: 0.4990
    Epoch 1204/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9045 - Accuracy: 0.4890
    Epoch 1205/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9056 - Accuracy: 0.4780
    Epoch 1206/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9976 - Accuracy: 0.4880
    Epoch 1207/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9370 - Accuracy: 0.4860
    Epoch 1208/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8815 - Accuracy: 0.5140
    Epoch 1209/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8978 - Accuracy: 0.4900
    Epoch 1210/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8239 - Accuracy: 0.5090
    Epoch 1211/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9014 - Accuracy: 0.4790
    Epoch 1212/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8486 - Accuracy: 0.5000
    Epoch 1213/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8888 - Accuracy: 0.4820
    Epoch 1214/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8653 - Accuracy: 0.5010
    Epoch 1215/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9119 - Accuracy: 0.4820
    Epoch 1216/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8785 - Accuracy: 0.4900
    Epoch 1217/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9261 - Accuracy: 0.5060
    Epoch 1218/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8944 - Accuracy: 0.4830
    Epoch 1219/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8999 - Accuracy: 0.5150
    Epoch 1220/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8425 - Accuracy: 0.5060
    Epoch 1221/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8755 - Accuracy: 0.5010
    Epoch 1222/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8530 - Accuracy: 0.4970
    Epoch 1223/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9066 - Accuracy: 0.5140
    Epoch 1224/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9270 - Accuracy: 0.4880
    Epoch 1225/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8517 - Accuracy: 0.5260
    Epoch 1226/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9104 - Accuracy: 0.5110
    Epoch 1227/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8537 - Accuracy: 0.5040
    Epoch 1228/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8752 - Accuracy: 0.4940
    Epoch 1229/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8587 - Accuracy: 0.5080
    Epoch 1230/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8690 - Accuracy: 0.4930
    Epoch 1231/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9264 - Accuracy: 0.5090
    Epoch 1232/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9388 - Accuracy: 0.4890
    Epoch 1233/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8894 - Accuracy: 0.5170
    Epoch 1234/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8737 - Accuracy: 0.5390
    Epoch 1235/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8667 - Accuracy: 0.4860
    Epoch 1236/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8451 - Accuracy: 0.5240
    Epoch 1237/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8666 - Accuracy: 0.4800
    Epoch 1238/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8828 - Accuracy: 0.4780
    Epoch 1239/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9319 - Accuracy: 0.4700
    Epoch 1240/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8761 - Accuracy: 0.5120
    Epoch 1241/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8312 - Accuracy: 0.5360
    Epoch 1242/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8597 - Accuracy: 0.5230
    Epoch 1243/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9447 - Accuracy: 0.4870
    Epoch 1244/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8906 - Accuracy: 0.4950
    Epoch 1245/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8502 - Accuracy: 0.5300
    Epoch 1246/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8516 - Accuracy: 0.5260
    Epoch 1247/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9054 - Accuracy: 0.5010
    Epoch 1248/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9237 - Accuracy: 0.5050
    Epoch 1249/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9137 - Accuracy: 0.5150
    Epoch 1250/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8782 - Accuracy: 0.5500
    Epoch 1251/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8921 - Accuracy: 0.5180
    Epoch 1252/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8521 - Accuracy: 0.5060
    Epoch 1253/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8887 - Accuracy: 0.4940
    Epoch 1254/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8338 - Accuracy: 0.5300
    Epoch 1255/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8413 - Accuracy: 0.4980
    Epoch 1256/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8864 - Accuracy: 0.5130
    Epoch 1257/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8493 - Accuracy: 0.4850
    Epoch 1258/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9973 - Accuracy: 0.5090
    Epoch 1259/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8424 - Accuracy: 0.5130
    Epoch 1260/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8918 - Accuracy: 0.5120
    Epoch 1261/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8641 - Accuracy: 0.5440
    Epoch 1262/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8584 - Accuracy: 0.5070
    Epoch 1263/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8255 - Accuracy: 0.5380
    Epoch 1264/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9024 - Accuracy: 0.5160
    Epoch 1265/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8448 - Accuracy: 0.5250
    Epoch 1266/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8961 - Accuracy: 0.5000
    Epoch 1267/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9068 - Accuracy: 0.4960
    Epoch 1268/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8721 - Accuracy: 0.4650
    Epoch 1269/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8702 - Accuracy: 0.4850
    Epoch 1270/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8974 - Accuracy: 0.5120
    Epoch 1271/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9386 - Accuracy: 0.5130
    Epoch 1272/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8723 - Accuracy: 0.4900
    Epoch 1273/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8951 - Accuracy: 0.5090
    Epoch 1274/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8801 - Accuracy: 0.5140
    Epoch 1275/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8518 - Accuracy: 0.5100
    Epoch 1276/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8729 - Accuracy: 0.4910
    Epoch 1277/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8343 - Accuracy: 0.5140
    Epoch 1278/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8918 - Accuracy: 0.4680
    Epoch 1279/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8598 - Accuracy: 0.4790
    Epoch 1280/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8766 - Accuracy: 0.5070
    Epoch 1281/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8497 - Accuracy: 0.5060
    Epoch 1282/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8451 - Accuracy: 0.4700
    Epoch 1283/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8637 - Accuracy: 0.5160
    Epoch 1284/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9544 - Accuracy: 0.4980
    Epoch 1285/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9861 - Accuracy: 0.4990
    Epoch 1286/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9018 - Accuracy: 0.4840
    Epoch 1287/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8970 - Accuracy: 0.4940
    Epoch 1288/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8841 - Accuracy: 0.5050
    Epoch 1289/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8490 - Accuracy: 0.4720
    Epoch 1290/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8957 - Accuracy: 0.4680
    Epoch 1291/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8449 - Accuracy: 0.4930
    Epoch 1292/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8836 - Accuracy: 0.5040
    Epoch 1293/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8972 - Accuracy: 0.5150
    Epoch 1294/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8322 - Accuracy: 0.5210
    Epoch 1295/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8548 - Accuracy: 0.4790
    Epoch 1296/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9290 - Accuracy: 0.5000
    Epoch 1297/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8372 - Accuracy: 0.5280
    Epoch 1298/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8400 - Accuracy: 0.5050
    Epoch 1299/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8678 - Accuracy: 0.5120
    Epoch 1300/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8383 - Accuracy: 0.5110
    Epoch 1301/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8526 - Accuracy: 0.5000
    Epoch 1302/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8655 - Accuracy: 0.5060
    Epoch 1303/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9818 - Accuracy: 0.4980
    Epoch 1304/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8377 - Accuracy: 0.5180
    Epoch 1305/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9015 - Accuracy: 0.5150
    Epoch 1306/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8348 - Accuracy: 0.5410
    Epoch 1307/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8764 - Accuracy: 0.5010
    Epoch 1308/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8273 - Accuracy: 0.5120
    Epoch 1309/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9320 - Accuracy: 0.5170
    Epoch 1310/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8609 - Accuracy: 0.5150
    Epoch 1311/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8532 - Accuracy: 0.4780
    Epoch 1312/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8654 - Accuracy: 0.4880
    Epoch 1313/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8667 - Accuracy: 0.5270
    Epoch 1314/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9432 - Accuracy: 0.4880
    Epoch 1315/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8837 - Accuracy: 0.5060
    Epoch 1316/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8476 - Accuracy: 0.4990
    Epoch 1317/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8661 - Accuracy: 0.5050
    Epoch 1318/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8662 - Accuracy: 0.5030
    Epoch 1319/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8356 - Accuracy: 0.5210
    Epoch 1320/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8484 - Accuracy: 0.5070
    Epoch 1321/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9324 - Accuracy: 0.5110
    Epoch 1322/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9209 - Accuracy: 0.4990
    Epoch 1323/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8641 - Accuracy: 0.4870
    Epoch 1324/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8847 - Accuracy: 0.5060
    Epoch 1325/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9233 - Accuracy: 0.4740
    Epoch 1326/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8429 - Accuracy: 0.5050
    Epoch 1327/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8496 - Accuracy: 0.5040
    Epoch 1328/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8803 - Accuracy: 0.5010
    Epoch 1329/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8856 - Accuracy: 0.5080
    Epoch 1330/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8855 - Accuracy: 0.5090
    Epoch 1331/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8682 - Accuracy: 0.5070
    Epoch 1332/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8960 - Accuracy: 0.5210
    Epoch 1333/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8404 - Accuracy: 0.4930
    Epoch 1334/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8546 - Accuracy: 0.4840
    Epoch 1335/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8248 - Accuracy: 0.5230
    Epoch 1336/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9356 - Accuracy: 0.4860
    Epoch 1337/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8607 - Accuracy: 0.4860
    Epoch 1338/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9206 - Accuracy: 0.4840
    Epoch 1339/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8912 - Accuracy: 0.5000
    Epoch 1340/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8406 - Accuracy: 0.4920
    Epoch 1341/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9142 - Accuracy: 0.4900
    Epoch 1342/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8447 - Accuracy: 0.5060
    Epoch 1343/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8359 - Accuracy: 0.5040
    Epoch 1344/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8870 - Accuracy: 0.4810
    Epoch 1345/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8310 - Accuracy: 0.5290
    Epoch 1346/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9285 - Accuracy: 0.5070
    Epoch 1347/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8922 - Accuracy: 0.5240
    Epoch 1348/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8565 - Accuracy: 0.5110
    Epoch 1349/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8279 - Accuracy: 0.5190
    Epoch 1350/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8503 - Accuracy: 0.5110
    Epoch 1351/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8646 - Accuracy: 0.4970
    Epoch 1352/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8694 - Accuracy: 0.4940
    Epoch 1353/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8847 - Accuracy: 0.4950
    Epoch 1354/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9063 - Accuracy: 0.4650
    Epoch 1355/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8373 - Accuracy: 0.4810
    Epoch 1356/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8592 - Accuracy: 0.5150
    Epoch 1357/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8608 - Accuracy: 0.5120
    Epoch 1358/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0110 - Accuracy: 0.4870
    Epoch 1359/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8385 - Accuracy: 0.5320
    Epoch 1360/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8319 - Accuracy: 0.5020
    Epoch 1361/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8341 - Accuracy: 0.4780
    Epoch 1362/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8473 - Accuracy: 0.5020
    Epoch 1363/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9960 - Accuracy: 0.4830
    Epoch 1364/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8761 - Accuracy: 0.5010
    Epoch 1365/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9152 - Accuracy: 0.4960
    Epoch 1366/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8261 - Accuracy: 0.5110
    Epoch 1367/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8991 - Accuracy: 0.4840
    Epoch 1368/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8523 - Accuracy: 0.5180
    Epoch 1369/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8640 - Accuracy: 0.4750
    Epoch 1370/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8561 - Accuracy: 0.4920
    Epoch 1371/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8619 - Accuracy: 0.5070
    Epoch 1372/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9334 - Accuracy: 0.5090
    Epoch 1373/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8441 - Accuracy: 0.5050
    Epoch 1374/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9431 - Accuracy: 0.5060
    Epoch 1375/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8646 - Accuracy: 0.5010
    Epoch 1376/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8361 - Accuracy: 0.5090
    Epoch 1377/3000
    32/32 [==============================] - 0s 5ms/step - loss: 0.8594 - Accuracy: 0.4980
    Epoch 1378/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8316 - Accuracy: 0.5280
    Epoch 1379/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8471 - Accuracy: 0.4880
    Epoch 1380/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8890 - Accuracy: 0.5420
    Epoch 1381/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8349 - Accuracy: 0.5200
    Epoch 1382/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9102 - Accuracy: 0.5180
    Epoch 1383/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8482 - Accuracy: 0.4890
    Epoch 1384/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8599 - Accuracy: 0.5040
    Epoch 1385/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8961 - Accuracy: 0.5050
    Epoch 1386/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9953 - Accuracy: 0.5000
    Epoch 1387/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8592 - Accuracy: 0.4940
    Epoch 1388/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8288 - Accuracy: 0.5300
    Epoch 1389/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8371 - Accuracy: 0.5090
    Epoch 1390/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8343 - Accuracy: 0.5070
    Epoch 1391/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8172 - Accuracy: 0.5350
    Epoch 1392/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9039 - Accuracy: 0.5100
    Epoch 1393/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8654 - Accuracy: 0.5110
    Epoch 1394/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8645 - Accuracy: 0.5050
    Epoch 1395/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9241 - Accuracy: 0.5000
    Epoch 1396/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8471 - Accuracy: 0.5200
    Epoch 1397/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8576 - Accuracy: 0.4890
    Epoch 1398/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9012 - Accuracy: 0.5200
    Epoch 1399/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8362 - Accuracy: 0.4940
    Epoch 1400/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8788 - Accuracy: 0.4850
    Epoch 1401/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9331 - Accuracy: 0.4860
    Epoch 1402/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8461 - Accuracy: 0.5060
    Epoch 1403/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8562 - Accuracy: 0.5060
    Epoch 1404/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8853 - Accuracy: 0.5020
    Epoch 1405/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8320 - Accuracy: 0.5240
    Epoch 1406/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8367 - Accuracy: 0.4960
    Epoch 1407/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8761 - Accuracy: 0.5250
    Epoch 1408/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8441 - Accuracy: 0.5090
    Epoch 1409/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8402 - Accuracy: 0.5120
    Epoch 1410/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8576 - Accuracy: 0.4850
    Epoch 1411/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8597 - Accuracy: 0.4780
    Epoch 1412/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8786 - Accuracy: 0.4930
    Epoch 1413/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8417 - Accuracy: 0.5210
    Epoch 1414/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8672 - Accuracy: 0.5190
    Epoch 1415/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8771 - Accuracy: 0.5280
    Epoch 1416/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9365 - Accuracy: 0.4890
    Epoch 1417/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8657 - Accuracy: 0.4970
    Epoch 1418/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8785 - Accuracy: 0.5020
    Epoch 1419/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9272 - Accuracy: 0.5000
    Epoch 1420/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8856 - Accuracy: 0.4950
    Epoch 1421/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9394 - Accuracy: 0.4830
    Epoch 1422/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8782 - Accuracy: 0.4720
    Epoch 1423/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9449 - Accuracy: 0.5170
    Epoch 1424/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8488 - Accuracy: 0.5110
    Epoch 1425/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8668 - Accuracy: 0.5100
    Epoch 1426/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8495 - Accuracy: 0.5010
    Epoch 1427/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8662 - Accuracy: 0.5080
    Epoch 1428/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8217 - Accuracy: 0.5220
    Epoch 1429/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8672 - Accuracy: 0.4920
    Epoch 1430/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8726 - Accuracy: 0.4760
    Epoch 1431/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0672 - Accuracy: 0.5310
    Epoch 1432/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9746 - Accuracy: 0.4800
    Epoch 1433/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8589 - Accuracy: 0.5190
    Epoch 1434/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8833 - Accuracy: 0.5200
    Epoch 1435/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8578 - Accuracy: 0.4960
    Epoch 1436/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8666 - Accuracy: 0.5160
    Epoch 1437/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8535 - Accuracy: 0.4910
    Epoch 1438/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8907 - Accuracy: 0.5020
    Epoch 1439/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8646 - Accuracy: 0.4870
    Epoch 1440/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8584 - Accuracy: 0.5120
    Epoch 1441/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8716 - Accuracy: 0.5000
    Epoch 1442/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8544 - Accuracy: 0.4820
    Epoch 1443/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8518 - Accuracy: 0.4890
    Epoch 1444/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8335 - Accuracy: 0.5000
    Epoch 1445/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8351 - Accuracy: 0.5000
    Epoch 1446/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8609 - Accuracy: 0.5010
    Epoch 1447/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8819 - Accuracy: 0.5220
    Epoch 1448/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8931 - Accuracy: 0.4810
    Epoch 1449/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9410 - Accuracy: 0.4830
    Epoch 1450/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8772 - Accuracy: 0.4940
    Epoch 1451/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8172 - Accuracy: 0.5300
    Epoch 1452/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8366 - Accuracy: 0.5160
    Epoch 1453/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8545 - Accuracy: 0.4970
    Epoch 1454/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8879 - Accuracy: 0.5030
    Epoch 1455/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8444 - Accuracy: 0.5260
    Epoch 1456/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8541 - Accuracy: 0.4950
    Epoch 1457/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8450 - Accuracy: 0.4860
    Epoch 1458/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8334 - Accuracy: 0.5220
    Epoch 1459/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8765 - Accuracy: 0.5280
    Epoch 1460/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8564 - Accuracy: 0.5230
    Epoch 1461/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8764 - Accuracy: 0.4990
    Epoch 1462/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8727 - Accuracy: 0.4990
    Epoch 1463/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9072 - Accuracy: 0.4830
    Epoch 1464/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8482 - Accuracy: 0.5050
    Epoch 1465/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8732 - Accuracy: 0.5070
    Epoch 1466/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8722 - Accuracy: 0.5170
    Epoch 1467/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8242 - Accuracy: 0.5300
    Epoch 1468/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8392 - Accuracy: 0.5060
    Epoch 1469/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8451 - Accuracy: 0.5060
    Epoch 1470/3000
    32/32 [==============================] - 0s 5ms/step - loss: 0.8444 - Accuracy: 0.5130
    Epoch 1471/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8430 - Accuracy: 0.4960
    Epoch 1472/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8487 - Accuracy: 0.4900
    Epoch 1473/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8251 - Accuracy: 0.5310
    Epoch 1474/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9260 - Accuracy: 0.4980
    Epoch 1475/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8733 - Accuracy: 0.5120
    Epoch 1476/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8638 - Accuracy: 0.5220
    Epoch 1477/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8777 - Accuracy: 0.4730
    Epoch 1478/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8222 - Accuracy: 0.5260
    Epoch 1479/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8860 - Accuracy: 0.5170
    Epoch 1480/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8545 - Accuracy: 0.5080
    Epoch 1481/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8914 - Accuracy: 0.5090
    Epoch 1482/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8758 - Accuracy: 0.4860
    Epoch 1483/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8719 - Accuracy: 0.5110
    Epoch 1484/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9141 - Accuracy: 0.4910
    Epoch 1485/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8519 - Accuracy: 0.5060
    Epoch 1486/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8621 - Accuracy: 0.5010
    Epoch 1487/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8176 - Accuracy: 0.4990
    Epoch 1488/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9042 - Accuracy: 0.5100
    Epoch 1489/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8525 - Accuracy: 0.5210
    Epoch 1490/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8828 - Accuracy: 0.4810
    Epoch 1491/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9823 - Accuracy: 0.4970
    Epoch 1492/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8378 - Accuracy: 0.5350
    Epoch 1493/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8884 - Accuracy: 0.5220
    Epoch 1494/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8417 - Accuracy: 0.5130
    Epoch 1495/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8565 - Accuracy: 0.5120
    Epoch 1496/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8655 - Accuracy: 0.4570
    Epoch 1497/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9246 - Accuracy: 0.4940
    Epoch 1498/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8671 - Accuracy: 0.4850
    Epoch 1499/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9441 - Accuracy: 0.4970
    Epoch 1500/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8533 - Accuracy: 0.4910
    Epoch 1501/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0027 - Accuracy: 0.4780
    Epoch 1502/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8521 - Accuracy: 0.5170
    Epoch 1503/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8750 - Accuracy: 0.5050
    Epoch 1504/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8332 - Accuracy: 0.5050
    Epoch 1505/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8552 - Accuracy: 0.5060
    Epoch 1506/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8404 - Accuracy: 0.5060
    Epoch 1507/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0418 - Accuracy: 0.4770
    Epoch 1508/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8359 - Accuracy: 0.4830
    Epoch 1509/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9685 - Accuracy: 0.5120
    Epoch 1510/3000
    32/32 [==============================] - 0s 5ms/step - loss: 0.8338 - Accuracy: 0.5100
    Epoch 1511/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9356 - Accuracy: 0.4780
    Epoch 1512/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8727 - Accuracy: 0.4980
    Epoch 1513/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8317 - Accuracy: 0.5100
    Epoch 1514/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8459 - Accuracy: 0.4950
    Epoch 1515/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8512 - Accuracy: 0.5060
    Epoch 1516/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9458 - Accuracy: 0.5210
    Epoch 1517/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8334 - Accuracy: 0.5090
    Epoch 1518/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8345 - Accuracy: 0.5020
    Epoch 1519/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8624 - Accuracy: 0.5060
    Epoch 1520/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8396 - Accuracy: 0.5020
    Epoch 1521/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8434 - Accuracy: 0.5050
    Epoch 1522/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8914 - Accuracy: 0.4890
    Epoch 1523/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8419 - Accuracy: 0.4910
    Epoch 1524/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8814 - Accuracy: 0.4930
    Epoch 1525/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8259 - Accuracy: 0.5150
    Epoch 1526/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8934 - Accuracy: 0.4820
    Epoch 1527/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9077 - Accuracy: 0.4860
    Epoch 1528/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8221 - Accuracy: 0.5180
    Epoch 1529/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8713 - Accuracy: 0.5010
    Epoch 1530/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8837 - Accuracy: 0.4720
    Epoch 1531/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8801 - Accuracy: 0.5040
    Epoch 1532/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8600 - Accuracy: 0.4990
    Epoch 1533/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8862 - Accuracy: 0.4890
    Epoch 1534/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8213 - Accuracy: 0.5290
    Epoch 1535/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8286 - Accuracy: 0.4950
    Epoch 1536/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8351 - Accuracy: 0.5410
    Epoch 1537/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8837 - Accuracy: 0.4980
    Epoch 1538/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8267 - Accuracy: 0.5250
    Epoch 1539/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8380 - Accuracy: 0.4820
    Epoch 1540/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9039 - Accuracy: 0.5100
    Epoch 1541/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8595 - Accuracy: 0.5180
    Epoch 1542/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8280 - Accuracy: 0.5140
    Epoch 1543/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8307 - Accuracy: 0.5240
    Epoch 1544/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8389 - Accuracy: 0.4970
    Epoch 1545/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8475 - Accuracy: 0.5040
    Epoch 1546/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8423 - Accuracy: 0.4850
    Epoch 1547/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8374 - Accuracy: 0.5150
    Epoch 1548/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8509 - Accuracy: 0.4780
    Epoch 1549/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8802 - Accuracy: 0.5330
    Epoch 1550/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8458 - Accuracy: 0.4970
    Epoch 1551/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9114 - Accuracy: 0.5050
    Epoch 1552/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8819 - Accuracy: 0.4680
    Epoch 1553/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9347 - Accuracy: 0.4520
    Epoch 1554/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8462 - Accuracy: 0.5340
    Epoch 1555/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9130 - Accuracy: 0.4820
    Epoch 1556/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8416 - Accuracy: 0.5330
    Epoch 1557/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8299 - Accuracy: 0.5070
    Epoch 1558/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8577 - Accuracy: 0.4960
    Epoch 1559/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8438 - Accuracy: 0.5260
    Epoch 1560/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8314 - Accuracy: 0.4970
    Epoch 1561/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8610 - Accuracy: 0.5410
    Epoch 1562/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8339 - Accuracy: 0.5050
    Epoch 1563/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8322 - Accuracy: 0.4930
    Epoch 1564/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8723 - Accuracy: 0.5090
    Epoch 1565/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8773 - Accuracy: 0.4890
    Epoch 1566/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8285 - Accuracy: 0.5240
    Epoch 1567/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8209 - Accuracy: 0.5190
    Epoch 1568/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8387 - Accuracy: 0.4970
    Epoch 1569/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8600 - Accuracy: 0.4970
    Epoch 1570/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8504 - Accuracy: 0.5080
    Epoch 1571/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9269 - Accuracy: 0.4740
    Epoch 1572/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8198 - Accuracy: 0.5390
    Epoch 1573/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8740 - Accuracy: 0.4910
    Epoch 1574/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8477 - Accuracy: 0.5030
    Epoch 1575/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8847 - Accuracy: 0.4840
    Epoch 1576/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8472 - Accuracy: 0.5020
    Epoch 1577/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8752 - Accuracy: 0.4920
    Epoch 1578/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8750 - Accuracy: 0.5210
    Epoch 1579/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9408 - Accuracy: 0.4700
    Epoch 1580/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8188 - Accuracy: 0.5110
    Epoch 1581/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8372 - Accuracy: 0.5070
    Epoch 1582/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9013 - Accuracy: 0.5270
    Epoch 1583/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8611 - Accuracy: 0.5310
    Epoch 1584/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8381 - Accuracy: 0.5120
    Epoch 1585/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8421 - Accuracy: 0.5030
    Epoch 1586/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8637 - Accuracy: 0.4930
    Epoch 1587/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9645 - Accuracy: 0.4800
    Epoch 1588/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8794 - Accuracy: 0.4950
    Epoch 1589/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8900 - Accuracy: 0.4840
    Epoch 1590/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8722 - Accuracy: 0.4750
    Epoch 1591/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8585 - Accuracy: 0.5050
    Epoch 1592/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8354 - Accuracy: 0.5070
    Epoch 1593/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8405 - Accuracy: 0.4930
    Epoch 1594/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8594 - Accuracy: 0.5150
    Epoch 1595/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8705 - Accuracy: 0.5040
    Epoch 1596/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8924 - Accuracy: 0.4950
    Epoch 1597/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8262 - Accuracy: 0.5280
    Epoch 1598/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8636 - Accuracy: 0.4870
    Epoch 1599/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9062 - Accuracy: 0.4910
    Epoch 1600/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9880 - Accuracy: 0.4990
    Epoch 1601/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8447 - Accuracy: 0.5180
    Epoch 1602/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8330 - Accuracy: 0.4950
    Epoch 1603/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8340 - Accuracy: 0.5380
    Epoch 1604/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8402 - Accuracy: 0.5130
    Epoch 1605/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8508 - Accuracy: 0.5090
    Epoch 1606/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8409 - Accuracy: 0.5240
    Epoch 1607/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8990 - Accuracy: 0.4830
    Epoch 1608/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8643 - Accuracy: 0.4650
    Epoch 1609/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8523 - Accuracy: 0.4970
    Epoch 1610/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8407 - Accuracy: 0.5070
    Epoch 1611/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8441 - Accuracy: 0.4860
    Epoch 1612/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9274 - Accuracy: 0.4840
    Epoch 1613/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8614 - Accuracy: 0.5050
    Epoch 1614/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9440 - Accuracy: 0.4970
    Epoch 1615/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8676 - Accuracy: 0.4510
    Epoch 1616/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9451 - Accuracy: 0.5240
    Epoch 1617/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8561 - Accuracy: 0.4780
    Epoch 1618/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8255 - Accuracy: 0.5040
    Epoch 1619/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8925 - Accuracy: 0.4800
    Epoch 1620/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8658 - Accuracy: 0.5270
    Epoch 1621/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8787 - Accuracy: 0.5000
    Epoch 1622/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8303 - Accuracy: 0.5020
    Epoch 1623/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8663 - Accuracy: 0.4990
    Epoch 1624/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9199 - Accuracy: 0.4850
    Epoch 1625/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9559 - Accuracy: 0.4960
    Epoch 1626/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8669 - Accuracy: 0.5120
    Epoch 1627/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8562 - Accuracy: 0.4680
    Epoch 1628/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8479 - Accuracy: 0.4900
    Epoch 1629/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9252 - Accuracy: 0.4530
    Epoch 1630/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9463 - Accuracy: 0.5200
    Epoch 1631/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8174 - Accuracy: 0.5080
    Epoch 1632/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8369 - Accuracy: 0.4980
    Epoch 1633/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8060 - Accuracy: 0.5240
    Epoch 1634/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8650 - Accuracy: 0.4960
    Epoch 1635/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8511 - Accuracy: 0.5160
    Epoch 1636/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8435 - Accuracy: 0.4880
    Epoch 1637/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8477 - Accuracy: 0.5160
    Epoch 1638/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8309 - Accuracy: 0.4830
    Epoch 1639/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8316 - Accuracy: 0.4760
    Epoch 1640/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8290 - Accuracy: 0.5050
    Epoch 1641/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8732 - Accuracy: 0.5070
    Epoch 1642/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9025 - Accuracy: 0.4920
    Epoch 1643/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8463 - Accuracy: 0.4830
    Epoch 1644/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8362 - Accuracy: 0.4770
    Epoch 1645/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8239 - Accuracy: 0.4990
    Epoch 1646/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8400 - Accuracy: 0.5110
    Epoch 1647/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8507 - Accuracy: 0.4810
    Epoch 1648/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9021 - Accuracy: 0.4570
    Epoch 1649/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9255 - Accuracy: 0.5010
    Epoch 1650/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8277 - Accuracy: 0.4980
    Epoch 1651/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8623 - Accuracy: 0.5220
    Epoch 1652/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8497 - Accuracy: 0.5010
    Epoch 1653/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8965 - Accuracy: 0.4980
    Epoch 1654/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8245 - Accuracy: 0.4830
    Epoch 1655/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8454 - Accuracy: 0.4760
    Epoch 1656/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8435 - Accuracy: 0.5250
    Epoch 1657/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9271 - Accuracy: 0.4950
    Epoch 1658/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8838 - Accuracy: 0.5010
    Epoch 1659/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9327 - Accuracy: 0.5030
    Epoch 1660/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8234 - Accuracy: 0.5050
    Epoch 1661/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8719 - Accuracy: 0.4890
    Epoch 1662/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8572 - Accuracy: 0.5200
    Epoch 1663/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8202 - Accuracy: 0.4930
    Epoch 1664/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8236 - Accuracy: 0.4830
    Epoch 1665/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8674 - Accuracy: 0.4910
    Epoch 1666/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9154 - Accuracy: 0.4960
    Epoch 1667/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8701 - Accuracy: 0.5010
    Epoch 1668/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8377 - Accuracy: 0.5350
    Epoch 1669/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8775 - Accuracy: 0.5540
    Epoch 1670/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8497 - Accuracy: 0.4920
    Epoch 1671/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8642 - Accuracy: 0.5140
    Epoch 1672/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8580 - Accuracy: 0.4940
    Epoch 1673/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9171 - Accuracy: 0.5040
    Epoch 1674/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8640 - Accuracy: 0.4690
    Epoch 1675/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8748 - Accuracy: 0.4970
    Epoch 1676/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8467 - Accuracy: 0.5170
    Epoch 1677/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8289 - Accuracy: 0.5100
    Epoch 1678/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8398 - Accuracy: 0.4840
    Epoch 1679/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8633 - Accuracy: 0.5220
    Epoch 1680/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8262 - Accuracy: 0.4970
    Epoch 1681/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9502 - Accuracy: 0.4800
    Epoch 1682/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8286 - Accuracy: 0.5040
    Epoch 1683/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8085 - Accuracy: 0.5400
    Epoch 1684/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8685 - Accuracy: 0.5320
    Epoch 1685/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8167 - Accuracy: 0.4810
    Epoch 1686/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8642 - Accuracy: 0.4920
    Epoch 1687/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9117 - Accuracy: 0.4730
    Epoch 1688/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8543 - Accuracy: 0.4980
    Epoch 1689/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8314 - Accuracy: 0.4930
    Epoch 1690/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8927 - Accuracy: 0.4850
    Epoch 1691/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8366 - Accuracy: 0.5450
    Epoch 1692/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8434 - Accuracy: 0.4930
    Epoch 1693/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8163 - Accuracy: 0.5080
    Epoch 1694/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8361 - Accuracy: 0.5140
    Epoch 1695/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8290 - Accuracy: 0.5020
    Epoch 1696/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0837 - Accuracy: 0.4950
    Epoch 1697/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8457 - Accuracy: 0.5230
    Epoch 1698/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8466 - Accuracy: 0.4970
    Epoch 1699/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8396 - Accuracy: 0.4660
    Epoch 1700/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8135 - Accuracy: 0.5210
    Epoch 1701/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8549 - Accuracy: 0.4900
    Epoch 1702/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8786 - Accuracy: 0.4800
    Epoch 1703/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8533 - Accuracy: 0.4870
    Epoch 1704/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8333 - Accuracy: 0.4940
    Epoch 1705/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8616 - Accuracy: 0.5190
    Epoch 1706/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9185 - Accuracy: 0.4750
    Epoch 1707/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8188 - Accuracy: 0.5210
    Epoch 1708/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8433 - Accuracy: 0.5170
    Epoch 1709/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8535 - Accuracy: 0.5140
    Epoch 1710/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8164 - Accuracy: 0.5130
    Epoch 1711/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0013 - Accuracy: 0.4980
    Epoch 1712/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8609 - Accuracy: 0.5240
    Epoch 1713/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9149 - Accuracy: 0.4860
    Epoch 1714/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8403 - Accuracy: 0.5070
    Epoch 1715/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8894 - Accuracy: 0.5430
    Epoch 1716/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8788 - Accuracy: 0.5150
    Epoch 1717/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9564 - Accuracy: 0.5360
    Epoch 1718/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9825 - Accuracy: 0.4950
    Epoch 1719/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8474 - Accuracy: 0.4940
    Epoch 1720/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9206 - Accuracy: 0.5090
    Epoch 1721/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8181 - Accuracy: 0.5220
    Epoch 1722/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8288 - Accuracy: 0.5070
    Epoch 1723/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8300 - Accuracy: 0.4870
    Epoch 1724/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8577 - Accuracy: 0.5040
    Epoch 1725/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8354 - Accuracy: 0.4960
    Epoch 1726/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8316 - Accuracy: 0.4710
    Epoch 1727/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8260 - Accuracy: 0.4760
    Epoch 1728/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8430 - Accuracy: 0.4920
    Epoch 1729/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8681 - Accuracy: 0.4860
    Epoch 1730/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8101 - Accuracy: 0.4900
    Epoch 1731/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8723 - Accuracy: 0.5060
    Epoch 1732/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9405 - Accuracy: 0.4980
    Epoch 1733/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8641 - Accuracy: 0.5300
    Epoch 1734/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8149 - Accuracy: 0.5020
    Epoch 1735/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8543 - Accuracy: 0.4910
    Epoch 1736/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8145 - Accuracy: 0.5020
    Epoch 1737/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8167 - Accuracy: 0.5290
    Epoch 1738/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8515 - Accuracy: 0.5050
    Epoch 1739/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8499 - Accuracy: 0.5010
    Epoch 1740/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9951 - Accuracy: 0.4680
    Epoch 1741/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9381 - Accuracy: 0.4820
    Epoch 1742/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8356 - Accuracy: 0.4680
    Epoch 1743/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8983 - Accuracy: 0.5000
    Epoch 1744/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8309 - Accuracy: 0.5240
    Epoch 1745/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8671 - Accuracy: 0.4850
    Epoch 1746/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0138 - Accuracy: 0.5210
    Epoch 1747/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9202 - Accuracy: 0.4980
    Epoch 1748/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8303 - Accuracy: 0.5200
    Epoch 1749/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8518 - Accuracy: 0.4850
    Epoch 1750/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9754 - Accuracy: 0.4820
    Epoch 1751/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8314 - Accuracy: 0.5010
    Epoch 1752/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8430 - Accuracy: 0.4760
    Epoch 1753/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8279 - Accuracy: 0.4760
    Epoch 1754/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8455 - Accuracy: 0.5000
    Epoch 1755/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8595 - Accuracy: 0.5050
    Epoch 1756/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8549 - Accuracy: 0.4990
    Epoch 1757/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8406 - Accuracy: 0.4970
    Epoch 1758/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8954 - Accuracy: 0.5360
    Epoch 1759/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0339 - Accuracy: 0.5080
    Epoch 1760/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8796 - Accuracy: 0.5460
    Epoch 1761/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8332 - Accuracy: 0.5260
    Epoch 1762/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8260 - Accuracy: 0.5440
    Epoch 1763/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8156 - Accuracy: 0.5040
    Epoch 1764/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8388 - Accuracy: 0.4960
    Epoch 1765/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8720 - Accuracy: 0.5110
    Epoch 1766/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8261 - Accuracy: 0.4990
    Epoch 1767/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8287 - Accuracy: 0.4850
    Epoch 1768/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9137 - Accuracy: 0.4740
    Epoch 1769/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8833 - Accuracy: 0.5080
    Epoch 1770/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0831 - Accuracy: 0.5180
    Epoch 1771/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8233 - Accuracy: 0.5290
    Epoch 1772/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8434 - Accuracy: 0.4900
    Epoch 1773/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9280 - Accuracy: 0.5440
    Epoch 1774/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8642 - Accuracy: 0.4770
    Epoch 1775/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9915 - Accuracy: 0.5210
    Epoch 1776/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9099 - Accuracy: 0.4990
    Epoch 1777/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8351 - Accuracy: 0.5090
    Epoch 1778/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8632 - Accuracy: 0.5210
    Epoch 1779/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8207 - Accuracy: 0.5150
    Epoch 1780/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8482 - Accuracy: 0.5090
    Epoch 1781/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9130 - Accuracy: 0.5010
    Epoch 1782/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8550 - Accuracy: 0.4970
    Epoch 1783/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8334 - Accuracy: 0.5160
    Epoch 1784/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8091 - Accuracy: 0.5090
    Epoch 1785/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8172 - Accuracy: 0.5150
    Epoch 1786/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8744 - Accuracy: 0.5190
    Epoch 1787/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8447 - Accuracy: 0.5140
    Epoch 1788/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8202 - Accuracy: 0.4730
    Epoch 1789/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9348 - Accuracy: 0.4730
    Epoch 1790/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8533 - Accuracy: 0.4800
    Epoch 1791/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8214 - Accuracy: 0.5150
    Epoch 1792/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8558 - Accuracy: 0.5120
    Epoch 1793/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8419 - Accuracy: 0.5070
    Epoch 1794/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8640 - Accuracy: 0.5430
    Epoch 1795/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9371 - Accuracy: 0.5060
    Epoch 1796/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9731 - Accuracy: 0.5140
    Epoch 1797/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8454 - Accuracy: 0.5330
    Epoch 1798/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8512 - Accuracy: 0.4940
    Epoch 1799/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8190 - Accuracy: 0.4990
    Epoch 1800/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8472 - Accuracy: 0.4910
    Epoch 1801/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9111 - Accuracy: 0.4710
    Epoch 1802/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8184 - Accuracy: 0.5180
    Epoch 1803/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8329 - Accuracy: 0.5060
    Epoch 1804/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8211 - Accuracy: 0.4430
    Epoch 1805/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8316 - Accuracy: 0.5030
    Epoch 1806/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8410 - Accuracy: 0.4790
    Epoch 1807/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8219 - Accuracy: 0.5150
    Epoch 1808/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8313 - Accuracy: 0.5100
    Epoch 1809/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8523 - Accuracy: 0.5120
    Epoch 1810/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8435 - Accuracy: 0.4740
    Epoch 1811/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8792 - Accuracy: 0.4910
    Epoch 1812/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8512 - Accuracy: 0.4700
    Epoch 1813/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8431 - Accuracy: 0.5030
    Epoch 1814/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8180 - Accuracy: 0.4890
    Epoch 1815/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8551 - Accuracy: 0.5240
    Epoch 1816/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8803 - Accuracy: 0.4930
    Epoch 1817/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8443 - Accuracy: 0.5120
    Epoch 1818/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8037 - Accuracy: 0.5110
    Epoch 1819/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9111 - Accuracy: 0.4960
    Epoch 1820/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8274 - Accuracy: 0.4710
    Epoch 1821/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8295 - Accuracy: 0.5020
    Epoch 1822/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9392 - Accuracy: 0.4870
    Epoch 1823/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8759 - Accuracy: 0.5050
    Epoch 1824/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8779 - Accuracy: 0.5150
    Epoch 1825/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8465 - Accuracy: 0.5070
    Epoch 1826/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8313 - Accuracy: 0.4930
    Epoch 1827/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8696 - Accuracy: 0.5450
    Epoch 1828/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8354 - Accuracy: 0.5040
    Epoch 1829/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8427 - Accuracy: 0.4750
    Epoch 1830/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9878 - Accuracy: 0.4750
    Epoch 1831/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8567 - Accuracy: 0.4730
    Epoch 1832/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8531 - Accuracy: 0.4900
    Epoch 1833/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9192 - Accuracy: 0.4920
    Epoch 1834/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8261 - Accuracy: 0.5260
    Epoch 1835/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8601 - Accuracy: 0.5000
    Epoch 1836/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8272 - Accuracy: 0.4680
    Epoch 1837/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8145 - Accuracy: 0.5220
    Epoch 1838/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8744 - Accuracy: 0.4960
    Epoch 1839/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8774 - Accuracy: 0.5110
    Epoch 1840/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8850 - Accuracy: 0.4860
    Epoch 1841/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8225 - Accuracy: 0.5270
    Epoch 1842/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8291 - Accuracy: 0.5090
    Epoch 1843/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9254 - Accuracy: 0.5100
    Epoch 1844/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0164 - Accuracy: 0.4800
    Epoch 1845/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8603 - Accuracy: 0.5040
    Epoch 1846/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8678 - Accuracy: 0.4760
    Epoch 1847/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8578 - Accuracy: 0.4740
    Epoch 1848/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8326 - Accuracy: 0.5090
    Epoch 1849/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8207 - Accuracy: 0.4940
    Epoch 1850/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8327 - Accuracy: 0.4920
    Epoch 1851/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8527 - Accuracy: 0.5160
    Epoch 1852/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8186 - Accuracy: 0.5220
    Epoch 1853/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8464 - Accuracy: 0.5060
    Epoch 1854/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8689 - Accuracy: 0.4970
    Epoch 1855/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9314 - Accuracy: 0.4860
    Epoch 1856/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8103 - Accuracy: 0.5170
    Epoch 1857/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8154 - Accuracy: 0.4940
    Epoch 1858/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8271 - Accuracy: 0.5110
    Epoch 1859/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8627 - Accuracy: 0.4860
    Epoch 1860/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8229 - Accuracy: 0.4860
    Epoch 1861/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8911 - Accuracy: 0.5100
    Epoch 1862/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8297 - Accuracy: 0.5300
    Epoch 1863/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1928 - Accuracy: 0.5080
    Epoch 1864/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8841 - Accuracy: 0.4860
    Epoch 1865/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8168 - Accuracy: 0.4970
    Epoch 1866/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8753 - Accuracy: 0.4920
    Epoch 1867/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8094 - Accuracy: 0.4920
    Epoch 1868/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8670 - Accuracy: 0.5270
    Epoch 1869/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8432 - Accuracy: 0.4930
    Epoch 1870/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8245 - Accuracy: 0.4990
    Epoch 1871/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8792 - Accuracy: 0.4770
    Epoch 1872/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8447 - Accuracy: 0.4900
    Epoch 1873/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8912 - Accuracy: 0.4920
    Epoch 1874/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8315 - Accuracy: 0.5200
    Epoch 1875/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8191 - Accuracy: 0.4920
    Epoch 1876/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9338 - Accuracy: 0.5140
    Epoch 1877/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8356 - Accuracy: 0.4990
    Epoch 1878/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8136 - Accuracy: 0.4950
    Epoch 1879/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8419 - Accuracy: 0.5150
    Epoch 1880/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9410 - Accuracy: 0.4960
    Epoch 1881/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8493 - Accuracy: 0.4750
    Epoch 1882/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8710 - Accuracy: 0.4840
    Epoch 1883/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8105 - Accuracy: 0.5150
    Epoch 1884/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8480 - Accuracy: 0.5010
    Epoch 1885/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8233 - Accuracy: 0.4990
    Epoch 1886/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8427 - Accuracy: 0.4990
    Epoch 1887/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0155 - Accuracy: 0.5060
    Epoch 1888/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8240 - Accuracy: 0.5060
    Epoch 1889/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8269 - Accuracy: 0.5330
    Epoch 1890/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8210 - Accuracy: 0.4840
    Epoch 1891/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8444 - Accuracy: 0.4850
    Epoch 1892/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8107 - Accuracy: 0.5350
    Epoch 1893/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8279 - Accuracy: 0.5190
    Epoch 1894/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8248 - Accuracy: 0.5000
    Epoch 1895/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8437 - Accuracy: 0.4850
    Epoch 1896/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8351 - Accuracy: 0.4940
    Epoch 1897/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8315 - Accuracy: 0.5190
    Epoch 1898/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8259 - Accuracy: 0.4890
    Epoch 1899/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8182 - Accuracy: 0.4760
    Epoch 1900/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8592 - Accuracy: 0.4720
    Epoch 1901/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8193 - Accuracy: 0.5030
    Epoch 1902/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8203 - Accuracy: 0.5070
    Epoch 1903/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8397 - Accuracy: 0.5100
    Epoch 1904/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9001 - Accuracy: 0.4830
    Epoch 1905/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9091 - Accuracy: 0.5150
    Epoch 1906/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8300 - Accuracy: 0.5280
    Epoch 1907/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8524 - Accuracy: 0.4960
    Epoch 1908/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8580 - Accuracy: 0.4890
    Epoch 1909/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9237 - Accuracy: 0.5270
    Epoch 1910/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8883 - Accuracy: 0.5010
    Epoch 1911/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8297 - Accuracy: 0.5170
    Epoch 1912/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8283 - Accuracy: 0.4980
    Epoch 1913/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9016 - Accuracy: 0.4960
    Epoch 1914/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8585 - Accuracy: 0.5150
    Epoch 1915/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8132 - Accuracy: 0.5330
    Epoch 1916/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8133 - Accuracy: 0.4990
    Epoch 1917/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8267 - Accuracy: 0.4950
    Epoch 1918/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9385 - Accuracy: 0.5220
    Epoch 1919/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8157 - Accuracy: 0.5220
    Epoch 1920/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9823 - Accuracy: 0.5170
    Epoch 1921/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8386 - Accuracy: 0.5120
    Epoch 1922/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8833 - Accuracy: 0.5150
    Epoch 1923/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8712 - Accuracy: 0.4530
    Epoch 1924/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8874 - Accuracy: 0.4600
    Epoch 1925/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8372 - Accuracy: 0.5050
    Epoch 1926/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8494 - Accuracy: 0.5000
    Epoch 1927/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8319 - Accuracy: 0.5280
    Epoch 1928/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8911 - Accuracy: 0.4700
    Epoch 1929/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8464 - Accuracy: 0.5070
    Epoch 1930/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8512 - Accuracy: 0.5240
    Epoch 1931/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8306 - Accuracy: 0.4730
    Epoch 1932/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8537 - Accuracy: 0.5020
    Epoch 1933/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9132 - Accuracy: 0.5080
    Epoch 1934/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9724 - Accuracy: 0.4760
    Epoch 1935/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8390 - Accuracy: 0.4990
    Epoch 1936/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8535 - Accuracy: 0.5170
    Epoch 1937/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9103 - Accuracy: 0.4820
    Epoch 1938/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8462 - Accuracy: 0.5010
    Epoch 1939/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8987 - Accuracy: 0.5000
    Epoch 1940/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8768 - Accuracy: 0.4730
    Epoch 1941/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8448 - Accuracy: 0.4750
    Epoch 1942/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8641 - Accuracy: 0.4920
    Epoch 1943/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9931 - Accuracy: 0.4910
    Epoch 1944/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8841 - Accuracy: 0.5150
    Epoch 1945/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8567 - Accuracy: 0.5080
    Epoch 1946/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8778 - Accuracy: 0.4990
    Epoch 1947/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8484 - Accuracy: 0.5220
    Epoch 1948/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8609 - Accuracy: 0.4990
    Epoch 1949/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9221 - Accuracy: 0.4990
    Epoch 1950/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8214 - Accuracy: 0.4780
    Epoch 1951/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8073 - Accuracy: 0.5420
    Epoch 1952/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8561 - Accuracy: 0.5290
    Epoch 1953/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8632 - Accuracy: 0.5230
    Epoch 1954/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8323 - Accuracy: 0.5060
    Epoch 1955/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8162 - Accuracy: 0.5520
    Epoch 1956/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8152 - Accuracy: 0.5020
    Epoch 1957/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8403 - Accuracy: 0.5100
    Epoch 1958/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8412 - Accuracy: 0.4950
    Epoch 1959/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9008 - Accuracy: 0.4900
    Epoch 1960/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8378 - Accuracy: 0.5050
    Epoch 1961/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8222 - Accuracy: 0.5400
    Epoch 1962/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9115 - Accuracy: 0.4720
    Epoch 1963/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9448 - Accuracy: 0.4890
    Epoch 1964/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9524 - Accuracy: 0.4850
    Epoch 1965/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8249 - Accuracy: 0.5020
    Epoch 1966/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9126 - Accuracy: 0.4860
    Epoch 1967/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8733 - Accuracy: 0.5110
    Epoch 1968/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8697 - Accuracy: 0.4750
    Epoch 1969/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8221 - Accuracy: 0.4800
    Epoch 1970/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8343 - Accuracy: 0.5340
    Epoch 1971/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8984 - Accuracy: 0.4930
    Epoch 1972/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8144 - Accuracy: 0.5150
    Epoch 1973/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8226 - Accuracy: 0.4840
    Epoch 1974/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8148 - Accuracy: 0.4980
    Epoch 1975/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8386 - Accuracy: 0.5200
    Epoch 1976/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8723 - Accuracy: 0.5150
    Epoch 1977/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8468 - Accuracy: 0.5590
    Epoch 1978/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8122 - Accuracy: 0.4960
    Epoch 1979/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8569 - Accuracy: 0.5020
    Epoch 1980/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8604 - Accuracy: 0.5200
    Epoch 1981/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8091 - Accuracy: 0.5370
    Epoch 1982/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8513 - Accuracy: 0.5200
    Epoch 1983/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8254 - Accuracy: 0.5060
    Epoch 1984/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8053 - Accuracy: 0.5320
    Epoch 1985/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8090 - Accuracy: 0.4940
    Epoch 1986/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9335 - Accuracy: 0.5020
    Epoch 1987/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8368 - Accuracy: 0.4770
    Epoch 1988/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8303 - Accuracy: 0.4830
    Epoch 1989/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8144 - Accuracy: 0.5500
    Epoch 1990/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8149 - Accuracy: 0.5000
    Epoch 1991/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8392 - Accuracy: 0.4750
    Epoch 1992/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8529 - Accuracy: 0.4690
    Epoch 1993/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8597 - Accuracy: 0.5000
    Epoch 1994/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9547 - Accuracy: 0.4620
    Epoch 1995/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8595 - Accuracy: 0.4990
    Epoch 1996/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8120 - Accuracy: 0.5250
    Epoch 1997/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8303 - Accuracy: 0.5320
    Epoch 1998/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9151 - Accuracy: 0.4980
    Epoch 1999/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1041 - Accuracy: 0.5130
    Epoch 2000/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8067 - Accuracy: 0.5310
    Epoch 2001/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9045 - Accuracy: 0.5030
    Epoch 2002/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8751 - Accuracy: 0.4840
    Epoch 2003/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8394 - Accuracy: 0.4870
    Epoch 2004/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8155 - Accuracy: 0.4890
    Epoch 2005/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8145 - Accuracy: 0.5290
    Epoch 2006/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8195 - Accuracy: 0.5080
    Epoch 2007/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8644 - Accuracy: 0.4870
    Epoch 2008/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8298 - Accuracy: 0.5040
    Epoch 2009/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8072 - Accuracy: 0.5420
    Epoch 2010/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8296 - Accuracy: 0.5040
    Epoch 2011/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9014 - Accuracy: 0.5020
    Epoch 2012/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8323 - Accuracy: 0.5350
    Epoch 2013/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8359 - Accuracy: 0.4660
    Epoch 2014/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9062 - Accuracy: 0.5050
    Epoch 2015/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8608 - Accuracy: 0.4800
    Epoch 2016/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8455 - Accuracy: 0.4990
    Epoch 2017/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8664 - Accuracy: 0.4980
    Epoch 2018/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8049 - Accuracy: 0.4970
    Epoch 2019/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.7999 - Accuracy: 0.5280
    Epoch 2020/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8606 - Accuracy: 0.4930
    Epoch 2021/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9551 - Accuracy: 0.4860
    Epoch 2022/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8540 - Accuracy: 0.5360
    Epoch 2023/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0188 - Accuracy: 0.4610
    Epoch 2024/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8259 - Accuracy: 0.5220
    Epoch 2025/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9242 - Accuracy: 0.4310
    Epoch 2026/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8847 - Accuracy: 0.4770
    Epoch 2027/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9198 - Accuracy: 0.4670
    Epoch 2028/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8044 - Accuracy: 0.5060
    Epoch 2029/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8419 - Accuracy: 0.5220
    Epoch 2030/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8999 - Accuracy: 0.5120
    Epoch 2031/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8265 - Accuracy: 0.4950
    Epoch 2032/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8797 - Accuracy: 0.4960
    Epoch 2033/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8895 - Accuracy: 0.5220
    Epoch 2034/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8440 - Accuracy: 0.4900
    Epoch 2035/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8754 - Accuracy: 0.4910
    Epoch 2036/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8291 - Accuracy: 0.4940
    Epoch 2037/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8513 - Accuracy: 0.4960
    Epoch 2038/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8523 - Accuracy: 0.4950
    Epoch 2039/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8683 - Accuracy: 0.4850
    Epoch 2040/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8429 - Accuracy: 0.4760
    Epoch 2041/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8848 - Accuracy: 0.4830
    Epoch 2042/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8409 - Accuracy: 0.4880
    Epoch 2043/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8179 - Accuracy: 0.4890
    Epoch 2044/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8249 - Accuracy: 0.5350
    Epoch 2045/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8210 - Accuracy: 0.4890
    Epoch 2046/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8429 - Accuracy: 0.5070
    Epoch 2047/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8564 - Accuracy: 0.5010
    Epoch 2048/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8738 - Accuracy: 0.4830
    Epoch 2049/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8314 - Accuracy: 0.5230
    Epoch 2050/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8341 - Accuracy: 0.5190
    Epoch 2051/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8297 - Accuracy: 0.5210
    Epoch 2052/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8251 - Accuracy: 0.4990
    Epoch 2053/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8892 - Accuracy: 0.4700
    Epoch 2054/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8345 - Accuracy: 0.4860
    Epoch 2055/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8233 - Accuracy: 0.5120
    Epoch 2056/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9264 - Accuracy: 0.4970
    Epoch 2057/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8160 - Accuracy: 0.5020
    Epoch 2058/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8947 - Accuracy: 0.4890
    Epoch 2059/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8780 - Accuracy: 0.4910
    Epoch 2060/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8821 - Accuracy: 0.5250
    Epoch 2061/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8064 - Accuracy: 0.5350
    Epoch 2062/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8221 - Accuracy: 0.5340
    Epoch 2063/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8126 - Accuracy: 0.5170
    Epoch 2064/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8471 - Accuracy: 0.5260
    Epoch 2065/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8307 - Accuracy: 0.4990
    Epoch 2066/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8984 - Accuracy: 0.5080
    Epoch 2067/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8242 - Accuracy: 0.5510
    Epoch 2068/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8256 - Accuracy: 0.5220
    Epoch 2069/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8164 - Accuracy: 0.5050
    Epoch 2070/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8512 - Accuracy: 0.4920
    Epoch 2071/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8483 - Accuracy: 0.5040
    Epoch 2072/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8206 - Accuracy: 0.4910
    Epoch 2073/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8193 - Accuracy: 0.5030
    Epoch 2074/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8307 - Accuracy: 0.4970
    Epoch 2075/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8055 - Accuracy: 0.5250
    Epoch 2076/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8028 - Accuracy: 0.5140
    Epoch 2077/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9235 - Accuracy: 0.5010
    Epoch 2078/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9020 - Accuracy: 0.4830
    Epoch 2079/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9430 - Accuracy: 0.4760
    Epoch 2080/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8230 - Accuracy: 0.5330
    Epoch 2081/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8091 - Accuracy: 0.5190
    Epoch 2082/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8177 - Accuracy: 0.5130
    Epoch 2083/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9052 - Accuracy: 0.4940
    Epoch 2084/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8611 - Accuracy: 0.4780
    Epoch 2085/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8300 - Accuracy: 0.5030
    Epoch 2086/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8554 - Accuracy: 0.5170
    Epoch 2087/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8556 - Accuracy: 0.5060
    Epoch 2088/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8513 - Accuracy: 0.4550
    Epoch 2089/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8679 - Accuracy: 0.5200
    Epoch 2090/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8609 - Accuracy: 0.5010
    Epoch 2091/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8848 - Accuracy: 0.5120
    Epoch 2092/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8630 - Accuracy: 0.5000
    Epoch 2093/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8820 - Accuracy: 0.5140
    Epoch 2094/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8326 - Accuracy: 0.4750
    Epoch 2095/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8356 - Accuracy: 0.4710
    Epoch 2096/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8928 - Accuracy: 0.4820
    Epoch 2097/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8215 - Accuracy: 0.5020
    Epoch 2098/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9619 - Accuracy: 0.4890
    Epoch 2099/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8130 - Accuracy: 0.5050
    Epoch 2100/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8966 - Accuracy: 0.5090
    Epoch 2101/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8173 - Accuracy: 0.5020
    Epoch 2102/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8086 - Accuracy: 0.5080
    Epoch 2103/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8702 - Accuracy: 0.5210
    Epoch 2104/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8405 - Accuracy: 0.5220
    Epoch 2105/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8333 - Accuracy: 0.5320
    Epoch 2106/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8324 - Accuracy: 0.4970
    Epoch 2107/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8324 - Accuracy: 0.4970
    Epoch 2108/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8245 - Accuracy: 0.5130
    Epoch 2109/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8313 - Accuracy: 0.5240
    Epoch 2110/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8535 - Accuracy: 0.5230
    Epoch 2111/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8422 - Accuracy: 0.5430
    Epoch 2112/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8003 - Accuracy: 0.5240
    Epoch 2113/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9063 - Accuracy: 0.5080
    Epoch 2114/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.7962 - Accuracy: 0.5590
    Epoch 2115/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9295 - Accuracy: 0.4780
    Epoch 2116/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8264 - Accuracy: 0.5290
    Epoch 2117/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8365 - Accuracy: 0.5310
    Epoch 2118/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8445 - Accuracy: 0.5030
    Epoch 2119/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8475 - Accuracy: 0.5050
    Epoch 2120/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8522 - Accuracy: 0.4850
    Epoch 2121/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8244 - Accuracy: 0.5330
    Epoch 2122/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9270 - Accuracy: 0.5400
    Epoch 2123/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8449 - Accuracy: 0.5070
    Epoch 2124/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8232 - Accuracy: 0.4960
    Epoch 2125/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8380 - Accuracy: 0.4780
    Epoch 2126/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8152 - Accuracy: 0.5070
    Epoch 2127/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8231 - Accuracy: 0.4850
    Epoch 2128/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8447 - Accuracy: 0.5200
    Epoch 2129/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8171 - Accuracy: 0.5530
    Epoch 2130/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8985 - Accuracy: 0.5370
    Epoch 2131/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8387 - Accuracy: 0.5080
    Epoch 2132/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8215 - Accuracy: 0.5670
    Epoch 2133/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9612 - Accuracy: 0.5120
    Epoch 2134/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9139 - Accuracy: 0.4800
    Epoch 2135/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9171 - Accuracy: 0.4900
    Epoch 2136/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8173 - Accuracy: 0.5250
    Epoch 2137/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8848 - Accuracy: 0.5050
    Epoch 2138/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8318 - Accuracy: 0.5010
    Epoch 2139/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8911 - Accuracy: 0.4840
    Epoch 2140/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8840 - Accuracy: 0.4920
    Epoch 2141/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8428 - Accuracy: 0.5150
    Epoch 2142/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8295 - Accuracy: 0.5310
    Epoch 2143/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.1175 - Accuracy: 0.5160
    Epoch 2144/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8246 - Accuracy: 0.5300
    Epoch 2145/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8595 - Accuracy: 0.4890
    Epoch 2146/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8314 - Accuracy: 0.5260
    Epoch 2147/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9412 - Accuracy: 0.4970
    Epoch 2148/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9079 - Accuracy: 0.5060
    Epoch 2149/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9909 - Accuracy: 0.4990
    Epoch 2150/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8743 - Accuracy: 0.4930
    Epoch 2151/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8794 - Accuracy: 0.4940
    Epoch 2152/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8712 - Accuracy: 0.5270
    Epoch 2153/3000
    32/32 [==============================] - 0s 4ms/step - loss: 1.0259 - Accuracy: 0.4640
    Epoch 2154/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8232 - Accuracy: 0.5090
    Epoch 2155/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8653 - Accuracy: 0.5350
    Epoch 2156/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.7948 - Accuracy: 0.5560
    Epoch 2157/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8385 - Accuracy: 0.5540
    Epoch 2158/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8311 - Accuracy: 0.4970
    Epoch 2159/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8551 - Accuracy: 0.5370
    Epoch 2160/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8822 - Accuracy: 0.4910
    Epoch 2161/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8392 - Accuracy: 0.5080
    Epoch 2162/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8944 - Accuracy: 0.5230
    Epoch 2163/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8142 - Accuracy: 0.5030
    Epoch 2164/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8230 - Accuracy: 0.5050
    Epoch 2165/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8436 - Accuracy: 0.4890
    Epoch 2166/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9903 - Accuracy: 0.4990
    Epoch 2167/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8149 - Accuracy: 0.5350
    Epoch 2168/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8658 - Accuracy: 0.5240
    Epoch 2169/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8187 - Accuracy: 0.5110
    Epoch 2170/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8573 - Accuracy: 0.4990
    Epoch 2171/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8399 - Accuracy: 0.5340
    Epoch 2172/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8379 - Accuracy: 0.4960
    Epoch 2173/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8569 - Accuracy: 0.4960
    Epoch 2174/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8065 - Accuracy: 0.5370
    Epoch 2175/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8104 - Accuracy: 0.5180
    Epoch 2176/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8586 - Accuracy: 0.5270
    Epoch 2177/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8206 - Accuracy: 0.5350
    Epoch 2178/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8226 - Accuracy: 0.5330
    Epoch 2179/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8109 - Accuracy: 0.5230
    Epoch 2180/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8166 - Accuracy: 0.4790
    Epoch 2181/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8265 - Accuracy: 0.5080
    Epoch 2182/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8725 - Accuracy: 0.5420
    Epoch 2183/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8697 - Accuracy: 0.5430
    Epoch 2184/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8725 - Accuracy: 0.4720
    Epoch 2185/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8608 - Accuracy: 0.4800
    Epoch 2186/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9330 - Accuracy: 0.5090
    Epoch 2187/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8377 - Accuracy: 0.5040
    Epoch 2188/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8268 - Accuracy: 0.5240
    Epoch 2189/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8678 - Accuracy: 0.5110
    Epoch 2190/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8118 - Accuracy: 0.5430
    Epoch 2191/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9063 - Accuracy: 0.4890
    Epoch 2192/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8289 - Accuracy: 0.5240
    Epoch 2193/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8206 - Accuracy: 0.5290
    Epoch 2194/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8176 - Accuracy: 0.5080
    Epoch 2195/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8669 - Accuracy: 0.5190
    Epoch 2196/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8563 - Accuracy: 0.4960
    Epoch 2197/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8923 - Accuracy: 0.5270
    Epoch 2198/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8998 - Accuracy: 0.5320
    Epoch 2199/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8165 - Accuracy: 0.5080
    Epoch 2200/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8609 - Accuracy: 0.5150
    Epoch 2201/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8831 - Accuracy: 0.5340
    Epoch 2202/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8076 - Accuracy: 0.5340
    Epoch 2203/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8405 - Accuracy: 0.5080
    Epoch 2204/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8275 - Accuracy: 0.4710
    Epoch 2205/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8252 - Accuracy: 0.5130
    Epoch 2206/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.7994 - Accuracy: 0.5360
    Epoch 2207/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8726 - Accuracy: 0.5030
    Epoch 2208/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9550 - Accuracy: 0.4990
    Epoch 2209/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9281 - Accuracy: 0.5120
    Epoch 2210/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8495 - Accuracy: 0.5240
    Epoch 2211/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8831 - Accuracy: 0.5560
    Epoch 2212/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9273 - Accuracy: 0.5120
    Epoch 2213/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8267 - Accuracy: 0.5010
    Epoch 2214/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8466 - Accuracy: 0.5310
    Epoch 2215/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8212 - Accuracy: 0.4870
    Epoch 2216/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8166 - Accuracy: 0.5090
    Epoch 2217/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8557 - Accuracy: 0.4910
    Epoch 2218/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8653 - Accuracy: 0.5390
    Epoch 2219/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9717 - Accuracy: 0.5310
    Epoch 2220/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8478 - Accuracy: 0.5210
    Epoch 2221/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8122 - Accuracy: 0.5120
    Epoch 2222/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8504 - Accuracy: 0.5140
    Epoch 2223/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8291 - Accuracy: 0.5330
    Epoch 2224/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8338 - Accuracy: 0.5010
    Epoch 2225/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8499 - Accuracy: 0.4930
    Epoch 2226/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8285 - Accuracy: 0.5330
    Epoch 2227/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8306 - Accuracy: 0.5120
    Epoch 2228/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8387 - Accuracy: 0.4820
    Epoch 2229/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8247 - Accuracy: 0.5210
    Epoch 2230/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8163 - Accuracy: 0.5150
    Epoch 2231/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8993 - Accuracy: 0.4800
    Epoch 2232/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8465 - Accuracy: 0.4950
    Epoch 2233/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8403 - Accuracy: 0.5380
    Epoch 2234/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9481 - Accuracy: 0.5400
    Epoch 2235/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8878 - Accuracy: 0.5250
    Epoch 2236/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9007 - Accuracy: 0.5240
    Epoch 2237/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.9500 - Accuracy: 0.5090
    Epoch 2238/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8116 - Accuracy: 0.5350
    Epoch 2239/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8644 - Accuracy: 0.5010
    Epoch 2240/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8295 - Accuracy: 0.5280
    Epoch 2241/3000
    32/32 [==============================] - 0s 4ms/step - loss: 0.8722 - Accuracy: 0.5390
    Epoch 2242/3000
    15/32 [=============>................] - ETA: 0s - loss: 0.8526 - Accuracy: 0.5188


```python
#subset
fig = plt.figure(figsize=(10, 10))
plt.subplot(4, 1, 1)
plt.scatter(X_train[:1000,0], X_train[:1000,1], c=y_train[:1000])
plt.title('small train set')

```




    Text(0.5, 1.0, 'small train set')




    
![png](output_46_1.png)
    



```python
pred_mean, pred_stdv = infer(X_test, y_test, small_epistemic_model, 10) 

fig = plt.figure(figsize=(10, 10))
plt.subplot(4, 1, 1)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
plt.title('test set')

plt.subplot(4,2,3)
plt.scatter(X_test[:,0], X_test[:,1], c=pred_mean.round())
plt.title('predictions (bigger train set)')

plt.subplot(4,2,5)
plt.scatter(X_test[:,0], X_test[:,1], c=pred_mean)
plt.title('predictions (color gradient on prediction mean i.e sharpness)')

plt.subplot(4,2,7)
plt.scatter(X_test[:,0], X_test[:,1], c=pred_stdv)
plt.title('standard deviation on the inference, the closer to yellow => higher epistemic uncertainty')

plt.tight_layout()

```

    accuracy :  0.9125



    
![png](output_47_1.png)
    


## Epistemic regression


```python
X = np.linspace(-1, 1, 300)
y = 4 * X * np.cos(np.pi * np.sin(X)) + 1 + np.random.randn(X.shape[0]) * 0.5
plt.scatter(X, y)
```




    <matplotlib.collections.PathCollection at 0x13e9ccd15b0>




    
![png](output_49_1.png)
    



```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
```

    (225,) (75,)
    (225,) (75,)



```python
plt.scatter(X_train, y_train, alpha=0.3)
plt.scatter(X_test, y_test)
```




    <matplotlib.collections.PathCollection at 0x1e0c7649250>




    
![png](output_51_1.png)
    



```python
# Prior is not trainable
def prior(kernel_size, bias_size, dtype = None):
    n = kernel_size + bias_size # num of params
    return Sequential([
       tfp.layers.DistributionLambda(
           lambda t: tfd.Laplace(loc = tf.zeros(n), scale= 2 * tf.ones(n))
       )                     
  ])

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(

          tfd.Normal(loc=t[..., :n],
                     scale= 1e-5 + 0.003 * tf.nn.softplus(t[..., n:])),
          reinterpreted_batch_ndims=1)),
    ])
```


```python
def epistemic_regressor():
  inputs = keras.Input(shape=(1,))

  x = tfp.layers.DenseVariational(units=128, 
                                  make_prior_fn=prior,
                                  make_posterior_fn=posterior,
                                  kl_weight = 1 / X_train.shape[0],
                                  activation='relu')(inputs)
  
  x = tfp.layers.DenseVariational(units=64, 
                                  make_prior_fn=prior,
                                  make_posterior_fn=posterior,
                                  kl_weight = 1 / X_train.shape[0],
                                  activation='relu')(x)
                    
  outputs = tfp.layers.DenseVariational(units=1, 
                                  make_prior_fn=prior,
                                  make_posterior_fn=posterior,
                                  kl_weight = 1 / X_train.shape[0])(x)

  model = keras.Model(inputs=inputs, outputs=outputs, name='epistemic_BNN')

  opt = keras.optimizers.Adam(learning_rate=0.001)

  model.compile(loss='mse', optimizer=opt)
  model.summary()

  return model
  
```


```python
def train_regressor(X_train, X_test, y_train, y_test, model, epochs=100, batch_size=32, verbose=0):
  device_name = tf.test.gpu_device_name()
  if len(device_name) > 0:
    print("Found GPU at: {}".format(device_name))
  else:
    device_name = "/device:CPU:0"
    print("No GPU, using {}.".format(device_name))

  with tf.device(device_name):
    model.fit(X_train, y_train, batch_size=batch_size, verbose=verbose, epochs=epochs, 
                                validation_data=(X_test, y_test),) 
    
    
  return model 
```


```python
regressor = epistemic_regressor()
regressor = train_regressor(X_train, X_test, y_train, y_test, regressor, verbose=1, epochs=2000)
```

    Model: "epistemic_BNN"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_12 (InputLayer)       [(None, 1)]               0         
                                                                     
     dense_variational_10 (Dense  (None, 128)              512       
     Variational)                                                    
                                                                     
     dense_variational_11 (Dense  (None, 64)               16512     
     Variational)                                                    
                                                                     
     dense_variational_12 (Dense  (None, 1)                130       
     Variational)                                                    
                                                                     
    =================================================================
    Total params: 17,154
    Trainable params: 17,154
    Non-trainable params: 0
    _________________________________________________________________
    No GPU, using /device:CPU:0.
    Epoch 1/2000
    8/8 [==============================] - 1s 49ms/step - loss: 1440328.0000 - val_loss: 1439146.2500
    Epoch 2/2000
    8/8 [==============================] - 0s 5ms/step - loss: 1438762.7500 - val_loss: 1436514.3750
    Epoch 3/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1437982.0000 - val_loss: 1436660.0000
    Epoch 4/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1434294.3750 - val_loss: 1432834.8750
    Epoch 5/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1433392.6250 - val_loss: 1431637.0000
    Epoch 6/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1431163.5000 - val_loss: 1427465.7500
    Epoch 7/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1430565.1250 - val_loss: 1429938.5000
    Epoch 8/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1427990.6250 - val_loss: 1426667.6250
    Epoch 9/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1426036.5000 - val_loss: 1424314.3750
    Epoch 10/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1423875.2500 - val_loss: 1422988.8750
    Epoch 11/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1423384.7500 - val_loss: 1421806.7500
    Epoch 12/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1422455.0000 - val_loss: 1418999.6250
    Epoch 13/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1420559.6250 - val_loss: 1418476.7500
    Epoch 14/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1417241.8750 - val_loss: 1417151.0000
    Epoch 15/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1416318.8750 - val_loss: 1417629.2500
    Epoch 16/2000
    8/8 [==============================] - 0s 5ms/step - loss: 1414807.7500 - val_loss: 1410804.1250
    Epoch 17/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1413373.2500 - val_loss: 1411602.6250
    Epoch 18/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1410836.0000 - val_loss: 1409362.2500
    Epoch 19/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1410457.0000 - val_loss: 1409475.6250
    Epoch 20/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1406935.3750 - val_loss: 1405262.0000
    Epoch 21/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1406011.0000 - val_loss: 1405835.2500
    Epoch 22/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1404698.6250 - val_loss: 1405244.2500
    Epoch 23/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1403708.6250 - val_loss: 1402529.7500
    Epoch 24/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1401878.8750 - val_loss: 1402986.5000
    Epoch 25/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1399559.1250 - val_loss: 1397898.0000
    Epoch 26/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1397474.2500 - val_loss: 1397342.6250
    Epoch 27/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1395596.5000 - val_loss: 1396493.5000
    Epoch 28/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1395624.5000 - val_loss: 1396435.1250
    Epoch 29/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1392785.6250 - val_loss: 1393058.5000
    Epoch 30/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1391839.6250 - val_loss: 1390794.3750
    Epoch 31/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1390811.0000 - val_loss: 1388160.1250
    Epoch 32/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1389158.1250 - val_loss: 1387073.8750
    Epoch 33/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1388338.8750 - val_loss: 1385578.5000
    Epoch 34/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1385559.0000 - val_loss: 1386432.7500
    Epoch 35/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1383815.2500 - val_loss: 1382247.8750
    Epoch 36/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1383539.5000 - val_loss: 1385460.5000
    Epoch 37/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1379608.7500 - val_loss: 1379588.5000
    Epoch 38/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1379685.7500 - val_loss: 1380176.8750
    Epoch 39/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1377935.7500 - val_loss: 1377604.5000
    Epoch 40/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1375444.3750 - val_loss: 1376548.7500
    Epoch 41/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1375893.2500 - val_loss: 1372256.3750
    Epoch 42/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1370881.8750 - val_loss: 1372430.2500
    Epoch 43/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1372082.6250 - val_loss: 1369769.6250
    Epoch 44/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1369386.5000 - val_loss: 1371156.0000
    Epoch 45/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1367743.7500 - val_loss: 1369156.5000
    Epoch 46/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1366661.0000 - val_loss: 1365081.5000
    Epoch 47/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1365130.2500 - val_loss: 1363547.1250
    Epoch 48/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1363696.2500 - val_loss: 1364701.7500
    Epoch 49/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1359571.8750 - val_loss: 1358588.6250
    Epoch 50/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1360629.6250 - val_loss: 1360391.8750
    Epoch 51/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1358672.8750 - val_loss: 1359293.7500
    Epoch 52/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1357594.7500 - val_loss: 1354206.2500
    Epoch 53/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1356709.0000 - val_loss: 1355623.1250
    Epoch 54/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1353325.3750 - val_loss: 1351923.0000
    Epoch 55/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1354115.2500 - val_loss: 1352885.1250
    Epoch 56/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1351728.2500 - val_loss: 1351150.2500
    Epoch 57/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1350401.1250 - val_loss: 1350537.2500
    Epoch 58/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1346829.0000 - val_loss: 1346565.8750
    Epoch 59/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1346407.6250 - val_loss: 1345668.0000
    Epoch 60/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1345455.8750 - val_loss: 1346362.8750
    Epoch 61/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1344594.8750 - val_loss: 1342674.0000
    Epoch 62/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1342089.1250 - val_loss: 1341400.2500
    Epoch 63/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1340515.8750 - val_loss: 1337852.6250
    Epoch 64/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1339509.6250 - val_loss: 1337953.7500
    Epoch 65/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1338775.5000 - val_loss: 1339883.2500
    Epoch 66/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1336724.7500 - val_loss: 1336057.7500
    Epoch 67/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1336247.6250 - val_loss: 1333905.5000
    Epoch 68/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1334180.2500 - val_loss: 1333551.8750
    Epoch 69/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1332590.0000 - val_loss: 1331165.8750
    Epoch 70/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1330811.8750 - val_loss: 1331322.7500
    Epoch 71/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1329172.8750 - val_loss: 1329681.3750
    Epoch 72/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1329728.7500 - val_loss: 1326119.3750
    Epoch 73/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1325662.8750 - val_loss: 1324934.5000
    Epoch 74/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1324661.8750 - val_loss: 1324625.8750
    Epoch 75/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1324008.1250 - val_loss: 1322348.8750
    Epoch 76/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1321995.6250 - val_loss: 1321079.2500
    Epoch 77/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1319860.0000 - val_loss: 1319576.8750
    Epoch 78/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1320129.6250 - val_loss: 1321296.3750
    Epoch 79/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1318751.2500 - val_loss: 1317144.3750
    Epoch 80/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1314928.3750 - val_loss: 1316497.8750
    Epoch 81/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1317028.7500 - val_loss: 1313445.7500
    Epoch 82/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1313239.3750 - val_loss: 1313335.0000
    Epoch 83/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1313625.6250 - val_loss: 1310972.5000
    Epoch 84/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1312098.0000 - val_loss: 1310277.5000
    Epoch 85/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1308516.1250 - val_loss: 1310991.5000
    Epoch 86/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1308048.5000 - val_loss: 1308308.5000
    Epoch 87/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1305983.8750 - val_loss: 1306921.1250
    Epoch 88/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1306107.7500 - val_loss: 1303262.7500
    Epoch 89/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1304520.0000 - val_loss: 1304275.5000
    Epoch 90/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1302494.7500 - val_loss: 1300842.2500
    Epoch 91/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1302175.8750 - val_loss: 1301027.5000
    Epoch 92/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1300694.6250 - val_loss: 1297454.3750
    Epoch 93/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1299599.5000 - val_loss: 1299527.6250
    Epoch 94/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1298093.7500 - val_loss: 1298711.5000
    Epoch 95/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1296557.6250 - val_loss: 1294674.7500
    Epoch 96/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1295120.5000 - val_loss: 1294045.2500
    Epoch 97/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1292390.6250 - val_loss: 1292602.1250
    Epoch 98/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1290529.1250 - val_loss: 1289978.3750
    Epoch 99/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1290304.2500 - val_loss: 1289805.0000
    Epoch 100/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1290022.2500 - val_loss: 1288205.8750
    Epoch 101/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1287160.1250 - val_loss: 1286753.7500
    Epoch 102/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1287008.6250 - val_loss: 1284416.2500
    Epoch 103/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1285422.5000 - val_loss: 1283624.1250
    Epoch 104/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1284342.7500 - val_loss: 1283402.2500
    Epoch 105/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1283127.5000 - val_loss: 1283352.0000
    Epoch 106/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1280473.8750 - val_loss: 1281911.3750
    Epoch 107/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1280074.0000 - val_loss: 1278978.7500
    Epoch 108/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1279288.8750 - val_loss: 1277530.8750
    Epoch 109/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1276121.5000 - val_loss: 1277714.2500
    Epoch 110/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1277062.8750 - val_loss: 1274156.1250
    Epoch 111/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1275801.6250 - val_loss: 1275292.2500
    Epoch 112/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1274578.0000 - val_loss: 1272833.8750
    Epoch 113/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1272765.8750 - val_loss: 1272236.1250
    Epoch 114/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1270721.3750 - val_loss: 1270430.3750
    Epoch 115/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1270620.2500 - val_loss: 1269353.7500
    Epoch 116/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1268479.2500 - val_loss: 1267667.7500
    Epoch 117/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1265889.0000 - val_loss: 1265019.5000
    Epoch 118/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1264919.3750 - val_loss: 1263976.0000
    Epoch 119/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1264499.8750 - val_loss: 1266249.5000
    Epoch 120/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1263162.2500 - val_loss: 1260477.0000
    Epoch 121/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1262880.8750 - val_loss: 1261936.2500
    Epoch 122/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1260045.1250 - val_loss: 1260705.8750
    Epoch 123/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1259745.3750 - val_loss: 1257656.2500
    Epoch 124/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1257603.0000 - val_loss: 1257454.5000
    Epoch 125/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1256949.5000 - val_loss: 1255033.5000
    Epoch 126/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1255491.7500 - val_loss: 1254989.3750
    Epoch 127/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1255559.3750 - val_loss: 1256111.7500
    Epoch 128/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1255293.5000 - val_loss: 1253273.1250
    Epoch 129/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1252504.7500 - val_loss: 1251242.0000
    Epoch 130/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1250822.1250 - val_loss: 1248387.8750
    Epoch 131/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1250442.6250 - val_loss: 1246177.8750
    Epoch 132/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1250074.0000 - val_loss: 1248136.3750
    Epoch 133/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1247429.1250 - val_loss: 1245346.3750
    Epoch 134/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1246425.8750 - val_loss: 1244783.1250
    Epoch 135/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1245315.3750 - val_loss: 1243525.2500
    Epoch 136/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1243774.6250 - val_loss: 1245883.2500
    Epoch 137/2000


    8/8 [==============================] - 0s 4ms/step - loss: 1243050.3750 - val_loss: 1241332.2500
    Epoch 138/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1240920.0000 - val_loss: 1242470.0000
    Epoch 139/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1241562.5000 - val_loss: 1238588.2500
    Epoch 140/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1238054.1250 - val_loss: 1238285.5000
    Epoch 141/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1238738.2500 - val_loss: 1235728.0000
    Epoch 142/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1236759.6250 - val_loss: 1237192.3750
    Epoch 143/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1236092.0000 - val_loss: 1233981.8750
    Epoch 144/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1233634.0000 - val_loss: 1234341.7500
    Epoch 145/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1234802.3750 - val_loss: 1231456.3750
    Epoch 146/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1230942.0000 - val_loss: 1232240.1250
    Epoch 147/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1231139.1250 - val_loss: 1229200.2500
    Epoch 148/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1229181.2500 - val_loss: 1226443.2500
    Epoch 149/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1228344.5000 - val_loss: 1226642.5000
    Epoch 150/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1227107.0000 - val_loss: 1227903.8750
    Epoch 151/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1227131.2500 - val_loss: 1226953.7500
    Epoch 152/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1224282.7500 - val_loss: 1224584.6250
    Epoch 153/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1224612.2500 - val_loss: 1224641.7500
    Epoch 154/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1222542.7500 - val_loss: 1221675.7500
    Epoch 155/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1221287.5000 - val_loss: 1219850.0000
    Epoch 156/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1220303.5000 - val_loss: 1219898.0000
    Epoch 157/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1219503.3750 - val_loss: 1219494.7500
    Epoch 158/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1219975.6250 - val_loss: 1216057.8750
    Epoch 159/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1217341.5000 - val_loss: 1216461.6250
    Epoch 160/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1216511.7500 - val_loss: 1215204.8750
    Epoch 161/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1215632.5000 - val_loss: 1215670.1250
    Epoch 162/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1213944.1250 - val_loss: 1215005.7500
    Epoch 163/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1213441.8750 - val_loss: 1211290.7500
    Epoch 164/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1212685.5000 - val_loss: 1211291.5000
    Epoch 165/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1211405.0000 - val_loss: 1210913.5000
    Epoch 166/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1209277.5000 - val_loss: 1208377.8750
    Epoch 167/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1208389.0000 - val_loss: 1207781.6250
    Epoch 168/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1207796.8750 - val_loss: 1206234.0000
    Epoch 169/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1207188.2500 - val_loss: 1206120.5000
    Epoch 170/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1204151.1250 - val_loss: 1205639.7500
    Epoch 171/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1203202.7500 - val_loss: 1205556.3750
    Epoch 172/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1204130.8750 - val_loss: 1202259.7500
    Epoch 173/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1203079.0000 - val_loss: 1199653.5000
    Epoch 174/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1201353.0000 - val_loss: 1199257.3750
    Epoch 175/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1199046.6250 - val_loss: 1200543.7500
    Epoch 176/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1198741.3750 - val_loss: 1200194.7500
    Epoch 177/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1197468.8750 - val_loss: 1195984.6250
    Epoch 178/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1195523.2500 - val_loss: 1197702.1250
    Epoch 179/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1196697.3750 - val_loss: 1196577.8750
    Epoch 180/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1195581.0000 - val_loss: 1191928.3750
    Epoch 181/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1193926.3750 - val_loss: 1193820.1250
    Epoch 182/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1193373.2500 - val_loss: 1193879.6250
    Epoch 183/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1192144.8750 - val_loss: 1192045.6250
    Epoch 184/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1189536.7500 - val_loss: 1189556.5000
    Epoch 185/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1189259.1250 - val_loss: 1190382.1250
    Epoch 186/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1189339.5000 - val_loss: 1191107.5000
    Epoch 187/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1186000.2500 - val_loss: 1186373.0000
    Epoch 188/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1184248.1250 - val_loss: 1182456.6250
    Epoch 189/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1186092.3750 - val_loss: 1184563.2500
    Epoch 190/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1183822.1250 - val_loss: 1185413.0000
    Epoch 191/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1182757.5000 - val_loss: 1179703.8750
    Epoch 192/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1181226.2500 - val_loss: 1180808.2500
    Epoch 193/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1180635.2500 - val_loss: 1181531.5000
    Epoch 194/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1179102.7500 - val_loss: 1177044.8750
    Epoch 195/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1179893.6250 - val_loss: 1178177.1250
    Epoch 196/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1177005.7500 - val_loss: 1175694.0000
    Epoch 197/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1177008.5000 - val_loss: 1175614.3750
    Epoch 198/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1175105.1250 - val_loss: 1174213.5000
    Epoch 199/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1174894.7500 - val_loss: 1174128.7500
    Epoch 200/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1173611.1250 - val_loss: 1175231.7500
    Epoch 201/2000
    8/8 [==============================] - 0s 5ms/step - loss: 1172409.2500 - val_loss: 1174138.8750
    Epoch 202/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1171598.7500 - val_loss: 1173271.3750
    Epoch 203/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1171005.5000 - val_loss: 1171340.6250
    Epoch 204/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1169707.2500 - val_loss: 1168435.2500
    Epoch 205/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1168876.7500 - val_loss: 1170698.5000
    Epoch 206/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1166746.1250 - val_loss: 1167811.2500
    Epoch 207/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1169052.2500 - val_loss: 1167219.2500
    Epoch 208/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1166287.0000 - val_loss: 1163943.5000
    Epoch 209/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1165094.7500 - val_loss: 1166831.2500
    Epoch 210/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1164588.6250 - val_loss: 1163122.5000
    Epoch 211/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1162888.8750 - val_loss: 1164840.5000
    Epoch 212/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1163258.6250 - val_loss: 1161733.8750
    Epoch 213/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1161598.7500 - val_loss: 1160397.6250
    Epoch 214/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1160239.5000 - val_loss: 1158287.5000
    Epoch 215/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1159301.5000 - val_loss: 1161108.2500
    Epoch 216/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1157691.5000 - val_loss: 1158746.2500
    Epoch 217/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1157157.7500 - val_loss: 1156761.8750
    Epoch 218/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1156640.3750 - val_loss: 1157072.8750
    Epoch 219/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1153756.6250 - val_loss: 1154385.8750
    Epoch 220/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1154280.1250 - val_loss: 1152671.6250
    Epoch 221/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1153494.5000 - val_loss: 1150967.1250
    Epoch 222/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1153250.0000 - val_loss: 1152825.3750
    Epoch 223/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1151497.5000 - val_loss: 1148767.1250
    Epoch 224/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1151429.0000 - val_loss: 1150666.2500
    Epoch 225/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1150109.8750 - val_loss: 1147713.0000
    Epoch 226/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1148195.2500 - val_loss: 1150928.3750
    Epoch 227/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1147481.3750 - val_loss: 1147079.2500
    Epoch 228/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1147642.7500 - val_loss: 1142861.3750
    Epoch 229/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1144269.0000 - val_loss: 1147559.1250
    Epoch 230/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1146045.7500 - val_loss: 1144752.5000
    Epoch 231/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1142581.7500 - val_loss: 1145383.5000
    Epoch 232/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1143863.3750 - val_loss: 1145310.5000
    Epoch 233/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1141564.6250 - val_loss: 1142979.3750
    Epoch 234/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1142438.5000 - val_loss: 1138826.8750
    Epoch 235/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1138883.7500 - val_loss: 1140717.6250
    Epoch 236/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1139108.7500 - val_loss: 1140860.7500
    Epoch 237/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1137453.5000 - val_loss: 1141536.8750
    Epoch 238/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1139122.0000 - val_loss: 1136607.0000
    Epoch 239/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1137449.8750 - val_loss: 1136561.3750
    Epoch 240/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1136553.7500 - val_loss: 1134814.5000
    Epoch 241/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1134854.0000 - val_loss: 1136524.7500
    Epoch 242/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1133907.8750 - val_loss: 1132706.6250
    Epoch 243/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1133865.8750 - val_loss: 1133263.0000
    Epoch 244/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1132351.1250 - val_loss: 1132401.7500
    Epoch 245/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1131382.0000 - val_loss: 1132280.3750
    Epoch 246/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1130268.1250 - val_loss: 1129847.7500
    Epoch 247/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1128994.1250 - val_loss: 1128348.1250
    Epoch 248/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1129406.3750 - val_loss: 1125956.2500
    Epoch 249/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1129311.3750 - val_loss: 1125655.7500
    Epoch 250/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1127138.6250 - val_loss: 1125898.8750
    Epoch 251/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1125612.3750 - val_loss: 1127676.3750
    Epoch 252/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1126515.7500 - val_loss: 1126895.5000
    Epoch 253/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1124801.8750 - val_loss: 1124126.7500
    Epoch 254/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1123289.0000 - val_loss: 1121797.2500
    Epoch 255/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1122584.6250 - val_loss: 1120864.3750
    Epoch 256/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1120722.6250 - val_loss: 1121827.1250
    Epoch 257/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1122024.5000 - val_loss: 1122224.7500
    Epoch 258/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1122113.2500 - val_loss: 1121549.3750
    Epoch 259/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1118639.5000 - val_loss: 1113501.2500
    Epoch 260/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1117804.8750 - val_loss: 1119039.6250
    Epoch 261/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1117211.6250 - val_loss: 1118675.1250
    Epoch 262/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1117945.1250 - val_loss: 1115617.2500
    Epoch 263/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1115715.7500 - val_loss: 1113925.0000
    Epoch 264/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1115758.2500 - val_loss: 1116892.2500
    Epoch 265/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1114907.5000 - val_loss: 1111576.8750
    Epoch 266/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1114424.3750 - val_loss: 1109473.5000
    Epoch 267/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1111657.6250 - val_loss: 1109339.2500
    Epoch 268/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1111188.5000 - val_loss: 1108888.6250
    Epoch 269/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1110882.2500 - val_loss: 1109885.0000
    Epoch 270/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1109394.2500 - val_loss: 1108821.5000
    Epoch 271/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1109015.6250 - val_loss: 1109892.0000
    Epoch 272/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1107039.1250 - val_loss: 1106667.0000
    Epoch 273/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1107372.5000 - val_loss: 1107578.8750
    Epoch 274/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1106303.6250 - val_loss: 1107757.3750
    Epoch 275/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1106004.6250 - val_loss: 1104806.3750
    Epoch 276/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1104452.2500 - val_loss: 1105680.3750
    Epoch 277/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1104062.8750 - val_loss: 1104657.3750
    Epoch 278/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1104778.8750 - val_loss: 1099234.3750
    Epoch 279/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1102698.2500 - val_loss: 1101658.3750
    Epoch 280/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1102105.5000 - val_loss: 1100613.3750
    Epoch 281/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1101079.5000 - val_loss: 1101900.8750
    Epoch 282/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1100795.0000 - val_loss: 1100302.5000
    Epoch 283/2000


    8/8 [==============================] - 0s 4ms/step - loss: 1100497.1250 - val_loss: 1097284.0000
    Epoch 284/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1100528.3750 - val_loss: 1097513.5000
    Epoch 285/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1098104.5000 - val_loss: 1095923.0000
    Epoch 286/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1097462.0000 - val_loss: 1097005.7500
    Epoch 287/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1097111.2500 - val_loss: 1093925.3750
    Epoch 288/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1095082.1250 - val_loss: 1094128.0000
    Epoch 289/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1093357.8750 - val_loss: 1092120.3750
    Epoch 290/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1093928.0000 - val_loss: 1093034.8750
    Epoch 291/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1092891.6250 - val_loss: 1091100.0000
    Epoch 292/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1092162.1250 - val_loss: 1094811.7500
    Epoch 293/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1090982.1250 - val_loss: 1088302.1250
    Epoch 294/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1091286.6250 - val_loss: 1089505.8750
    Epoch 295/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1089241.7500 - val_loss: 1092628.3750
    Epoch 296/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1089790.2500 - val_loss: 1087574.2500
    Epoch 297/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1089013.2500 - val_loss: 1088314.6250
    Epoch 298/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1086745.5000 - val_loss: 1087265.3750
    Epoch 299/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1086644.0000 - val_loss: 1086087.1250
    Epoch 300/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1084804.7500 - val_loss: 1084852.3750
    Epoch 301/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1085743.6250 - val_loss: 1086499.5000
    Epoch 302/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1084274.2500 - val_loss: 1085258.6250
    Epoch 303/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1083491.3750 - val_loss: 1082049.7500
    Epoch 304/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1081603.8750 - val_loss: 1081247.3750
    Epoch 305/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1082794.0000 - val_loss: 1079487.2500
    Epoch 306/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1081574.0000 - val_loss: 1078374.5000
    Epoch 307/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1080483.5000 - val_loss: 1079734.8750
    Epoch 308/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1079474.8750 - val_loss: 1080593.2500
    Epoch 309/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1078592.2500 - val_loss: 1076550.1250
    Epoch 310/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1078882.0000 - val_loss: 1076955.7500
    Epoch 311/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1076117.3750 - val_loss: 1075748.0000
    Epoch 312/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1075668.7500 - val_loss: 1075319.8750
    Epoch 313/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1076170.8750 - val_loss: 1072675.5000
    Epoch 314/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1075279.1250 - val_loss: 1072931.5000
    Epoch 315/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1072394.1250 - val_loss: 1073367.6250
    Epoch 316/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1072131.8750 - val_loss: 1073844.3750
    Epoch 317/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1072598.3750 - val_loss: 1069866.5000
    Epoch 318/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1070581.6250 - val_loss: 1071939.2500
    Epoch 319/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1072298.2500 - val_loss: 1067770.5000
    Epoch 320/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1071914.0000 - val_loss: 1068771.2500
    Epoch 321/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1069909.0000 - val_loss: 1070592.5000
    Epoch 322/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1069940.1250 - val_loss: 1066716.0000
    Epoch 323/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1068936.7500 - val_loss: 1068910.2500
    Epoch 324/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1067144.0000 - val_loss: 1066099.6250
    Epoch 325/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1066298.0000 - val_loss: 1066079.7500
    Epoch 326/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1067420.7500 - val_loss: 1066412.8750
    Epoch 327/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1064341.0000 - val_loss: 1065653.8750
    Epoch 328/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1065334.2500 - val_loss: 1061793.0000
    Epoch 329/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1065631.1250 - val_loss: 1062080.7500
    Epoch 330/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1062826.2500 - val_loss: 1062737.0000
    Epoch 331/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1061753.3750 - val_loss: 1061408.0000
    Epoch 332/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1061658.1250 - val_loss: 1059262.0000
    Epoch 333/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1061820.6250 - val_loss: 1059387.6250
    Epoch 334/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1061065.0000 - val_loss: 1058993.8750
    Epoch 335/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1058761.6250 - val_loss: 1058761.6250
    Epoch 336/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1059294.6250 - val_loss: 1056180.5000
    Epoch 337/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1059994.2500 - val_loss: 1055779.3750
    Epoch 338/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1057579.6250 - val_loss: 1055022.1250
    Epoch 339/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1054963.3750 - val_loss: 1056517.5000
    Epoch 340/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1057136.7500 - val_loss: 1055689.6250
    Epoch 341/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1055044.2500 - val_loss: 1053442.6250
    Epoch 342/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1054892.7500 - val_loss: 1056350.1250
    Epoch 343/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1053039.2500 - val_loss: 1052073.8750
    Epoch 344/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1053178.1250 - val_loss: 1050397.5000
    Epoch 345/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1051030.0000 - val_loss: 1049792.6250
    Epoch 346/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1049371.1250 - val_loss: 1051432.5000
    Epoch 347/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1051268.6250 - val_loss: 1047872.1875
    Epoch 348/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1049559.0000 - val_loss: 1050881.8750
    Epoch 349/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1050179.5000 - val_loss: 1047195.7500
    Epoch 350/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1049458.0000 - val_loss: 1048518.5000
    Epoch 351/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1047936.1250 - val_loss: 1046711.8125
    Epoch 352/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1046985.2500 - val_loss: 1046685.4375
    Epoch 353/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1046410.8125 - val_loss: 1044371.5000
    Epoch 354/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1046469.7500 - val_loss: 1045020.2500
    Epoch 355/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1045099.3750 - val_loss: 1042251.8125
    Epoch 356/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1044682.8750 - val_loss: 1042510.0000
    Epoch 357/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1045281.4375 - val_loss: 1043753.6875
    Epoch 358/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1042314.6250 - val_loss: 1044305.8125
    Epoch 359/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1042226.6875 - val_loss: 1044598.0000
    Epoch 360/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1042008.5625 - val_loss: 1040951.0625
    Epoch 361/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1041406.6250 - val_loss: 1039393.0625
    Epoch 362/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1041594.6250 - val_loss: 1040318.1875
    Epoch 363/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1040265.6250 - val_loss: 1040703.8125
    Epoch 364/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1039496.5625 - val_loss: 1038190.0625
    Epoch 365/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1038521.7500 - val_loss: 1037668.1875
    Epoch 366/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1039135.5625 - val_loss: 1038481.5000
    Epoch 367/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1036018.8750 - val_loss: 1036667.3125
    Epoch 368/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1036538.1250 - val_loss: 1036857.2500
    Epoch 369/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1034896.5000 - val_loss: 1035706.8750
    Epoch 370/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1033821.5625 - val_loss: 1031021.4375
    Epoch 371/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1035518.3750 - val_loss: 1033537.6250
    Epoch 372/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1034943.8125 - val_loss: 1034926.5000
    Epoch 373/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1032371.3125 - val_loss: 1032719.8125
    Epoch 374/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1032135.6250 - val_loss: 1034877.8750
    Epoch 375/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1031512.9375 - val_loss: 1031125.5625
    Epoch 376/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1030215.1875 - val_loss: 1031227.5000
    Epoch 377/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1031073.7500 - val_loss: 1028583.5625
    Epoch 378/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1029706.9375 - val_loss: 1029750.5000
    Epoch 379/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1028833.2500 - val_loss: 1029355.8125
    Epoch 380/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1027803.3125 - val_loss: 1026444.2500
    Epoch 381/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1027977.6875 - val_loss: 1025671.8125
    Epoch 382/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1028269.0000 - val_loss: 1024053.5625
    Epoch 383/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1027385.6250 - val_loss: 1025126.3750
    Epoch 384/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1024916.6875 - val_loss: 1024979.0625
    Epoch 385/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1024698.5000 - val_loss: 1025554.6875
    Epoch 386/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1024274.6875 - val_loss: 1024169.5000
    Epoch 387/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1021434.9375 - val_loss: 1024975.8125
    Epoch 388/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1022012.3750 - val_loss: 1024355.4375
    Epoch 389/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1021878.5000 - val_loss: 1021261.6250
    Epoch 390/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1022437.8750 - val_loss: 1024223.3750
    Epoch 391/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1020683.7500 - val_loss: 1018945.1875
    Epoch 392/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1018942.3750 - val_loss: 1020317.2500
    Epoch 393/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1020340.1250 - val_loss: 1021060.0625
    Epoch 394/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1019398.1875 - val_loss: 1016045.2500
    Epoch 395/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1018981.3125 - val_loss: 1017139.5000
    Epoch 396/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1019122.7500 - val_loss: 1018061.3125
    Epoch 397/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1018977.9375 - val_loss: 1016313.5000
    Epoch 398/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1016833.8750 - val_loss: 1015683.1875
    Epoch 399/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1016023.8125 - val_loss: 1017451.6250
    Epoch 400/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1016948.1250 - val_loss: 1014489.1875
    Epoch 401/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1014112.5000 - val_loss: 1014841.0625
    Epoch 402/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1014189.6250 - val_loss: 1015581.5625
    Epoch 403/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1012022.5625 - val_loss: 1012297.3750
    Epoch 404/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1013840.8125 - val_loss: 1010308.2500
    Epoch 405/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1011558.0000 - val_loss: 1012011.5000
    Epoch 406/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1011956.3125 - val_loss: 1011028.3750
    Epoch 407/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1011663.4375 - val_loss: 1011655.0625
    Epoch 408/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1008736.5625 - val_loss: 1012076.3750
    Epoch 409/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1008054.3125 - val_loss: 1010083.6250
    Epoch 410/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1008932.4375 - val_loss: 1010355.6250
    Epoch 411/2000
    8/8 [==============================] - 0s 5ms/step - loss: 1007139.6250 - val_loss: 1010380.1875
    Epoch 412/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1010164.2500 - val_loss: 1008812.3750
    Epoch 413/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1009255.0000 - val_loss: 1004601.6250
    Epoch 414/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1005774.9375 - val_loss: 1004941.1250
    Epoch 415/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1005735.8750 - val_loss: 1002856.6250
    Epoch 416/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1006711.8125 - val_loss: 1002500.5625
    Epoch 417/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1004674.7500 - val_loss: 1004678.8125
    Epoch 418/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1004486.8125 - val_loss: 1004722.4375
    Epoch 419/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1003039.5625 - val_loss: 1005158.6250
    Epoch 420/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1003057.5000 - val_loss: 1002628.5000
    Epoch 421/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1003280.1250 - val_loss: 1003717.7500
    Epoch 422/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1002497.5000 - val_loss: 1001141.6250
    Epoch 423/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1000887.3125 - val_loss: 1001110.9375
    Epoch 424/2000
    8/8 [==============================] - 0s 4ms/step - loss: 998684.7500 - val_loss: 1000656.5625
    Epoch 425/2000
    8/8 [==============================] - 0s 4ms/step - loss: 1000385.6250 - val_loss: 999924.5625
    Epoch 426/2000
    8/8 [==============================] - 0s 4ms/step - loss: 998892.3750 - val_loss: 1000518.0625
    Epoch 427/2000
    8/8 [==============================] - 0s 4ms/step - loss: 998555.0625 - val_loss: 998800.7500
    Epoch 428/2000
    8/8 [==============================] - 0s 4ms/step - loss: 997271.8750 - val_loss: 999912.7500
    Epoch 429/2000


    8/8 [==============================] - 0s 4ms/step - loss: 996702.0625 - val_loss: 998554.8750
    Epoch 430/2000
    8/8 [==============================] - 0s 4ms/step - loss: 997825.4375 - val_loss: 998688.6250
    Epoch 431/2000
    8/8 [==============================] - 0s 4ms/step - loss: 997623.6875 - val_loss: 998320.1875
    Epoch 432/2000
    8/8 [==============================] - 0s 4ms/step - loss: 994709.8750 - val_loss: 993243.8125
    Epoch 433/2000
    8/8 [==============================] - 0s 4ms/step - loss: 994788.2500 - val_loss: 993552.4375
    Epoch 434/2000
    8/8 [==============================] - 0s 4ms/step - loss: 993193.2500 - val_loss: 994893.7500
    Epoch 435/2000
    8/8 [==============================] - 0s 4ms/step - loss: 994514.7500 - val_loss: 994087.0625
    Epoch 436/2000
    8/8 [==============================] - 0s 4ms/step - loss: 993330.7500 - val_loss: 993710.3750
    Epoch 437/2000
    8/8 [==============================] - 0s 4ms/step - loss: 993472.1250 - val_loss: 991545.5000
    Epoch 438/2000
    8/8 [==============================] - 0s 4ms/step - loss: 993599.1250 - val_loss: 992611.7500
    Epoch 439/2000
    8/8 [==============================] - 0s 4ms/step - loss: 990914.0000 - val_loss: 990451.0000
    Epoch 440/2000
    8/8 [==============================] - 0s 4ms/step - loss: 990207.3750 - val_loss: 990906.8750
    Epoch 441/2000
    8/8 [==============================] - 0s 4ms/step - loss: 989603.2500 - val_loss: 989560.3125
    Epoch 442/2000
    8/8 [==============================] - 0s 4ms/step - loss: 989619.5625 - val_loss: 988870.7500
    Epoch 443/2000
    8/8 [==============================] - 0s 4ms/step - loss: 989807.3125 - val_loss: 988068.5000
    Epoch 444/2000
    8/8 [==============================] - 0s 4ms/step - loss: 988331.5000 - val_loss: 987855.6875
    Epoch 445/2000
    8/8 [==============================] - 0s 4ms/step - loss: 987766.0625 - val_loss: 986334.7500
    Epoch 446/2000
    8/8 [==============================] - 0s 4ms/step - loss: 986259.2500 - val_loss: 988000.0000
    Epoch 447/2000
    8/8 [==============================] - 0s 4ms/step - loss: 987804.6875 - val_loss: 986479.0625
    Epoch 448/2000
    8/8 [==============================] - 0s 4ms/step - loss: 985807.1875 - val_loss: 985564.2500
    Epoch 449/2000
    8/8 [==============================] - 0s 4ms/step - loss: 985674.5000 - val_loss: 984004.0625
    Epoch 450/2000
    8/8 [==============================] - 0s 4ms/step - loss: 984384.8125 - val_loss: 983590.3125
    Epoch 451/2000
    8/8 [==============================] - 0s 4ms/step - loss: 984361.7500 - val_loss: 984899.5000
    Epoch 452/2000
    8/8 [==============================] - 0s 4ms/step - loss: 983063.3750 - val_loss: 985587.1875
    Epoch 453/2000
    8/8 [==============================] - 0s 4ms/step - loss: 982275.3125 - val_loss: 983821.6250
    Epoch 454/2000
    8/8 [==============================] - 0s 4ms/step - loss: 982187.5625 - val_loss: 980975.1250
    Epoch 455/2000
    8/8 [==============================] - 0s 4ms/step - loss: 982406.8125 - val_loss: 980886.3125
    Epoch 456/2000
    8/8 [==============================] - 0s 4ms/step - loss: 981543.9375 - val_loss: 978347.3125
    Epoch 457/2000
    8/8 [==============================] - 0s 4ms/step - loss: 981979.6875 - val_loss: 981934.5000
    Epoch 458/2000
    8/8 [==============================] - 0s 4ms/step - loss: 979077.3125 - val_loss: 980875.6250
    Epoch 459/2000
    8/8 [==============================] - 0s 4ms/step - loss: 980011.7500 - val_loss: 978925.3125
    Epoch 460/2000
    8/8 [==============================] - 0s 4ms/step - loss: 979378.0625 - val_loss: 981095.2500
    Epoch 461/2000
    8/8 [==============================] - 0s 4ms/step - loss: 978807.0000 - val_loss: 979949.0000
    Epoch 462/2000
    8/8 [==============================] - 0s 4ms/step - loss: 976799.1875 - val_loss: 978517.3125
    Epoch 463/2000
    8/8 [==============================] - 0s 4ms/step - loss: 976767.6250 - val_loss: 978630.5000
    Epoch 464/2000
    8/8 [==============================] - 0s 4ms/step - loss: 976193.0000 - val_loss: 974460.3750
    Epoch 465/2000
    8/8 [==============================] - 0s 4ms/step - loss: 976779.7500 - val_loss: 978371.8125
    Epoch 466/2000
    8/8 [==============================] - 0s 4ms/step - loss: 974973.4375 - val_loss: 977053.7500
    Epoch 467/2000
    8/8 [==============================] - 0s 4ms/step - loss: 975891.3125 - val_loss: 974495.8125
    Epoch 468/2000
    8/8 [==============================] - 0s 4ms/step - loss: 974308.7500 - val_loss: 975000.6250
    Epoch 469/2000
    8/8 [==============================] - 0s 4ms/step - loss: 973028.4375 - val_loss: 971786.7500
    Epoch 470/2000
    8/8 [==============================] - 0s 4ms/step - loss: 973775.8750 - val_loss: 972628.6875
    Epoch 471/2000
    8/8 [==============================] - 0s 4ms/step - loss: 974303.4375 - val_loss: 972769.1875
    Epoch 472/2000
    8/8 [==============================] - 0s 4ms/step - loss: 970521.8750 - val_loss: 973317.1250
    Epoch 473/2000
    8/8 [==============================] - 0s 4ms/step - loss: 971763.5625 - val_loss: 971412.3750
    Epoch 474/2000
    8/8 [==============================] - 0s 4ms/step - loss: 970215.6875 - val_loss: 972163.9375
    Epoch 475/2000
    8/8 [==============================] - 0s 4ms/step - loss: 969943.6875 - val_loss: 972607.1250
    Epoch 476/2000
    8/8 [==============================] - 0s 4ms/step - loss: 970769.8750 - val_loss: 971184.1250
    Epoch 477/2000
    8/8 [==============================] - 0s 4ms/step - loss: 969746.1875 - val_loss: 967615.2500
    Epoch 478/2000
    8/8 [==============================] - 0s 4ms/step - loss: 969965.0625 - val_loss: 967471.2500
    Epoch 479/2000
    8/8 [==============================] - 0s 4ms/step - loss: 968672.0000 - val_loss: 968386.3750
    Epoch 480/2000
    8/8 [==============================] - 0s 4ms/step - loss: 967579.3125 - val_loss: 966051.0000
    Epoch 481/2000
    8/8 [==============================] - 0s 4ms/step - loss: 967380.4375 - val_loss: 967592.0000
    Epoch 482/2000
    8/8 [==============================] - 0s 4ms/step - loss: 965671.6250 - val_loss: 967033.2500
    Epoch 483/2000
    8/8 [==============================] - 0s 4ms/step - loss: 966986.2500 - val_loss: 965470.8125
    Epoch 484/2000
    8/8 [==============================] - 0s 4ms/step - loss: 966256.9375 - val_loss: 965796.5625
    Epoch 485/2000
    8/8 [==============================] - 0s 4ms/step - loss: 966934.1250 - val_loss: 964442.8750
    Epoch 486/2000
    8/8 [==============================] - 0s 4ms/step - loss: 963972.2500 - val_loss: 960440.7500
    Epoch 487/2000
    8/8 [==============================] - 0s 4ms/step - loss: 964193.7500 - val_loss: 961583.8125
    Epoch 488/2000
    8/8 [==============================] - 0s 4ms/step - loss: 963450.3125 - val_loss: 966387.0000
    Epoch 489/2000
    8/8 [==============================] - 0s 4ms/step - loss: 963886.9375 - val_loss: 964933.8750
    Epoch 490/2000
    8/8 [==============================] - 0s 4ms/step - loss: 962063.8750 - val_loss: 961301.5625
    Epoch 491/2000
    8/8 [==============================] - 0s 4ms/step - loss: 961872.8750 - val_loss: 961752.9375
    Epoch 492/2000
    8/8 [==============================] - 0s 4ms/step - loss: 960293.2500 - val_loss: 959600.6250
    Epoch 493/2000
    8/8 [==============================] - 0s 4ms/step - loss: 961999.1250 - val_loss: 962618.3750
    Epoch 494/2000
    8/8 [==============================] - 0s 4ms/step - loss: 960451.1875 - val_loss: 959354.7500
    Epoch 495/2000
    8/8 [==============================] - 0s 4ms/step - loss: 959592.4375 - val_loss: 963245.8750
    Epoch 496/2000
    8/8 [==============================] - 0s 5ms/step - loss: 960859.8750 - val_loss: 960240.9375
    Epoch 497/2000
    8/8 [==============================] - 0s 4ms/step - loss: 959545.6250 - val_loss: 959035.7500
    Epoch 498/2000
    8/8 [==============================] - 0s 4ms/step - loss: 957133.0625 - val_loss: 957400.7500
    Epoch 499/2000
    8/8 [==============================] - 0s 4ms/step - loss: 957002.3750 - val_loss: 956049.3750
    Epoch 500/2000
    8/8 [==============================] - 0s 4ms/step - loss: 956183.7500 - val_loss: 957694.8125
    Epoch 501/2000
    8/8 [==============================] - 0s 4ms/step - loss: 957227.9375 - val_loss: 955089.2500
    Epoch 502/2000
    8/8 [==============================] - 0s 4ms/step - loss: 954650.1250 - val_loss: 956643.9375
    Epoch 503/2000
    8/8 [==============================] - 0s 4ms/step - loss: 955608.3125 - val_loss: 954796.9375
    Epoch 504/2000
    8/8 [==============================] - 0s 4ms/step - loss: 954925.0000 - val_loss: 954594.5625
    Epoch 505/2000
    8/8 [==============================] - 0s 4ms/step - loss: 953341.4375 - val_loss: 954317.8750
    Epoch 506/2000
    8/8 [==============================] - 0s 4ms/step - loss: 954600.1250 - val_loss: 952342.0625
    Epoch 507/2000
    8/8 [==============================] - 0s 4ms/step - loss: 951513.1875 - val_loss: 951169.2500
    Epoch 508/2000
    8/8 [==============================] - 0s 4ms/step - loss: 953344.0000 - val_loss: 953265.5000
    Epoch 509/2000
    8/8 [==============================] - 0s 4ms/step - loss: 951423.1875 - val_loss: 952358.6250
    Epoch 510/2000
    8/8 [==============================] - 0s 4ms/step - loss: 952127.5000 - val_loss: 951092.3750
    Epoch 511/2000
    8/8 [==============================] - 0s 4ms/step - loss: 951643.0625 - val_loss: 950494.6250
    Epoch 512/2000
    8/8 [==============================] - 0s 4ms/step - loss: 951818.1250 - val_loss: 950592.4375
    Epoch 513/2000
    8/8 [==============================] - 0s 4ms/step - loss: 951755.3750 - val_loss: 950262.3750
    Epoch 514/2000
    8/8 [==============================] - 0s 4ms/step - loss: 949382.6250 - val_loss: 951589.2500
    Epoch 515/2000
    8/8 [==============================] - 0s 4ms/step - loss: 948599.1875 - val_loss: 947453.0000
    Epoch 516/2000
    8/8 [==============================] - 0s 4ms/step - loss: 949559.0000 - val_loss: 946017.0625
    Epoch 517/2000
    8/8 [==============================] - 0s 4ms/step - loss: 948624.0625 - val_loss: 947552.3125
    Epoch 518/2000
    8/8 [==============================] - 0s 4ms/step - loss: 949198.1250 - val_loss: 948728.3125
    Epoch 519/2000
    8/8 [==============================] - 0s 4ms/step - loss: 948202.5000 - val_loss: 947665.5000
    Epoch 520/2000
    8/8 [==============================] - 0s 4ms/step - loss: 945041.2500 - val_loss: 945080.7500
    Epoch 521/2000
    8/8 [==============================] - 0s 4ms/step - loss: 946061.7500 - val_loss: 944616.4375
    Epoch 522/2000
    8/8 [==============================] - 0s 4ms/step - loss: 945323.0000 - val_loss: 944408.9375
    Epoch 523/2000
    8/8 [==============================] - 0s 4ms/step - loss: 944191.3125 - val_loss: 943372.9375
    Epoch 524/2000
    8/8 [==============================] - 0s 4ms/step - loss: 944466.1875 - val_loss: 945782.7500
    Epoch 525/2000
    8/8 [==============================] - 0s 4ms/step - loss: 945086.0625 - val_loss: 946517.2500
    Epoch 526/2000
    8/8 [==============================] - 0s 4ms/step - loss: 943816.8125 - val_loss: 942656.0000
    Epoch 527/2000
    8/8 [==============================] - 0s 4ms/step - loss: 943321.4375 - val_loss: 941163.7500
    Epoch 528/2000
    8/8 [==============================] - 0s 4ms/step - loss: 942916.8125 - val_loss: 944374.3750
    Epoch 529/2000
    8/8 [==============================] - 0s 4ms/step - loss: 941538.5625 - val_loss: 943402.6875
    Epoch 530/2000
    8/8 [==============================] - 0s 4ms/step - loss: 942020.6250 - val_loss: 941508.8125
    Epoch 531/2000
    8/8 [==============================] - 0s 4ms/step - loss: 940452.2500 - val_loss: 940482.3750
    Epoch 532/2000
    8/8 [==============================] - 0s 4ms/step - loss: 939873.0000 - val_loss: 941799.2500
    Epoch 533/2000
    8/8 [==============================] - 0s 4ms/step - loss: 938885.2500 - val_loss: 938814.0625
    Epoch 534/2000
    8/8 [==============================] - 0s 4ms/step - loss: 941039.4375 - val_loss: 939817.2500
    Epoch 535/2000
    8/8 [==============================] - 0s 4ms/step - loss: 938465.0625 - val_loss: 937448.0000
    Epoch 536/2000
    8/8 [==============================] - 0s 4ms/step - loss: 938379.0000 - val_loss: 936004.5000
    Epoch 537/2000
    8/8 [==============================] - 0s 4ms/step - loss: 936666.6250 - val_loss: 935392.7500
    Epoch 538/2000
    8/8 [==============================] - 0s 4ms/step - loss: 935017.6250 - val_loss: 938566.3125
    Epoch 539/2000
    8/8 [==============================] - 0s 4ms/step - loss: 935242.3750 - val_loss: 935696.4375
    Epoch 540/2000
    8/8 [==============================] - 0s 4ms/step - loss: 936308.8125 - val_loss: 935229.5625
    Epoch 541/2000
    8/8 [==============================] - 0s 4ms/step - loss: 934542.3125 - val_loss: 934228.6875
    Epoch 542/2000
    8/8 [==============================] - 0s 4ms/step - loss: 935402.8750 - val_loss: 933687.1250
    Epoch 543/2000
    8/8 [==============================] - 0s 4ms/step - loss: 935092.9375 - val_loss: 935548.3750
    Epoch 544/2000
    8/8 [==============================] - 0s 4ms/step - loss: 935740.9375 - val_loss: 932580.0625
    Epoch 545/2000
    8/8 [==============================] - 0s 4ms/step - loss: 934518.3750 - val_loss: 933135.2500
    Epoch 546/2000
    8/8 [==============================] - 0s 4ms/step - loss: 933067.5625 - val_loss: 931654.0625
    Epoch 547/2000
    8/8 [==============================] - 0s 4ms/step - loss: 931013.0625 - val_loss: 930386.4375
    Epoch 548/2000
    8/8 [==============================] - 0s 4ms/step - loss: 932097.4375 - val_loss: 935322.1250
    Epoch 549/2000
    8/8 [==============================] - 0s 4ms/step - loss: 931621.1250 - val_loss: 930790.3125
    Epoch 550/2000
    8/8 [==============================] - 0s 4ms/step - loss: 932239.1250 - val_loss: 930670.3125
    Epoch 551/2000
    8/8 [==============================] - 0s 4ms/step - loss: 931551.9375 - val_loss: 928016.0000
    Epoch 552/2000
    8/8 [==============================] - 0s 4ms/step - loss: 929796.9375 - val_loss: 931996.5625
    Epoch 553/2000
    8/8 [==============================] - 0s 4ms/step - loss: 929744.0000 - val_loss: 929887.4375
    Epoch 554/2000
    8/8 [==============================] - 0s 4ms/step - loss: 928725.5000 - val_loss: 930090.7500
    Epoch 555/2000
    8/8 [==============================] - 0s 4ms/step - loss: 928672.3750 - val_loss: 929173.1250
    Epoch 556/2000
    8/8 [==============================] - 0s 4ms/step - loss: 928690.7500 - val_loss: 927393.6875
    Epoch 557/2000
    8/8 [==============================] - 0s 4ms/step - loss: 928775.2500 - val_loss: 923387.5000
    Epoch 558/2000
    8/8 [==============================] - 0s 4ms/step - loss: 928597.5000 - val_loss: 926391.2500
    Epoch 559/2000
    8/8 [==============================] - 0s 4ms/step - loss: 925744.1250 - val_loss: 924040.1250
    Epoch 560/2000
    8/8 [==============================] - 0s 4ms/step - loss: 925600.6250 - val_loss: 927840.6250
    Epoch 561/2000
    8/8 [==============================] - 0s 4ms/step - loss: 924915.5000 - val_loss: 922560.9375
    Epoch 562/2000
    8/8 [==============================] - 0s 4ms/step - loss: 924354.9375 - val_loss: 925905.2500
    Epoch 563/2000
    8/8 [==============================] - 0s 4ms/step - loss: 925006.0000 - val_loss: 921252.9375
    Epoch 564/2000
    8/8 [==============================] - 0s 4ms/step - loss: 922916.5000 - val_loss: 923295.4375
    Epoch 565/2000
    8/8 [==============================] - 0s 4ms/step - loss: 924395.8125 - val_loss: 921972.9375
    Epoch 566/2000
    8/8 [==============================] - 0s 4ms/step - loss: 923094.1250 - val_loss: 922145.2500
    Epoch 567/2000
    8/8 [==============================] - 0s 4ms/step - loss: 923471.9375 - val_loss: 922482.0000
    Epoch 568/2000
    8/8 [==============================] - 0s 4ms/step - loss: 923356.3750 - val_loss: 922494.9375
    Epoch 569/2000
    8/8 [==============================] - 0s 4ms/step - loss: 921823.3125 - val_loss: 923091.1875
    Epoch 570/2000
    8/8 [==============================] - 0s 4ms/step - loss: 922190.8750 - val_loss: 920087.5625
    Epoch 571/2000
    8/8 [==============================] - 0s 4ms/step - loss: 920876.4375 - val_loss: 919850.7500
    Epoch 572/2000
    8/8 [==============================] - 0s 4ms/step - loss: 920749.8125 - val_loss: 920321.9375
    Epoch 573/2000
    8/8 [==============================] - 0s 4ms/step - loss: 918042.3750 - val_loss: 920919.5625
    Epoch 574/2000
    8/8 [==============================] - 0s 4ms/step - loss: 918966.0625 - val_loss: 918334.1875
    Epoch 575/2000
    8/8 [==============================] - 0s 4ms/step - loss: 919683.3125 - val_loss: 922049.8125
    Epoch 576/2000
    8/8 [==============================] - 0s 4ms/step - loss: 919067.0625 - val_loss: 917806.3125
    Epoch 577/2000


    8/8 [==============================] - 0s 4ms/step - loss: 918858.8750 - val_loss: 916156.0625
    Epoch 578/2000
    8/8 [==============================] - 0s 4ms/step - loss: 917322.3750 - val_loss: 919258.0000
    Epoch 579/2000
    8/8 [==============================] - 0s 4ms/step - loss: 917082.9375 - val_loss: 915428.9375
    Epoch 580/2000
    8/8 [==============================] - 0s 4ms/step - loss: 915553.0000 - val_loss: 914668.9375
    Epoch 581/2000
    8/8 [==============================] - 0s 4ms/step - loss: 914881.2500 - val_loss: 917174.5000
    Epoch 582/2000
    8/8 [==============================] - 0s 4ms/step - loss: 915740.8750 - val_loss: 917185.9375
    Epoch 583/2000
    8/8 [==============================] - 0s 4ms/step - loss: 915946.1250 - val_loss: 914638.3125
    Epoch 584/2000
    8/8 [==============================] - 0s 4ms/step - loss: 915846.5000 - val_loss: 913551.0625
    Epoch 585/2000
    8/8 [==============================] - 0s 4ms/step - loss: 913335.6875 - val_loss: 912751.2500
    Epoch 586/2000
    8/8 [==============================] - 0s 4ms/step - loss: 913195.5000 - val_loss: 915667.0000
    Epoch 587/2000
    8/8 [==============================] - 0s 4ms/step - loss: 912689.0625 - val_loss: 911266.4375
    Epoch 588/2000
    8/8 [==============================] - 0s 4ms/step - loss: 912160.3750 - val_loss: 912707.6250
    Epoch 589/2000
    8/8 [==============================] - 0s 4ms/step - loss: 911324.8750 - val_loss: 911413.1250
    Epoch 590/2000
    8/8 [==============================] - 0s 4ms/step - loss: 911424.3125 - val_loss: 911502.0000
    Epoch 591/2000
    8/8 [==============================] - 0s 4ms/step - loss: 910190.9375 - val_loss: 910505.2500
    Epoch 592/2000
    8/8 [==============================] - 0s 4ms/step - loss: 911084.1875 - val_loss: 910364.2500
    Epoch 593/2000
    8/8 [==============================] - 0s 4ms/step - loss: 909858.9375 - val_loss: 908783.6875
    Epoch 594/2000
    8/8 [==============================] - 0s 4ms/step - loss: 909578.9375 - val_loss: 908734.5000
    Epoch 595/2000
    8/8 [==============================] - 0s 4ms/step - loss: 908249.0000 - val_loss: 908379.9375
    Epoch 596/2000
    8/8 [==============================] - 0s 4ms/step - loss: 907909.0625 - val_loss: 909912.7500
    Epoch 597/2000
    8/8 [==============================] - 0s 4ms/step - loss: 907129.3750 - val_loss: 909516.5625
    Epoch 598/2000
    8/8 [==============================] - 0s 4ms/step - loss: 908488.7500 - val_loss: 908176.3125
    Epoch 599/2000
    8/8 [==============================] - 0s 4ms/step - loss: 908615.5625 - val_loss: 906940.2500
    Epoch 600/2000
    8/8 [==============================] - 0s 4ms/step - loss: 906552.5625 - val_loss: 905982.9375
    Epoch 601/2000
    8/8 [==============================] - 0s 4ms/step - loss: 907250.5000 - val_loss: 907133.0000
    Epoch 602/2000
    8/8 [==============================] - 0s 5ms/step - loss: 906766.9375 - val_loss: 906544.8750
    Epoch 603/2000
    8/8 [==============================] - 0s 4ms/step - loss: 906065.0000 - val_loss: 903824.1875
    Epoch 604/2000
    8/8 [==============================] - 0s 4ms/step - loss: 905602.6875 - val_loss: 903722.8750
    Epoch 605/2000
    8/8 [==============================] - 0s 4ms/step - loss: 905130.6875 - val_loss: 904153.5000
    Epoch 606/2000
    8/8 [==============================] - 0s 4ms/step - loss: 904819.6875 - val_loss: 902465.2500
    Epoch 607/2000
    8/8 [==============================] - 0s 4ms/step - loss: 904196.4375 - val_loss: 905098.0000
    Epoch 608/2000
    8/8 [==============================] - 0s 4ms/step - loss: 903500.8750 - val_loss: 904442.8750
    Epoch 609/2000
    8/8 [==============================] - 0s 4ms/step - loss: 901668.5000 - val_loss: 902413.8750
    Epoch 610/2000
    8/8 [==============================] - 0s 4ms/step - loss: 901456.9375 - val_loss: 903054.0625
    Epoch 611/2000
    8/8 [==============================] - 0s 4ms/step - loss: 902443.2500 - val_loss: 901109.5625
    Epoch 612/2000
    8/8 [==============================] - 0s 4ms/step - loss: 902952.1875 - val_loss: 902519.0625
    Epoch 613/2000
    8/8 [==============================] - 0s 4ms/step - loss: 900102.7500 - val_loss: 899633.2500
    Epoch 614/2000
    8/8 [==============================] - 0s 5ms/step - loss: 900555.0625 - val_loss: 899798.3750
    Epoch 615/2000
    8/8 [==============================] - 0s 4ms/step - loss: 899826.7500 - val_loss: 900304.7500
    Epoch 616/2000
    8/8 [==============================] - 0s 4ms/step - loss: 899269.0625 - val_loss: 898987.4375
    Epoch 617/2000
    8/8 [==============================] - 0s 4ms/step - loss: 898007.5625 - val_loss: 896926.7500
    Epoch 618/2000
    8/8 [==============================] - 0s 4ms/step - loss: 899312.0625 - val_loss: 899449.9375
    Epoch 619/2000
    8/8 [==============================] - 0s 4ms/step - loss: 897827.4375 - val_loss: 897989.1250
    Epoch 620/2000
    8/8 [==============================] - 0s 4ms/step - loss: 898157.5625 - val_loss: 896217.3750
    Epoch 621/2000
    8/8 [==============================] - 0s 4ms/step - loss: 898504.8125 - val_loss: 900782.0000
    Epoch 622/2000
    8/8 [==============================] - 0s 4ms/step - loss: 895585.5625 - val_loss: 896297.9375
    Epoch 623/2000
    8/8 [==============================] - 0s 4ms/step - loss: 895316.7500 - val_loss: 896483.7500
    Epoch 624/2000
    8/8 [==============================] - 0s 4ms/step - loss: 896642.4375 - val_loss: 895875.5000
    Epoch 625/2000
    8/8 [==============================] - 0s 4ms/step - loss: 894817.5625 - val_loss: 891362.0000
    Epoch 626/2000
    8/8 [==============================] - 0s 4ms/step - loss: 893804.7500 - val_loss: 897604.8125
    Epoch 627/2000
    8/8 [==============================] - 0s 4ms/step - loss: 896661.5625 - val_loss: 895432.0000
    Epoch 628/2000
    8/8 [==============================] - 0s 4ms/step - loss: 893963.0625 - val_loss: 894584.0000
    Epoch 629/2000
    8/8 [==============================] - 0s 4ms/step - loss: 894670.0625 - val_loss: 891823.5625
    Epoch 630/2000
    8/8 [==============================] - 0s 4ms/step - loss: 894657.5625 - val_loss: 892660.8125
    Epoch 631/2000
    8/8 [==============================] - 0s 4ms/step - loss: 893177.4375 - val_loss: 895860.8125
    Epoch 632/2000
    8/8 [==============================] - 0s 4ms/step - loss: 892384.6250 - val_loss: 890861.8125
    Epoch 633/2000
    8/8 [==============================] - 0s 4ms/step - loss: 891691.8125 - val_loss: 893401.2500
    Epoch 634/2000
    8/8 [==============================] - 0s 4ms/step - loss: 891408.1875 - val_loss: 891231.0625
    Epoch 635/2000
    8/8 [==============================] - 0s 4ms/step - loss: 890930.6875 - val_loss: 888581.1250
    Epoch 636/2000
    8/8 [==============================] - 0s 4ms/step - loss: 890613.0000 - val_loss: 889552.1250
    Epoch 637/2000
    8/8 [==============================] - 0s 4ms/step - loss: 889687.1875 - val_loss: 891441.9375
    Epoch 638/2000
    8/8 [==============================] - 0s 4ms/step - loss: 889489.6875 - val_loss: 889396.6875
    Epoch 639/2000
    8/8 [==============================] - 0s 4ms/step - loss: 890759.6875 - val_loss: 889986.8750
    Epoch 640/2000
    8/8 [==============================] - 0s 4ms/step - loss: 889505.7500 - val_loss: 886863.5000
    Epoch 641/2000
    8/8 [==============================] - 0s 4ms/step - loss: 888708.6250 - val_loss: 886437.3750
    Epoch 642/2000
    8/8 [==============================] - 0s 4ms/step - loss: 887650.6250 - val_loss: 891086.2500
    Epoch 643/2000
    8/8 [==============================] - 0s 4ms/step - loss: 889053.9375 - val_loss: 888794.1250
    Epoch 644/2000
    8/8 [==============================] - 0s 4ms/step - loss: 887342.3125 - val_loss: 885174.1250
    Epoch 645/2000
    8/8 [==============================] - 0s 4ms/step - loss: 886414.2500 - val_loss: 887908.3125
    Epoch 646/2000
    8/8 [==============================] - 0s 4ms/step - loss: 886834.0000 - val_loss: 888111.8750
    Epoch 647/2000
    8/8 [==============================] - 0s 4ms/step - loss: 885700.0000 - val_loss: 884071.8125
    Epoch 648/2000
    8/8 [==============================] - 0s 4ms/step - loss: 887148.5625 - val_loss: 886288.1250
    Epoch 649/2000
    8/8 [==============================] - 0s 4ms/step - loss: 886397.3125 - val_loss: 884995.5625
    Epoch 650/2000
    8/8 [==============================] - 0s 4ms/step - loss: 885046.6250 - val_loss: 884335.0625
    Epoch 651/2000
    8/8 [==============================] - 0s 4ms/step - loss: 884340.4375 - val_loss: 883510.3750
    Epoch 652/2000
    8/8 [==============================] - 0s 4ms/step - loss: 882685.1250 - val_loss: 882929.2500
    Epoch 653/2000
    8/8 [==============================] - 0s 4ms/step - loss: 882601.4375 - val_loss: 882243.5625
    Epoch 654/2000
    8/8 [==============================] - 0s 4ms/step - loss: 883435.3750 - val_loss: 883007.0625
    Epoch 655/2000
    8/8 [==============================] - 0s 4ms/step - loss: 883566.1250 - val_loss: 883092.3750
    Epoch 656/2000
    8/8 [==============================] - 0s 4ms/step - loss: 883352.7500 - val_loss: 882948.6875
    Epoch 657/2000
    8/8 [==============================] - 0s 4ms/step - loss: 882475.1875 - val_loss: 882521.6250
    Epoch 658/2000
    8/8 [==============================] - 0s 4ms/step - loss: 880573.0000 - val_loss: 881030.6875
    Epoch 659/2000
    8/8 [==============================] - 0s 4ms/step - loss: 880188.0625 - val_loss: 878590.7500
    Epoch 660/2000
    8/8 [==============================] - 0s 4ms/step - loss: 879913.9375 - val_loss: 880233.5625
    Epoch 661/2000
    8/8 [==============================] - 0s 4ms/step - loss: 879286.2500 - val_loss: 880584.0625
    Epoch 662/2000
    8/8 [==============================] - 0s 4ms/step - loss: 880061.3750 - val_loss: 874962.3750
    Epoch 663/2000
    8/8 [==============================] - 0s 4ms/step - loss: 880147.4375 - val_loss: 878351.3125
    Epoch 664/2000
    8/8 [==============================] - 0s 4ms/step - loss: 879261.5000 - val_loss: 879470.5625
    Epoch 665/2000
    8/8 [==============================] - 0s 4ms/step - loss: 877994.6250 - val_loss: 879172.6875
    Epoch 666/2000
    8/8 [==============================] - 0s 4ms/step - loss: 877511.1250 - val_loss: 876642.6250
    Epoch 667/2000
    8/8 [==============================] - 0s 4ms/step - loss: 876180.1250 - val_loss: 875095.4375
    Epoch 668/2000
    8/8 [==============================] - 0s 4ms/step - loss: 876466.0625 - val_loss: 875570.2500
    Epoch 669/2000
    8/8 [==============================] - 0s 4ms/step - loss: 876470.6875 - val_loss: 874487.0625
    Epoch 670/2000
    8/8 [==============================] - 0s 4ms/step - loss: 876319.3750 - val_loss: 876606.3125
    Epoch 671/2000
    8/8 [==============================] - 0s 4ms/step - loss: 877056.0625 - val_loss: 875725.0625
    Epoch 672/2000
    8/8 [==============================] - 0s 4ms/step - loss: 875929.1875 - val_loss: 872595.6875
    Epoch 673/2000
    8/8 [==============================] - 0s 4ms/step - loss: 873396.6250 - val_loss: 874545.6875
    Epoch 674/2000
    8/8 [==============================] - 0s 4ms/step - loss: 874135.8750 - val_loss: 874372.0625
    Epoch 675/2000
    8/8 [==============================] - 0s 4ms/step - loss: 875218.0625 - val_loss: 872262.9375
    Epoch 676/2000
    8/8 [==============================] - 0s 4ms/step - loss: 873785.3750 - val_loss: 872236.1250
    Epoch 677/2000
    8/8 [==============================] - 0s 4ms/step - loss: 871754.3125 - val_loss: 876074.0625
    Epoch 678/2000
    8/8 [==============================] - 0s 4ms/step - loss: 872465.4375 - val_loss: 870802.7500
    Epoch 679/2000
    8/8 [==============================] - 0s 4ms/step - loss: 874156.7500 - val_loss: 870696.6875
    Epoch 680/2000
    8/8 [==============================] - 0s 4ms/step - loss: 870477.0000 - val_loss: 870841.0000
    Epoch 681/2000
    8/8 [==============================] - 0s 4ms/step - loss: 872080.1875 - val_loss: 870195.6250
    Epoch 682/2000
    8/8 [==============================] - 0s 4ms/step - loss: 872849.0625 - val_loss: 871117.2500
    Epoch 683/2000
    8/8 [==============================] - 0s 4ms/step - loss: 870188.1875 - val_loss: 870119.2500
    Epoch 684/2000
    8/8 [==============================] - 0s 4ms/step - loss: 869639.3750 - val_loss: 868822.0625
    Epoch 685/2000
    8/8 [==============================] - 0s 4ms/step - loss: 870941.8750 - val_loss: 869424.1875
    Epoch 686/2000
    8/8 [==============================] - 0s 4ms/step - loss: 869538.0625 - val_loss: 867850.0625
    Epoch 687/2000
    8/8 [==============================] - 0s 4ms/step - loss: 868962.6875 - val_loss: 868510.0625
    Epoch 688/2000
    8/8 [==============================] - 0s 4ms/step - loss: 868572.6875 - val_loss: 868869.0625
    Epoch 689/2000
    8/8 [==============================] - 0s 4ms/step - loss: 868684.0625 - val_loss: 866713.7500
    Epoch 690/2000
    8/8 [==============================] - 0s 4ms/step - loss: 867185.1875 - val_loss: 870001.0625
    Epoch 691/2000
    8/8 [==============================] - 0s 4ms/step - loss: 867181.5000 - val_loss: 867335.2500
    Epoch 692/2000
    8/8 [==============================] - 0s 4ms/step - loss: 867756.8125 - val_loss: 867214.8125
    Epoch 693/2000
    8/8 [==============================] - 0s 4ms/step - loss: 864369.7500 - val_loss: 865545.0625
    Epoch 694/2000
    8/8 [==============================] - 0s 4ms/step - loss: 865806.3125 - val_loss: 866430.6250
    Epoch 695/2000
    8/8 [==============================] - 0s 4ms/step - loss: 866859.9375 - val_loss: 867360.1875
    Epoch 696/2000
    8/8 [==============================] - 0s 4ms/step - loss: 864633.3750 - val_loss: 864851.0000
    Epoch 697/2000
    8/8 [==============================] - 0s 4ms/step - loss: 864924.8750 - val_loss: 862685.1250
    Epoch 698/2000
    8/8 [==============================] - 0s 4ms/step - loss: 864572.0000 - val_loss: 865184.1250
    Epoch 699/2000
    8/8 [==============================] - 0s 4ms/step - loss: 863868.4375 - val_loss: 865814.3125
    Epoch 700/2000
    8/8 [==============================] - 0s 4ms/step - loss: 863378.3750 - val_loss: 865126.2500
    Epoch 701/2000
    8/8 [==============================] - 0s 4ms/step - loss: 863511.3125 - val_loss: 859685.3125
    Epoch 702/2000
    8/8 [==============================] - 0s 4ms/step - loss: 864660.5625 - val_loss: 862045.9375
    Epoch 703/2000
    8/8 [==============================] - 0s 4ms/step - loss: 864153.6875 - val_loss: 863313.2500
    Epoch 704/2000
    8/8 [==============================] - 0s 5ms/step - loss: 861138.7500 - val_loss: 861157.0000
    Epoch 705/2000
    8/8 [==============================] - 0s 4ms/step - loss: 861053.3125 - val_loss: 861354.6875
    Epoch 706/2000
    8/8 [==============================] - 0s 4ms/step - loss: 861801.0000 - val_loss: 859561.0625
    Epoch 707/2000
    8/8 [==============================] - 0s 4ms/step - loss: 860555.3125 - val_loss: 862198.1875
    Epoch 708/2000
    8/8 [==============================] - 0s 4ms/step - loss: 859945.5000 - val_loss: 860062.8125
    Epoch 709/2000
    8/8 [==============================] - 0s 4ms/step - loss: 861117.7500 - val_loss: 857888.1250
    Epoch 710/2000
    8/8 [==============================] - 0s 4ms/step - loss: 859671.8125 - val_loss: 859657.4375
    Epoch 711/2000
    8/8 [==============================] - 0s 4ms/step - loss: 858272.9375 - val_loss: 855929.3750
    Epoch 712/2000
    8/8 [==============================] - 0s 4ms/step - loss: 856667.8125 - val_loss: 858914.2500
    Epoch 713/2000
    8/8 [==============================] - 0s 4ms/step - loss: 858313.6250 - val_loss: 859276.0000
    Epoch 714/2000
    8/8 [==============================] - 0s 4ms/step - loss: 857669.6875 - val_loss: 857634.8750
    Epoch 715/2000
    8/8 [==============================] - 0s 4ms/step - loss: 858122.1250 - val_loss: 855974.8125
    Epoch 716/2000
    8/8 [==============================] - 0s 4ms/step - loss: 857633.9375 - val_loss: 855461.5000
    Epoch 717/2000
    8/8 [==============================] - 0s 4ms/step - loss: 856421.7500 - val_loss: 858294.8125
    Epoch 718/2000
    8/8 [==============================] - 0s 4ms/step - loss: 855876.2500 - val_loss: 855106.1875
    Epoch 719/2000
    8/8 [==============================] - 0s 4ms/step - loss: 857405.8750 - val_loss: 856671.1250
    Epoch 720/2000
    8/8 [==============================] - 0s 4ms/step - loss: 854827.0000 - val_loss: 852997.6875
    Epoch 721/2000
    8/8 [==============================] - 0s 4ms/step - loss: 854018.3750 - val_loss: 855004.9375
    Epoch 722/2000
    8/8 [==============================] - 0s 4ms/step - loss: 855220.3125 - val_loss: 855585.5000
    Epoch 723/2000
    8/8 [==============================] - 0s 4ms/step - loss: 854012.6875 - val_loss: 856225.3125
    Epoch 724/2000
    8/8 [==============================] - 0s 4ms/step - loss: 853301.5000 - val_loss: 852340.1875
    Epoch 725/2000


    8/8 [==============================] - 0s 4ms/step - loss: 853174.8125 - val_loss: 852823.1250
    Epoch 726/2000
    8/8 [==============================] - 0s 4ms/step - loss: 852670.2500 - val_loss: 850248.8750
    Epoch 727/2000
    8/8 [==============================] - 0s 4ms/step - loss: 853051.8750 - val_loss: 853064.4375
    Epoch 728/2000
    8/8 [==============================] - 0s 4ms/step - loss: 852791.5625 - val_loss: 851500.5625
    Epoch 729/2000
    8/8 [==============================] - 0s 4ms/step - loss: 851106.5625 - val_loss: 849323.0625
    Epoch 730/2000
    8/8 [==============================] - 0s 4ms/step - loss: 851124.9375 - val_loss: 853640.3125
    Epoch 731/2000
    8/8 [==============================] - 0s 4ms/step - loss: 850333.8750 - val_loss: 849278.7500
    Epoch 732/2000
    8/8 [==============================] - 0s 4ms/step - loss: 851893.5625 - val_loss: 848982.7500
    Epoch 733/2000
    8/8 [==============================] - 0s 4ms/step - loss: 849629.3125 - val_loss: 848858.5625
    Epoch 734/2000
    8/8 [==============================] - 0s 4ms/step - loss: 848494.0000 - val_loss: 849256.2500
    Epoch 735/2000
    8/8 [==============================] - 0s 4ms/step - loss: 848794.6250 - val_loss: 850172.1875
    Epoch 736/2000
    8/8 [==============================] - 0s 4ms/step - loss: 848034.2500 - val_loss: 848797.0000
    Epoch 737/2000
    8/8 [==============================] - 0s 4ms/step - loss: 848705.5625 - val_loss: 849356.2500
    Epoch 738/2000
    8/8 [==============================] - 0s 4ms/step - loss: 848756.4375 - val_loss: 849902.5000
    Epoch 739/2000
    8/8 [==============================] - 0s 4ms/step - loss: 847592.9375 - val_loss: 845928.8750
    Epoch 740/2000
    8/8 [==============================] - 0s 4ms/step - loss: 847199.5625 - val_loss: 848202.3750
    Epoch 741/2000
    8/8 [==============================] - 0s 4ms/step - loss: 845957.3750 - val_loss: 846682.1250
    Epoch 742/2000
    8/8 [==============================] - 0s 4ms/step - loss: 847015.1875 - val_loss: 844829.1875
    Epoch 743/2000
    8/8 [==============================] - 0s 4ms/step - loss: 846868.6250 - val_loss: 844191.8125
    Epoch 744/2000
    8/8 [==============================] - 0s 4ms/step - loss: 846200.8125 - val_loss: 844584.9375
    Epoch 745/2000
    8/8 [==============================] - 0s 4ms/step - loss: 845903.8125 - val_loss: 843815.1250
    Epoch 746/2000
    8/8 [==============================] - 0s 4ms/step - loss: 845608.5625 - val_loss: 845860.5000
    Epoch 747/2000
    8/8 [==============================] - 0s 4ms/step - loss: 843756.2500 - val_loss: 844640.0000
    Epoch 748/2000
    8/8 [==============================] - 0s 4ms/step - loss: 844062.0000 - val_loss: 843816.0625
    Epoch 749/2000
    8/8 [==============================] - 0s 4ms/step - loss: 844309.3750 - val_loss: 844074.5000
    Epoch 750/2000
    8/8 [==============================] - 0s 4ms/step - loss: 843952.0625 - val_loss: 844300.5625
    Epoch 751/2000
    8/8 [==============================] - 0s 4ms/step - loss: 843711.0000 - val_loss: 843499.0000
    Epoch 752/2000
    8/8 [==============================] - 0s 4ms/step - loss: 841370.4375 - val_loss: 841806.6250
    Epoch 753/2000
    8/8 [==============================] - 0s 4ms/step - loss: 841934.0625 - val_loss: 844198.0000
    Epoch 754/2000
    8/8 [==============================] - 0s 4ms/step - loss: 842349.0000 - val_loss: 843003.3125
    Epoch 755/2000
    8/8 [==============================] - 0s 4ms/step - loss: 839955.8125 - val_loss: 841517.5625
    Epoch 756/2000
    8/8 [==============================] - 0s 4ms/step - loss: 842031.3750 - val_loss: 838422.3750
    Epoch 757/2000
    8/8 [==============================] - 0s 4ms/step - loss: 839784.1875 - val_loss: 841266.1875
    Epoch 758/2000
    8/8 [==============================] - 0s 4ms/step - loss: 841571.5000 - val_loss: 840498.8125
    Epoch 759/2000
    8/8 [==============================] - 0s 4ms/step - loss: 839253.0000 - val_loss: 840656.2500
    Epoch 760/2000
    8/8 [==============================] - 0s 4ms/step - loss: 840318.5000 - val_loss: 838167.8125
    Epoch 761/2000
    8/8 [==============================] - 0s 4ms/step - loss: 839976.0625 - val_loss: 839120.1875
    Epoch 762/2000
    8/8 [==============================] - 0s 4ms/step - loss: 838483.7500 - val_loss: 837389.2500
    Epoch 763/2000
    8/8 [==============================] - 0s 4ms/step - loss: 838987.4375 - val_loss: 837954.7500
    Epoch 764/2000
    8/8 [==============================] - 0s 4ms/step - loss: 837722.0000 - val_loss: 837581.3125
    Epoch 765/2000
    8/8 [==============================] - 0s 4ms/step - loss: 838431.0625 - val_loss: 836198.0000
    Epoch 766/2000
    8/8 [==============================] - 0s 4ms/step - loss: 837877.5000 - val_loss: 837791.2500
    Epoch 767/2000
    8/8 [==============================] - 0s 4ms/step - loss: 835561.8125 - val_loss: 832076.7500
    Epoch 768/2000
    8/8 [==============================] - 0s 4ms/step - loss: 837882.7500 - val_loss: 833771.1250
    Epoch 769/2000
    8/8 [==============================] - 0s 4ms/step - loss: 836632.4375 - val_loss: 833573.0000
    Epoch 770/2000
    8/8 [==============================] - 0s 4ms/step - loss: 835167.8750 - val_loss: 833791.1875
    Epoch 771/2000
    8/8 [==============================] - 0s 4ms/step - loss: 835556.5625 - val_loss: 834570.7500
    Epoch 772/2000
    8/8 [==============================] - 0s 4ms/step - loss: 834473.1250 - val_loss: 836520.3750
    Epoch 773/2000
    8/8 [==============================] - 0s 4ms/step - loss: 835325.5625 - val_loss: 833669.8125
    Epoch 774/2000
    8/8 [==============================] - 0s 4ms/step - loss: 833404.3125 - val_loss: 831911.0625
    Epoch 775/2000
    8/8 [==============================] - 0s 4ms/step - loss: 833918.2500 - val_loss: 832098.0000
    Epoch 776/2000
    8/8 [==============================] - 0s 4ms/step - loss: 832980.2500 - val_loss: 831937.5000
    Epoch 777/2000
    8/8 [==============================] - 0s 4ms/step - loss: 833641.6875 - val_loss: 835982.5625
    Epoch 778/2000
    8/8 [==============================] - 0s 4ms/step - loss: 834480.8750 - val_loss: 829159.3125
    Epoch 779/2000
    8/8 [==============================] - 0s 4ms/step - loss: 832472.1250 - val_loss: 832024.7500
    Epoch 780/2000
    8/8 [==============================] - 0s 4ms/step - loss: 831652.6875 - val_loss: 829786.1250
    Epoch 781/2000
    8/8 [==============================] - 0s 4ms/step - loss: 832900.3125 - val_loss: 830817.9375
    Epoch 782/2000
    8/8 [==============================] - 0s 4ms/step - loss: 832971.2500 - val_loss: 830232.7500
    Epoch 783/2000
    8/8 [==============================] - 0s 4ms/step - loss: 831527.0625 - val_loss: 829515.5625
    Epoch 784/2000
    8/8 [==============================] - 0s 4ms/step - loss: 830423.6875 - val_loss: 828360.2500
    Epoch 785/2000
    8/8 [==============================] - 0s 4ms/step - loss: 830176.5625 - val_loss: 828742.6250
    Epoch 786/2000
    8/8 [==============================] - 0s 4ms/step - loss: 831384.1250 - val_loss: 829072.3125
    Epoch 787/2000
    8/8 [==============================] - 0s 4ms/step - loss: 830042.9375 - val_loss: 829104.6250
    Epoch 788/2000
    8/8 [==============================] - 0s 4ms/step - loss: 827989.1250 - val_loss: 829468.1875
    Epoch 789/2000
    8/8 [==============================] - 0s 4ms/step - loss: 828429.1250 - val_loss: 826811.8750
    Epoch 790/2000
    8/8 [==============================] - 0s 4ms/step - loss: 828367.4375 - val_loss: 828477.0625
    Epoch 791/2000
    8/8 [==============================] - 0s 4ms/step - loss: 827209.0000 - val_loss: 830297.8750
    Epoch 792/2000
    8/8 [==============================] - 0s 4ms/step - loss: 828391.0000 - val_loss: 828387.1875
    Epoch 793/2000
    8/8 [==============================] - 0s 4ms/step - loss: 827762.5625 - val_loss: 827723.8750
    Epoch 794/2000
    8/8 [==============================] - 0s 4ms/step - loss: 826520.4375 - val_loss: 827165.1875
    Epoch 795/2000
    8/8 [==============================] - 0s 4ms/step - loss: 825591.3125 - val_loss: 826420.5000
    Epoch 796/2000
    8/8 [==============================] - 0s 4ms/step - loss: 825291.5000 - val_loss: 827047.0625
    Epoch 797/2000
    8/8 [==============================] - 0s 4ms/step - loss: 825333.6250 - val_loss: 824956.9375
    Epoch 798/2000
    8/8 [==============================] - 0s 4ms/step - loss: 825551.1250 - val_loss: 824629.7500
    Epoch 799/2000
    8/8 [==============================] - 0s 4ms/step - loss: 827834.3125 - val_loss: 824879.1250
    Epoch 800/2000
    8/8 [==============================] - 0s 4ms/step - loss: 824870.3125 - val_loss: 824960.8125
    Epoch 801/2000
    8/8 [==============================] - 0s 4ms/step - loss: 825909.5625 - val_loss: 823284.0625
    Epoch 802/2000
    8/8 [==============================] - 0s 4ms/step - loss: 823317.2500 - val_loss: 823901.1875
    Epoch 803/2000
    8/8 [==============================] - 0s 4ms/step - loss: 824075.2500 - val_loss: 821903.5625
    Epoch 804/2000
    8/8 [==============================] - 0s 4ms/step - loss: 822109.4375 - val_loss: 823325.6250
    Epoch 805/2000
    8/8 [==============================] - 0s 4ms/step - loss: 823114.9375 - val_loss: 821016.9375
    Epoch 806/2000
    8/8 [==============================] - 0s 4ms/step - loss: 822906.8750 - val_loss: 825313.9375
    Epoch 807/2000
    8/8 [==============================] - 0s 4ms/step - loss: 821977.5000 - val_loss: 822453.1875
    Epoch 808/2000
    8/8 [==============================] - 0s 4ms/step - loss: 822172.4375 - val_loss: 821911.3125
    Epoch 809/2000
    8/8 [==============================] - 0s 4ms/step - loss: 820569.6250 - val_loss: 821046.0000
    Epoch 810/2000
    8/8 [==============================] - 0s 4ms/step - loss: 820371.0000 - val_loss: 821298.8125
    Epoch 811/2000
    8/8 [==============================] - 0s 4ms/step - loss: 821278.3750 - val_loss: 819001.6250
    Epoch 812/2000
    8/8 [==============================] - 0s 4ms/step - loss: 820372.5000 - val_loss: 820814.1250
    Epoch 813/2000
    8/8 [==============================] - 0s 4ms/step - loss: 819853.9375 - val_loss: 819923.0625
    Epoch 814/2000
    8/8 [==============================] - 0s 4ms/step - loss: 820867.7500 - val_loss: 817352.0625
    Epoch 815/2000
    8/8 [==============================] - 0s 4ms/step - loss: 818714.9375 - val_loss: 818805.2500
    Epoch 816/2000
    8/8 [==============================] - 0s 4ms/step - loss: 818235.9375 - val_loss: 817191.5000
    Epoch 817/2000
    8/8 [==============================] - 0s 4ms/step - loss: 817931.3750 - val_loss: 820229.3750
    Epoch 818/2000
    8/8 [==============================] - 0s 4ms/step - loss: 818948.9375 - val_loss: 815284.0625
    Epoch 819/2000
    8/8 [==============================] - 0s 4ms/step - loss: 818029.2500 - val_loss: 816056.1875
    Epoch 820/2000
    8/8 [==============================] - 0s 4ms/step - loss: 817332.4375 - val_loss: 816421.7500
    Epoch 821/2000
    8/8 [==============================] - 0s 4ms/step - loss: 815565.8750 - val_loss: 816551.0625
    Epoch 822/2000
    8/8 [==============================] - 0s 4ms/step - loss: 817083.5000 - val_loss: 817945.0000
    Epoch 823/2000
    8/8 [==============================] - 0s 4ms/step - loss: 815633.7500 - val_loss: 815490.6250
    Epoch 824/2000
    8/8 [==============================] - 0s 4ms/step - loss: 815170.0625 - val_loss: 816672.1875
    Epoch 825/2000
    8/8 [==============================] - 0s 4ms/step - loss: 815151.9375 - val_loss: 816175.6875
    Epoch 826/2000
    8/8 [==============================] - 0s 4ms/step - loss: 816441.0000 - val_loss: 814939.5625
    Epoch 827/2000
    8/8 [==============================] - 0s 4ms/step - loss: 815234.1250 - val_loss: 812484.6250
    Epoch 828/2000
    8/8 [==============================] - 0s 4ms/step - loss: 812871.3125 - val_loss: 814177.1250
    Epoch 829/2000
    8/8 [==============================] - 0s 4ms/step - loss: 814990.3125 - val_loss: 811507.8125
    Epoch 830/2000
    8/8 [==============================] - 0s 4ms/step - loss: 813134.8125 - val_loss: 814293.2500
    Epoch 831/2000
    8/8 [==============================] - 0s 4ms/step - loss: 813739.0625 - val_loss: 809927.1875
    Epoch 832/2000
    8/8 [==============================] - 0s 4ms/step - loss: 813667.5000 - val_loss: 812422.0625
    Epoch 833/2000
    8/8 [==============================] - 0s 4ms/step - loss: 812508.9375 - val_loss: 810154.9375
    Epoch 834/2000
    8/8 [==============================] - 0s 4ms/step - loss: 813545.4375 - val_loss: 813135.8750
    Epoch 835/2000
    8/8 [==============================] - 0s 4ms/step - loss: 812696.0625 - val_loss: 813364.1875
    Epoch 836/2000
    8/8 [==============================] - 0s 4ms/step - loss: 810515.2500 - val_loss: 810958.0000
    Epoch 837/2000
    8/8 [==============================] - 0s 4ms/step - loss: 812107.3750 - val_loss: 810783.1875
    Epoch 838/2000
    8/8 [==============================] - 0s 4ms/step - loss: 811112.2500 - val_loss: 811067.1875
    Epoch 839/2000
    8/8 [==============================] - 0s 4ms/step - loss: 809161.3125 - val_loss: 806057.1250
    Epoch 840/2000
    8/8 [==============================] - 0s 4ms/step - loss: 810940.1875 - val_loss: 810448.1250
    Epoch 841/2000
    8/8 [==============================] - 0s 4ms/step - loss: 810345.9375 - val_loss: 809180.1875
    Epoch 842/2000
    8/8 [==============================] - 0s 4ms/step - loss: 810252.0625 - val_loss: 809370.0000
    Epoch 843/2000
    8/8 [==============================] - 0s 4ms/step - loss: 807789.0625 - val_loss: 808393.2500
    Epoch 844/2000
    8/8 [==============================] - 0s 4ms/step - loss: 808305.2500 - val_loss: 808248.3750
    Epoch 845/2000
    8/8 [==============================] - 0s 4ms/step - loss: 808420.1875 - val_loss: 807236.6875
    Epoch 846/2000
    8/8 [==============================] - 0s 4ms/step - loss: 809013.0000 - val_loss: 808551.0625
    Epoch 847/2000
    8/8 [==============================] - 0s 4ms/step - loss: 808460.3125 - val_loss: 806190.0000
    Epoch 848/2000
    8/8 [==============================] - 0s 4ms/step - loss: 807172.0625 - val_loss: 807017.8750
    Epoch 849/2000
    8/8 [==============================] - 0s 4ms/step - loss: 806223.0625 - val_loss: 807013.4375
    Epoch 850/2000
    8/8 [==============================] - 0s 4ms/step - loss: 806732.5625 - val_loss: 806076.0625
    Epoch 851/2000
    8/8 [==============================] - 0s 4ms/step - loss: 806183.1250 - val_loss: 806605.1250
    Epoch 852/2000
    8/8 [==============================] - 0s 4ms/step - loss: 807350.3125 - val_loss: 804408.0625
    Epoch 853/2000
    8/8 [==============================] - 0s 4ms/step - loss: 806101.7500 - val_loss: 806287.4375
    Epoch 854/2000
    8/8 [==============================] - 0s 4ms/step - loss: 805285.7500 - val_loss: 806073.9375
    Epoch 855/2000
    8/8 [==============================] - 0s 4ms/step - loss: 805018.0000 - val_loss: 802624.5000
    Epoch 856/2000
    8/8 [==============================] - 0s 4ms/step - loss: 803778.9375 - val_loss: 805572.9375
    Epoch 857/2000
    8/8 [==============================] - 0s 4ms/step - loss: 802486.8750 - val_loss: 805876.0625
    Epoch 858/2000
    8/8 [==============================] - 0s 4ms/step - loss: 803178.4375 - val_loss: 801937.6250
    Epoch 859/2000
    8/8 [==============================] - 0s 4ms/step - loss: 804810.6250 - val_loss: 803609.8750
    Epoch 860/2000
    8/8 [==============================] - 0s 4ms/step - loss: 803644.1875 - val_loss: 803606.5000
    Epoch 861/2000
    8/8 [==============================] - 0s 4ms/step - loss: 803124.5625 - val_loss: 802855.6875
    Epoch 862/2000
    8/8 [==============================] - 0s 4ms/step - loss: 801497.8125 - val_loss: 801872.7500
    Epoch 863/2000
    8/8 [==============================] - 0s 4ms/step - loss: 800577.0625 - val_loss: 799853.1250
    Epoch 864/2000
    8/8 [==============================] - 0s 4ms/step - loss: 799757.3125 - val_loss: 801330.3750
    Epoch 865/2000
    8/8 [==============================] - 0s 4ms/step - loss: 800188.0000 - val_loss: 801081.3125
    Epoch 866/2000
    8/8 [==============================] - 0s 4ms/step - loss: 801182.3125 - val_loss: 801450.0625
    Epoch 867/2000
    8/8 [==============================] - 0s 4ms/step - loss: 799547.8750 - val_loss: 800196.8750
    Epoch 868/2000
    8/8 [==============================] - 0s 4ms/step - loss: 800616.4375 - val_loss: 799067.8125
    Epoch 869/2000
    8/8 [==============================] - 0s 4ms/step - loss: 799781.6875 - val_loss: 799744.3125
    Epoch 870/2000
    8/8 [==============================] - 0s 4ms/step - loss: 799036.8750 - val_loss: 799669.4375
    Epoch 871/2000
    8/8 [==============================] - 0s 4ms/step - loss: 799224.6250 - val_loss: 798661.0000
    Epoch 872/2000
    8/8 [==============================] - 0s 4ms/step - loss: 800414.6250 - val_loss: 799753.5000
    Epoch 873/2000


    8/8 [==============================] - 0s 4ms/step - loss: 796931.0625 - val_loss: 799169.9375
    Epoch 874/2000
    8/8 [==============================] - 0s 4ms/step - loss: 799409.8750 - val_loss: 795235.8125
    Epoch 875/2000
    8/8 [==============================] - 0s 4ms/step - loss: 797552.0000 - val_loss: 796991.1250
    Epoch 876/2000
    8/8 [==============================] - 0s 4ms/step - loss: 797941.0625 - val_loss: 797459.0000
    Epoch 877/2000
    8/8 [==============================] - 0s 4ms/step - loss: 796578.3750 - val_loss: 797363.2500
    Epoch 878/2000
    8/8 [==============================] - 0s 4ms/step - loss: 798072.5625 - val_loss: 794705.5000
    Epoch 879/2000
    8/8 [==============================] - 0s 4ms/step - loss: 796732.0000 - val_loss: 794900.4375
    Epoch 880/2000
    8/8 [==============================] - 0s 4ms/step - loss: 795751.8125 - val_loss: 797483.0000
    Epoch 881/2000
    8/8 [==============================] - 0s 4ms/step - loss: 797609.9375 - val_loss: 795519.0625
    Epoch 882/2000
    8/8 [==============================] - 0s 4ms/step - loss: 795084.5625 - val_loss: 794518.9375
    Epoch 883/2000
    8/8 [==============================] - 0s 4ms/step - loss: 793388.6875 - val_loss: 797814.5625
    Epoch 884/2000
    8/8 [==============================] - 0s 4ms/step - loss: 795008.0625 - val_loss: 797065.5000
    Epoch 885/2000
    8/8 [==============================] - 0s 4ms/step - loss: 794593.2500 - val_loss: 793936.5625
    Epoch 886/2000
    8/8 [==============================] - 0s 4ms/step - loss: 794071.7500 - val_loss: 793570.0625
    Epoch 887/2000
    8/8 [==============================] - 0s 4ms/step - loss: 793380.0000 - val_loss: 793087.8750
    Epoch 888/2000
    8/8 [==============================] - 0s 4ms/step - loss: 793719.7500 - val_loss: 793905.6250
    Epoch 889/2000
    8/8 [==============================] - 0s 4ms/step - loss: 793404.8750 - val_loss: 793934.6250
    Epoch 890/2000
    8/8 [==============================] - 0s 4ms/step - loss: 792654.0000 - val_loss: 791172.5625
    Epoch 891/2000
    8/8 [==============================] - 0s 4ms/step - loss: 791735.8750 - val_loss: 793051.2500
    Epoch 892/2000
    8/8 [==============================] - 0s 4ms/step - loss: 790836.2500 - val_loss: 792299.0000
    Epoch 893/2000
    8/8 [==============================] - 0s 4ms/step - loss: 791935.3125 - val_loss: 792193.8125
    Epoch 894/2000
    8/8 [==============================] - 0s 4ms/step - loss: 790248.3125 - val_loss: 791892.2500
    Epoch 895/2000
    8/8 [==============================] - 0s 4ms/step - loss: 791219.3125 - val_loss: 789075.5625
    Epoch 896/2000
    8/8 [==============================] - 0s 4ms/step - loss: 791004.8125 - val_loss: 789339.5000
    Epoch 897/2000
    8/8 [==============================] - 0s 4ms/step - loss: 789376.8750 - val_loss: 788919.8125
    Epoch 898/2000
    8/8 [==============================] - 0s 4ms/step - loss: 788659.0625 - val_loss: 791206.6250
    Epoch 899/2000
    8/8 [==============================] - 0s 4ms/step - loss: 789046.1875 - val_loss: 793457.5000
    Epoch 900/2000
    8/8 [==============================] - 0s 4ms/step - loss: 790486.2500 - val_loss: 788544.1875
    Epoch 901/2000
    8/8 [==============================] - 0s 4ms/step - loss: 790333.9375 - val_loss: 788564.3125
    Epoch 902/2000
    8/8 [==============================] - 0s 4ms/step - loss: 787979.1875 - val_loss: 786464.0000
    Epoch 903/2000
    8/8 [==============================] - 0s 4ms/step - loss: 789307.3125 - val_loss: 787946.4375
    Epoch 904/2000
    8/8 [==============================] - 0s 4ms/step - loss: 786753.9375 - val_loss: 787385.9375
    Epoch 905/2000
    8/8 [==============================] - 0s 4ms/step - loss: 786120.7500 - val_loss: 788625.0000
    Epoch 906/2000
    8/8 [==============================] - 0s 4ms/step - loss: 785343.6250 - val_loss: 787920.8125
    Epoch 907/2000
    8/8 [==============================] - 0s 4ms/step - loss: 787180.0000 - val_loss: 786653.0625
    Epoch 908/2000
    8/8 [==============================] - 0s 4ms/step - loss: 788197.0625 - val_loss: 787949.2500
    Epoch 909/2000
    8/8 [==============================] - 0s 4ms/step - loss: 784849.9375 - val_loss: 782392.1875
    Epoch 910/2000
    8/8 [==============================] - 0s 4ms/step - loss: 784565.5625 - val_loss: 785716.0000
    Epoch 911/2000
    8/8 [==============================] - 0s 4ms/step - loss: 784229.5000 - val_loss: 786552.2500
    Epoch 912/2000
    8/8 [==============================] - 0s 4ms/step - loss: 786005.8750 - val_loss: 785025.8125
    Epoch 913/2000
    8/8 [==============================] - 0s 4ms/step - loss: 785556.0000 - val_loss: 785670.3750
    Epoch 914/2000
    8/8 [==============================] - 0s 5ms/step - loss: 784755.1250 - val_loss: 785135.0625
    Epoch 915/2000
    8/8 [==============================] - 0s 4ms/step - loss: 783675.9375 - val_loss: 784676.1875
    Epoch 916/2000
    8/8 [==============================] - 0s 4ms/step - loss: 784463.9375 - val_loss: 784133.2500
    Epoch 917/2000
    8/8 [==============================] - 0s 4ms/step - loss: 784147.5000 - val_loss: 786057.4375
    Epoch 918/2000
    8/8 [==============================] - 0s 4ms/step - loss: 784387.1250 - val_loss: 783201.6875
    Epoch 919/2000
    8/8 [==============================] - 0s 4ms/step - loss: 783604.6875 - val_loss: 784502.1875
    Epoch 920/2000
    8/8 [==============================] - 0s 4ms/step - loss: 782949.8125 - val_loss: 780221.6875
    Epoch 921/2000
    8/8 [==============================] - 0s 4ms/step - loss: 784098.5625 - val_loss: 783095.0625
    Epoch 922/2000
    8/8 [==============================] - 0s 4ms/step - loss: 782425.4375 - val_loss: 781616.0000
    Epoch 923/2000
    8/8 [==============================] - 0s 4ms/step - loss: 782116.3125 - val_loss: 781079.8125
    Epoch 924/2000
    8/8 [==============================] - 0s 4ms/step - loss: 782545.2500 - val_loss: 781211.8750
    Epoch 925/2000
    8/8 [==============================] - 0s 4ms/step - loss: 781931.5000 - val_loss: 780666.0625
    Epoch 926/2000
    8/8 [==============================] - 0s 4ms/step - loss: 779315.1875 - val_loss: 781100.9375
    Epoch 927/2000
    8/8 [==============================] - 0s 4ms/step - loss: 781327.1875 - val_loss: 778181.5000
    Epoch 928/2000
    8/8 [==============================] - 0s 4ms/step - loss: 780290.0000 - val_loss: 778408.4375
    Epoch 929/2000
    8/8 [==============================] - 0s 4ms/step - loss: 779337.1250 - val_loss: 778344.8750
    Epoch 930/2000
    8/8 [==============================] - 0s 4ms/step - loss: 779506.0000 - val_loss: 781713.5625
    Epoch 931/2000
    8/8 [==============================] - 0s 4ms/step - loss: 779407.4375 - val_loss: 780909.3125
    Epoch 932/2000
    8/8 [==============================] - 0s 4ms/step - loss: 778951.4375 - val_loss: 777490.2500
    Epoch 933/2000
    8/8 [==============================] - 0s 4ms/step - loss: 778163.1875 - val_loss: 778439.0625
    Epoch 934/2000
    8/8 [==============================] - 0s 4ms/step - loss: 776518.7500 - val_loss: 778499.9375
    Epoch 935/2000
    8/8 [==============================] - 0s 4ms/step - loss: 778469.5000 - val_loss: 776407.4375
    Epoch 936/2000
    8/8 [==============================] - 0s 4ms/step - loss: 777479.0625 - val_loss: 778204.3750
    Epoch 937/2000
    8/8 [==============================] - 0s 4ms/step - loss: 778207.6250 - val_loss: 776959.5625
    Epoch 938/2000
    8/8 [==============================] - 0s 4ms/step - loss: 777399.4375 - val_loss: 775702.3750
    Epoch 939/2000
    8/8 [==============================] - 0s 4ms/step - loss: 776348.2500 - val_loss: 776718.7500
    Epoch 940/2000
    8/8 [==============================] - 0s 4ms/step - loss: 776304.9375 - val_loss: 776077.2500
    Epoch 941/2000
    8/8 [==============================] - 0s 4ms/step - loss: 775209.3125 - val_loss: 774059.8125
    Epoch 942/2000
    8/8 [==============================] - 0s 4ms/step - loss: 775614.7500 - val_loss: 776697.7500
    Epoch 943/2000
    8/8 [==============================] - 0s 4ms/step - loss: 775760.8125 - val_loss: 773865.6875
    Epoch 944/2000
    8/8 [==============================] - 0s 4ms/step - loss: 774915.0000 - val_loss: 774064.1250
    Epoch 945/2000
    8/8 [==============================] - 0s 4ms/step - loss: 774864.8125 - val_loss: 771671.4375
    Epoch 946/2000
    8/8 [==============================] - 0s 4ms/step - loss: 773189.2500 - val_loss: 773641.6250
    Epoch 947/2000
    8/8 [==============================] - 0s 4ms/step - loss: 774285.0000 - val_loss: 774952.5625
    Epoch 948/2000
    8/8 [==============================] - 0s 4ms/step - loss: 773431.3750 - val_loss: 772930.7500
    Epoch 949/2000
    8/8 [==============================] - 0s 4ms/step - loss: 771112.1875 - val_loss: 773876.3750
    Epoch 950/2000
    8/8 [==============================] - 0s 4ms/step - loss: 772890.7500 - val_loss: 771908.7500
    Epoch 951/2000
    8/8 [==============================] - 0s 4ms/step - loss: 771318.3125 - val_loss: 775201.6250
    Epoch 952/2000
    8/8 [==============================] - 0s 4ms/step - loss: 773079.8750 - val_loss: 772772.5625
    Epoch 953/2000
    8/8 [==============================] - 0s 4ms/step - loss: 772352.5625 - val_loss: 769428.9375
    Epoch 954/2000
    8/8 [==============================] - 0s 4ms/step - loss: 772194.5625 - val_loss: 770128.5625
    Epoch 955/2000
    8/8 [==============================] - 0s 4ms/step - loss: 771752.4375 - val_loss: 768373.7500
    Epoch 956/2000
    8/8 [==============================] - 0s 4ms/step - loss: 770676.2500 - val_loss: 769975.9375
    Epoch 957/2000
    8/8 [==============================] - 0s 4ms/step - loss: 768661.1250 - val_loss: 769150.1875
    Epoch 958/2000
    8/8 [==============================] - 0s 4ms/step - loss: 770734.0000 - val_loss: 770146.0000
    Epoch 959/2000
    8/8 [==============================] - 0s 4ms/step - loss: 770268.1875 - val_loss: 768640.9375
    Epoch 960/2000
    8/8 [==============================] - 0s 4ms/step - loss: 770261.1875 - val_loss: 770106.5625
    Epoch 961/2000
    8/8 [==============================] - 0s 4ms/step - loss: 769367.0000 - val_loss: 770209.6250
    Epoch 962/2000
    8/8 [==============================] - 0s 4ms/step - loss: 769762.0625 - val_loss: 769674.7500
    Epoch 963/2000
    8/8 [==============================] - 0s 4ms/step - loss: 768822.2500 - val_loss: 768447.8125
    Epoch 964/2000
    8/8 [==============================] - 0s 4ms/step - loss: 767263.8125 - val_loss: 768603.8125
    Epoch 965/2000
    8/8 [==============================] - 0s 4ms/step - loss: 768215.6250 - val_loss: 770540.0000
    Epoch 966/2000
    8/8 [==============================] - 0s 4ms/step - loss: 767951.8750 - val_loss: 767550.3750
    Epoch 967/2000
    8/8 [==============================] - 0s 4ms/step - loss: 768550.1875 - val_loss: 767335.8750
    Epoch 968/2000
    8/8 [==============================] - 0s 4ms/step - loss: 766760.8125 - val_loss: 769309.2500
    Epoch 969/2000
    8/8 [==============================] - 0s 4ms/step - loss: 767086.0000 - val_loss: 766095.6875
    Epoch 970/2000
    8/8 [==============================] - 0s 4ms/step - loss: 767033.0000 - val_loss: 765914.7500
    Epoch 971/2000
    8/8 [==============================] - 0s 4ms/step - loss: 765517.4375 - val_loss: 765833.2500
    Epoch 972/2000
    8/8 [==============================] - 0s 4ms/step - loss: 767020.5625 - val_loss: 765747.8125
    Epoch 973/2000
    8/8 [==============================] - 0s 4ms/step - loss: 766695.5625 - val_loss: 766522.7500
    Epoch 974/2000
    8/8 [==============================] - 0s 4ms/step - loss: 765168.8125 - val_loss: 765938.8750
    Epoch 975/2000
    8/8 [==============================] - 0s 4ms/step - loss: 763791.3125 - val_loss: 765134.4375
    Epoch 976/2000
    8/8 [==============================] - 0s 4ms/step - loss: 765541.0625 - val_loss: 765133.3125
    Epoch 977/2000
    8/8 [==============================] - 0s 4ms/step - loss: 763963.7500 - val_loss: 763907.1875
    Epoch 978/2000
    8/8 [==============================] - 0s 4ms/step - loss: 764467.0000 - val_loss: 761589.1875
    Epoch 979/2000
    8/8 [==============================] - 0s 4ms/step - loss: 763348.1250 - val_loss: 767339.6250
    Epoch 980/2000
    8/8 [==============================] - 0s 4ms/step - loss: 763757.8750 - val_loss: 763415.0625
    Epoch 981/2000
    8/8 [==============================] - 0s 4ms/step - loss: 763703.3750 - val_loss: 763939.7500
    Epoch 982/2000
    8/8 [==============================] - 0s 4ms/step - loss: 762066.8750 - val_loss: 762293.2500
    Epoch 983/2000
    8/8 [==============================] - 0s 4ms/step - loss: 762739.1250 - val_loss: 763812.6875
    Epoch 984/2000
    8/8 [==============================] - 0s 4ms/step - loss: 760624.4375 - val_loss: 762839.2500
    Epoch 985/2000
    8/8 [==============================] - 0s 4ms/step - loss: 762248.6250 - val_loss: 762590.6250
    Epoch 986/2000
    8/8 [==============================] - 0s 4ms/step - loss: 760837.8750 - val_loss: 762968.1250
    Epoch 987/2000
    8/8 [==============================] - 0s 4ms/step - loss: 760064.3750 - val_loss: 757303.0625
    Epoch 988/2000
    8/8 [==============================] - 0s 4ms/step - loss: 762161.1875 - val_loss: 762033.0625
    Epoch 989/2000
    8/8 [==============================] - 0s 4ms/step - loss: 761195.8750 - val_loss: 760094.7500
    Epoch 990/2000
    8/8 [==============================] - 0s 4ms/step - loss: 759118.7500 - val_loss: 757808.8750
    Epoch 991/2000
    8/8 [==============================] - 0s 4ms/step - loss: 759295.1875 - val_loss: 760238.0000
    Epoch 992/2000
    8/8 [==============================] - 0s 4ms/step - loss: 761028.1875 - val_loss: 759691.6250
    Epoch 993/2000
    8/8 [==============================] - 0s 4ms/step - loss: 760515.1875 - val_loss: 759012.6875
    Epoch 994/2000
    8/8 [==============================] - 0s 4ms/step - loss: 756867.2500 - val_loss: 757419.8125
    Epoch 995/2000
    8/8 [==============================] - 0s 4ms/step - loss: 758204.0000 - val_loss: 760162.4375
    Epoch 996/2000
    8/8 [==============================] - 0s 4ms/step - loss: 757128.0625 - val_loss: 760252.1875
    Epoch 997/2000
    8/8 [==============================] - 0s 4ms/step - loss: 757455.8125 - val_loss: 758633.9375
    Epoch 998/2000
    8/8 [==============================] - 0s 4ms/step - loss: 758749.8750 - val_loss: 756058.7500
    Epoch 999/2000
    8/8 [==============================] - 0s 4ms/step - loss: 758813.3750 - val_loss: 757599.5000
    Epoch 1000/2000
    8/8 [==============================] - 0s 4ms/step - loss: 758045.7500 - val_loss: 758399.6875
    Epoch 1001/2000
    8/8 [==============================] - 0s 4ms/step - loss: 757479.0000 - val_loss: 757756.0000
    Epoch 1002/2000
    8/8 [==============================] - 0s 4ms/step - loss: 756600.1875 - val_loss: 757644.3750
    Epoch 1003/2000
    8/8 [==============================] - 0s 4ms/step - loss: 755991.6250 - val_loss: 756792.4375
    Epoch 1004/2000
    8/8 [==============================] - 0s 4ms/step - loss: 755665.2500 - val_loss: 754007.5000
    Epoch 1005/2000
    8/8 [==============================] - 0s 4ms/step - loss: 756093.1250 - val_loss: 755273.0000
    Epoch 1006/2000
    8/8 [==============================] - 0s 4ms/step - loss: 754617.3750 - val_loss: 754682.0000
    Epoch 1007/2000
    8/8 [==============================] - 0s 4ms/step - loss: 755486.4375 - val_loss: 753895.0000
    Epoch 1008/2000
    8/8 [==============================] - 0s 4ms/step - loss: 755009.2500 - val_loss: 757410.4375
    Epoch 1009/2000
    8/8 [==============================] - 0s 4ms/step - loss: 754621.0000 - val_loss: 753508.7500
    Epoch 1010/2000
    8/8 [==============================] - 0s 4ms/step - loss: 755229.3125 - val_loss: 750795.6875
    Epoch 1011/2000
    8/8 [==============================] - 0s 4ms/step - loss: 755726.0625 - val_loss: 750673.0625
    Epoch 1012/2000
    8/8 [==============================] - 0s 4ms/step - loss: 754363.6875 - val_loss: 751965.8750
    Epoch 1013/2000
    8/8 [==============================] - 0s 4ms/step - loss: 754386.0000 - val_loss: 751324.1250
    Epoch 1014/2000
    8/8 [==============================] - 0s 4ms/step - loss: 754135.1250 - val_loss: 753311.5000
    Epoch 1015/2000
    8/8 [==============================] - 0s 4ms/step - loss: 752677.1875 - val_loss: 752363.8125
    Epoch 1016/2000
    8/8 [==============================] - 0s 4ms/step - loss: 751593.5000 - val_loss: 754364.1875
    Epoch 1017/2000
    8/8 [==============================] - 0s 4ms/step - loss: 752266.3750 - val_loss: 752725.0625
    Epoch 1018/2000
    8/8 [==============================] - 0s 4ms/step - loss: 752661.8750 - val_loss: 750051.0625
    Epoch 1019/2000
    8/8 [==============================] - 0s 4ms/step - loss: 752404.1875 - val_loss: 749778.3750
    Epoch 1020/2000
    8/8 [==============================] - 0s 4ms/step - loss: 751571.1875 - val_loss: 751389.3125
    Epoch 1021/2000


    8/8 [==============================] - 0s 4ms/step - loss: 750508.4375 - val_loss: 753044.2500
    Epoch 1022/2000
    8/8 [==============================] - 0s 4ms/step - loss: 751190.0000 - val_loss: 753590.8750
    Epoch 1023/2000
    8/8 [==============================] - 0s 4ms/step - loss: 749881.6875 - val_loss: 749289.0000
    Epoch 1024/2000
    8/8 [==============================] - 0s 4ms/step - loss: 750117.6875 - val_loss: 749043.8125
    Epoch 1025/2000
    8/8 [==============================] - 0s 4ms/step - loss: 748805.6250 - val_loss: 751067.6250
    Epoch 1026/2000
    8/8 [==============================] - 0s 4ms/step - loss: 749093.2500 - val_loss: 749518.3750
    Epoch 1027/2000
    8/8 [==============================] - 0s 4ms/step - loss: 748860.5625 - val_loss: 747959.9375
    Epoch 1028/2000
    8/8 [==============================] - 0s 4ms/step - loss: 747814.0000 - val_loss: 750259.8125
    Epoch 1029/2000
    8/8 [==============================] - 0s 4ms/step - loss: 749199.3125 - val_loss: 747686.7500
    Epoch 1030/2000
    8/8 [==============================] - 0s 4ms/step - loss: 747890.1875 - val_loss: 750579.4375
    Epoch 1031/2000
    8/8 [==============================] - 0s 4ms/step - loss: 747658.7500 - val_loss: 746239.2500
    Epoch 1032/2000
    8/8 [==============================] - 0s 4ms/step - loss: 747513.4375 - val_loss: 746931.2500
    Epoch 1033/2000
    8/8 [==============================] - 0s 4ms/step - loss: 747109.5000 - val_loss: 745774.1875
    Epoch 1034/2000
    8/8 [==============================] - 0s 4ms/step - loss: 747930.5000 - val_loss: 745988.6250
    Epoch 1035/2000
    8/8 [==============================] - 0s 4ms/step - loss: 745581.1250 - val_loss: 743032.9375
    Epoch 1036/2000
    8/8 [==============================] - 0s 4ms/step - loss: 744629.3750 - val_loss: 748755.6250
    Epoch 1037/2000
    8/8 [==============================] - 0s 4ms/step - loss: 747100.9375 - val_loss: 747516.9375
    Epoch 1038/2000
    8/8 [==============================] - 0s 4ms/step - loss: 744794.2500 - val_loss: 743515.4375
    Epoch 1039/2000
    8/8 [==============================] - 0s 4ms/step - loss: 745932.7500 - val_loss: 746704.6250
    Epoch 1040/2000
    8/8 [==============================] - 0s 4ms/step - loss: 744155.1875 - val_loss: 746215.9375
    Epoch 1041/2000
    8/8 [==============================] - 0s 4ms/step - loss: 744180.6875 - val_loss: 743218.8750
    Epoch 1042/2000
    8/8 [==============================] - 0s 4ms/step - loss: 743719.0000 - val_loss: 742762.6875
    Epoch 1043/2000
    8/8 [==============================] - 0s 4ms/step - loss: 743185.2500 - val_loss: 744047.2500
    Epoch 1044/2000
    8/8 [==============================] - 0s 4ms/step - loss: 744942.0000 - val_loss: 743167.8750
    Epoch 1045/2000
    8/8 [==============================] - 0s 4ms/step - loss: 743434.6875 - val_loss: 743296.6250
    Epoch 1046/2000
    8/8 [==============================] - 0s 4ms/step - loss: 742634.9375 - val_loss: 742293.3750
    Epoch 1047/2000
    8/8 [==============================] - 0s 4ms/step - loss: 741811.1250 - val_loss: 743324.0625
    Epoch 1048/2000
    8/8 [==============================] - 0s 4ms/step - loss: 743240.5625 - val_loss: 742991.8125
    Epoch 1049/2000
    8/8 [==============================] - 0s 4ms/step - loss: 742466.6250 - val_loss: 742326.8750
    Epoch 1050/2000
    8/8 [==============================] - 0s 4ms/step - loss: 742863.0625 - val_loss: 740921.1875
    Epoch 1051/2000
    8/8 [==============================] - 0s 4ms/step - loss: 741660.7500 - val_loss: 740656.7500
    Epoch 1052/2000
    8/8 [==============================] - 0s 4ms/step - loss: 740937.4375 - val_loss: 738929.4375
    Epoch 1053/2000
    8/8 [==============================] - 0s 4ms/step - loss: 742153.9375 - val_loss: 738158.3750
    Epoch 1054/2000
    8/8 [==============================] - 0s 4ms/step - loss: 741558.3125 - val_loss: 739799.6875
    Epoch 1055/2000
    8/8 [==============================] - 0s 4ms/step - loss: 740409.6875 - val_loss: 740331.0625
    Epoch 1056/2000
    8/8 [==============================] - 0s 4ms/step - loss: 741539.1250 - val_loss: 737195.5000
    Epoch 1057/2000
    8/8 [==============================] - 0s 4ms/step - loss: 741650.6250 - val_loss: 741586.3750
    Epoch 1058/2000
    8/8 [==============================] - 0s 4ms/step - loss: 738453.8750 - val_loss: 740506.4375
    Epoch 1059/2000
    8/8 [==============================] - 0s 4ms/step - loss: 739982.5625 - val_loss: 740600.3125
    Epoch 1060/2000
    8/8 [==============================] - 0s 4ms/step - loss: 738771.0625 - val_loss: 739262.1250
    Epoch 1061/2000
    8/8 [==============================] - 0s 4ms/step - loss: 738571.6875 - val_loss: 737689.0625
    Epoch 1062/2000
    8/8 [==============================] - 0s 4ms/step - loss: 740501.5000 - val_loss: 737727.9375
    Epoch 1063/2000
    8/8 [==============================] - 0s 4ms/step - loss: 737549.1250 - val_loss: 738031.4375
    Epoch 1064/2000
    8/8 [==============================] - 0s 4ms/step - loss: 738346.3125 - val_loss: 736918.7500
    Epoch 1065/2000
    8/8 [==============================] - 0s 4ms/step - loss: 738424.1875 - val_loss: 735693.6875
    Epoch 1066/2000
    8/8 [==============================] - 0s 4ms/step - loss: 737435.3125 - val_loss: 735242.3750
    Epoch 1067/2000
    8/8 [==============================] - 0s 4ms/step - loss: 737185.0625 - val_loss: 741311.9375
    Epoch 1068/2000
    8/8 [==============================] - 0s 4ms/step - loss: 736295.2500 - val_loss: 737590.5000
    Epoch 1069/2000
    8/8 [==============================] - 0s 4ms/step - loss: 737144.5625 - val_loss: 735375.2500
    Epoch 1070/2000
    8/8 [==============================] - 0s 4ms/step - loss: 735447.6250 - val_loss: 734508.5000
    Epoch 1071/2000
    8/8 [==============================] - 0s 4ms/step - loss: 737001.4375 - val_loss: 731459.9375
    Epoch 1072/2000
    8/8 [==============================] - 0s 4ms/step - loss: 736386.0000 - val_loss: 737711.5625
    Epoch 1073/2000
    8/8 [==============================] - 0s 4ms/step - loss: 736738.8750 - val_loss: 736142.0625
    Epoch 1074/2000
    8/8 [==============================] - 0s 4ms/step - loss: 735942.6875 - val_loss: 734074.6250
    Epoch 1075/2000
    8/8 [==============================] - 0s 4ms/step - loss: 732337.4375 - val_loss: 734876.1875
    Epoch 1076/2000
    8/8 [==============================] - 0s 4ms/step - loss: 735573.6250 - val_loss: 734216.5625
    Epoch 1077/2000
    8/8 [==============================] - 0s 4ms/step - loss: 736036.6875 - val_loss: 733705.5000
    Epoch 1078/2000
    8/8 [==============================] - 0s 4ms/step - loss: 734442.5000 - val_loss: 734671.8750
    Epoch 1079/2000
    8/8 [==============================] - 0s 4ms/step - loss: 734917.5000 - val_loss: 733160.5625
    Epoch 1080/2000
    8/8 [==============================] - 0s 4ms/step - loss: 734016.3750 - val_loss: 733708.4375
    Epoch 1081/2000
    8/8 [==============================] - 0s 4ms/step - loss: 733208.8125 - val_loss: 735345.5000
    Epoch 1082/2000
    8/8 [==============================] - 0s 4ms/step - loss: 731672.6250 - val_loss: 734099.3750
    Epoch 1083/2000
    8/8 [==============================] - 0s 4ms/step - loss: 732634.9375 - val_loss: 733826.9375
    Epoch 1084/2000
    8/8 [==============================] - 0s 4ms/step - loss: 734229.3125 - val_loss: 733631.9375
    Epoch 1085/2000
    8/8 [==============================] - 0s 4ms/step - loss: 732505.4375 - val_loss: 733470.5000
    Epoch 1086/2000
    8/8 [==============================] - 0s 4ms/step - loss: 730589.4375 - val_loss: 730461.6250
    Epoch 1087/2000
    8/8 [==============================] - 0s 4ms/step - loss: 731227.5000 - val_loss: 727852.0625
    Epoch 1088/2000
    8/8 [==============================] - 0s 4ms/step - loss: 730932.2500 - val_loss: 731276.1875
    Epoch 1089/2000
    8/8 [==============================] - 0s 4ms/step - loss: 731676.4375 - val_loss: 728846.5625
    Epoch 1090/2000
    8/8 [==============================] - 0s 4ms/step - loss: 730300.8750 - val_loss: 730433.6250
    Epoch 1091/2000
    8/8 [==============================] - 0s 4ms/step - loss: 730360.0625 - val_loss: 731540.2500
    Epoch 1092/2000
    8/8 [==============================] - 0s 4ms/step - loss: 727789.6250 - val_loss: 727693.6875
    Epoch 1093/2000
    8/8 [==============================] - 0s 4ms/step - loss: 730650.5000 - val_loss: 730527.6250
    Epoch 1094/2000
    8/8 [==============================] - 0s 4ms/step - loss: 728552.2500 - val_loss: 728639.8125


    Epoch 1095/2000
    8/8 [==============================] - 0s 4ms/step - loss: 729744.0625 - val_loss: 730731.8125
    Epoch 1096/2000
    8/8 [==============================] - 0s 4ms/step - loss: 729014.5000 - val_loss: 730090.1875
    Epoch 1097/2000
    8/8 [==============================] - 0s 4ms/step - loss: 727920.6250 - val_loss: 727128.8125
    Epoch 1098/2000
    8/8 [==============================] - 0s 4ms/step - loss: 727261.1250 - val_loss: 728624.9375
    Epoch 1099/2000
    8/8 [==============================] - 0s 4ms/step - loss: 727598.3750 - val_loss: 727253.5000
    Epoch 1100/2000
    8/8 [==============================] - 0s 4ms/step - loss: 727323.3750 - val_loss: 726691.1875
    Epoch 1101/2000
    8/8 [==============================] - 0s 4ms/step - loss: 728786.6250 - val_loss: 726740.1250
    Epoch 1102/2000
    8/8 [==============================] - 0s 4ms/step - loss: 726492.8750 - val_loss: 725635.0000
    Epoch 1103/2000
    8/8 [==============================] - 0s 4ms/step - loss: 726547.1875 - val_loss: 728471.8125
    Epoch 1104/2000
    8/8 [==============================] - 0s 4ms/step - loss: 727565.1250 - val_loss: 730807.9375
    Epoch 1105/2000
    8/8 [==============================] - 0s 4ms/step - loss: 725854.5000 - val_loss: 726032.4375
    Epoch 1106/2000
    8/8 [==============================] - 0s 4ms/step - loss: 726266.8750 - val_loss: 725972.2500
    Epoch 1107/2000
    8/8 [==============================] - 0s 4ms/step - loss: 726833.0625 - val_loss: 725504.3750
    Epoch 1108/2000
    8/8 [==============================] - 0s 4ms/step - loss: 724875.4375 - val_loss: 724399.3125
    Epoch 1109/2000
    8/8 [==============================] - 0s 4ms/step - loss: 725274.3125 - val_loss: 725236.1250
    Epoch 1110/2000
    8/8 [==============================] - 0s 4ms/step - loss: 726547.2500 - val_loss: 724570.2500
    Epoch 1111/2000
    8/8 [==============================] - 0s 4ms/step - loss: 724626.0000 - val_loss: 724232.5000
    Epoch 1112/2000
    8/8 [==============================] - 0s 4ms/step - loss: 726468.4375 - val_loss: 725865.8125
    Epoch 1113/2000
    8/8 [==============================] - 0s 4ms/step - loss: 724992.6250 - val_loss: 724777.6250
    Epoch 1114/2000
    8/8 [==============================] - 0s 4ms/step - loss: 725685.7500 - val_loss: 724133.5000
    Epoch 1115/2000
    8/8 [==============================] - 0s 4ms/step - loss: 724273.1250 - val_loss: 724104.7500
    Epoch 1116/2000
    8/8 [==============================] - 0s 4ms/step - loss: 724216.8750 - val_loss: 720449.4375
    Epoch 1117/2000
    8/8 [==============================] - 0s 4ms/step - loss: 723410.7500 - val_loss: 723638.5625
    Epoch 1118/2000
    8/8 [==============================] - 0s 4ms/step - loss: 722572.1875 - val_loss: 722443.5000
    Epoch 1119/2000
    8/8 [==============================] - 0s 4ms/step - loss: 724702.3125 - val_loss: 720344.8750
    Epoch 1120/2000
    8/8 [==============================] - 0s 4ms/step - loss: 721638.5625 - val_loss: 720060.5625
    Epoch 1121/2000
    8/8 [==============================] - 0s 4ms/step - loss: 722116.0000 - val_loss: 723494.7500
    Epoch 1122/2000
    8/8 [==============================] - 0s 4ms/step - loss: 720680.4375 - val_loss: 721326.7500
    Epoch 1123/2000
    8/8 [==============================] - 0s 4ms/step - loss: 720114.6250 - val_loss: 721031.0625
    Epoch 1124/2000
    8/8 [==============================] - 0s 4ms/step - loss: 720220.5000 - val_loss: 723559.4375
    Epoch 1125/2000
    8/8 [==============================] - 0s 4ms/step - loss: 720991.6875 - val_loss: 721796.1250
    Epoch 1126/2000
    8/8 [==============================] - 0s 4ms/step - loss: 719725.8750 - val_loss: 719960.2500
    Epoch 1127/2000
    8/8 [==============================] - 0s 4ms/step - loss: 719681.7500 - val_loss: 723248.9375
    Epoch 1128/2000
    8/8 [==============================] - 0s 4ms/step - loss: 720264.8125 - val_loss: 719939.5000
    Epoch 1129/2000
    8/8 [==============================] - 0s 4ms/step - loss: 720258.5625 - val_loss: 716697.8750
    Epoch 1130/2000
    8/8 [==============================] - 0s 4ms/step - loss: 718572.5625 - val_loss: 716207.2500
    Epoch 1131/2000
    8/8 [==============================] - 0s 4ms/step - loss: 719490.6250 - val_loss: 719807.7500
    Epoch 1132/2000
    8/8 [==============================] - 0s 4ms/step - loss: 718884.3125 - val_loss: 716271.8125
    Epoch 1133/2000
    8/8 [==============================] - 0s 4ms/step - loss: 717435.2500 - val_loss: 717664.0000
    Epoch 1134/2000
    8/8 [==============================] - 0s 4ms/step - loss: 717822.0625 - val_loss: 716089.6250
    Epoch 1135/2000
    8/8 [==============================] - 0s 4ms/step - loss: 717702.3125 - val_loss: 717681.5000
    Epoch 1136/2000
    8/8 [==============================] - 0s 4ms/step - loss: 718454.1250 - val_loss: 719051.4375
    Epoch 1137/2000
    8/8 [==============================] - 0s 4ms/step - loss: 717521.1875 - val_loss: 717396.3125
    Epoch 1138/2000
    8/8 [==============================] - 0s 4ms/step - loss: 717255.6875 - val_loss: 717792.5625
    Epoch 1139/2000
    8/8 [==============================] - 0s 4ms/step - loss: 718097.6250 - val_loss: 716297.5000
    Epoch 1140/2000
    8/8 [==============================] - 0s 4ms/step - loss: 716375.6875 - val_loss: 714539.5000
    Epoch 1141/2000
    8/8 [==============================] - 0s 4ms/step - loss: 716303.5625 - val_loss: 713205.7500
    Epoch 1142/2000
    8/8 [==============================] - 0s 4ms/step - loss: 716023.1875 - val_loss: 717748.1875
    Epoch 1143/2000
    8/8 [==============================] - 0s 4ms/step - loss: 716360.0625 - val_loss: 717319.1875
    Epoch 1144/2000
    8/8 [==============================] - 0s 4ms/step - loss: 716458.3750 - val_loss: 717164.8750
    Epoch 1145/2000
    8/8 [==============================] - 0s 4ms/step - loss: 714015.8750 - val_loss: 716778.0000
    Epoch 1146/2000
    8/8 [==============================] - 0s 4ms/step - loss: 714872.0625 - val_loss: 713680.4375
    Epoch 1147/2000
    8/8 [==============================] - 0s 4ms/step - loss: 714739.4375 - val_loss: 713837.7500
    Epoch 1148/2000
    8/8 [==============================] - 0s 4ms/step - loss: 714452.1250 - val_loss: 714411.4375
    Epoch 1149/2000
    8/8 [==============================] - 0s 4ms/step - loss: 714748.8750 - val_loss: 713516.8125
    Epoch 1150/2000
    8/8 [==============================] - 0s 4ms/step - loss: 714046.4375 - val_loss: 712448.3750
    Epoch 1151/2000
    8/8 [==============================] - 0s 4ms/step - loss: 712692.5000 - val_loss: 714298.0000
    Epoch 1152/2000
    8/8 [==============================] - 0s 4ms/step - loss: 712747.8750 - val_loss: 712672.4375
    Epoch 1153/2000
    8/8 [==============================] - 0s 4ms/step - loss: 712303.5000 - val_loss: 710029.3125
    Epoch 1154/2000
    8/8 [==============================] - 0s 4ms/step - loss: 712526.2500 - val_loss: 711240.8750
    Epoch 1155/2000
    8/8 [==============================] - 0s 4ms/step - loss: 712957.5000 - val_loss: 711679.3750
    Epoch 1156/2000
    8/8 [==============================] - 0s 4ms/step - loss: 712288.0625 - val_loss: 710812.3750
    Epoch 1157/2000
    8/8 [==============================] - 0s 4ms/step - loss: 713065.3125 - val_loss: 710537.5000
    Epoch 1158/2000
    8/8 [==============================] - 0s 4ms/step - loss: 712018.0625 - val_loss: 712557.8750
    Epoch 1159/2000
    8/8 [==============================] - 0s 4ms/step - loss: 710801.7500 - val_loss: 709552.5000
    Epoch 1160/2000
    8/8 [==============================] - 0s 4ms/step - loss: 710914.1875 - val_loss: 711717.3125
    Epoch 1161/2000
    8/8 [==============================] - 0s 4ms/step - loss: 710540.2500 - val_loss: 711331.5000
    Epoch 1162/2000
    8/8 [==============================] - 0s 4ms/step - loss: 711729.4375 - val_loss: 711266.6250
    Epoch 1163/2000
    8/8 [==============================] - 0s 4ms/step - loss: 709721.7500 - val_loss: 709816.3125
    Epoch 1164/2000
    8/8 [==============================] - 0s 4ms/step - loss: 709965.4375 - val_loss: 709510.6250
    Epoch 1165/2000
    8/8 [==============================] - 0s 4ms/step - loss: 709883.2500 - val_loss: 708893.5000
    Epoch 1166/2000
    8/8 [==============================] - 0s 4ms/step - loss: 710377.5000 - val_loss: 709665.8125
    Epoch 1167/2000
    8/8 [==============================] - 0s 4ms/step - loss: 709893.7500 - val_loss: 712698.5625
    Epoch 1168/2000
    8/8 [==============================] - 0s 4ms/step - loss: 708695.8125 - val_loss: 706615.4375


    Epoch 1169/2000
    8/8 [==============================] - 0s 4ms/step - loss: 708230.2500 - val_loss: 707785.6250
    Epoch 1170/2000
    8/8 [==============================] - 0s 4ms/step - loss: 709618.8750 - val_loss: 711527.1250
    Epoch 1171/2000
    8/8 [==============================] - 0s 4ms/step - loss: 708570.6875 - val_loss: 706641.5625
    Epoch 1172/2000
    8/8 [==============================] - 0s 4ms/step - loss: 708616.1250 - val_loss: 703705.0625
    Epoch 1173/2000
    8/8 [==============================] - 0s 4ms/step - loss: 707843.7500 - val_loss: 707953.8125
    Epoch 1174/2000
    8/8 [==============================] - 0s 4ms/step - loss: 706221.7500 - val_loss: 706504.5625
    Epoch 1175/2000
    8/8 [==============================] - 0s 4ms/step - loss: 707639.8125 - val_loss: 708819.0625
    Epoch 1176/2000
    8/8 [==============================] - 0s 4ms/step - loss: 707712.9375 - val_loss: 707345.8125
    Epoch 1177/2000
    8/8 [==============================] - 0s 4ms/step - loss: 707869.3750 - val_loss: 707547.3750
    Epoch 1178/2000
    8/8 [==============================] - 0s 4ms/step - loss: 705377.7500 - val_loss: 704519.0625
    Epoch 1179/2000
    8/8 [==============================] - 0s 4ms/step - loss: 706063.8125 - val_loss: 705684.9375
    Epoch 1180/2000
    8/8 [==============================] - 0s 4ms/step - loss: 705614.1250 - val_loss: 707580.5000
    Epoch 1181/2000
    8/8 [==============================] - 0s 4ms/step - loss: 706789.3125 - val_loss: 707005.0000
    Epoch 1182/2000
    8/8 [==============================] - 0s 4ms/step - loss: 705343.8750 - val_loss: 705499.4375
    Epoch 1183/2000
    8/8 [==============================] - 0s 4ms/step - loss: 704743.0000 - val_loss: 704600.4375
    Epoch 1184/2000
    8/8 [==============================] - 0s 4ms/step - loss: 705149.9375 - val_loss: 706182.5000
    Epoch 1185/2000
    8/8 [==============================] - 0s 4ms/step - loss: 704532.5625 - val_loss: 703305.3125
    Epoch 1186/2000
    8/8 [==============================] - 0s 4ms/step - loss: 704234.5000 - val_loss: 705650.3750
    Epoch 1187/2000
    8/8 [==============================] - 0s 4ms/step - loss: 704675.9375 - val_loss: 705523.1250
    Epoch 1188/2000
    8/8 [==============================] - 0s 4ms/step - loss: 703229.8125 - val_loss: 702252.8750
    Epoch 1189/2000
    8/8 [==============================] - 0s 4ms/step - loss: 702864.9375 - val_loss: 703753.6250
    Epoch 1190/2000
    8/8 [==============================] - 0s 4ms/step - loss: 704085.1875 - val_loss: 704135.2500
    Epoch 1191/2000
    8/8 [==============================] - 0s 4ms/step - loss: 704211.9375 - val_loss: 702677.4375
    Epoch 1192/2000
    8/8 [==============================] - 0s 4ms/step - loss: 702765.1250 - val_loss: 700523.6875
    Epoch 1193/2000
    8/8 [==============================] - 0s 4ms/step - loss: 703061.2500 - val_loss: 700613.1250
    Epoch 1194/2000
    8/8 [==============================] - 0s 4ms/step - loss: 702195.1875 - val_loss: 701753.0625
    Epoch 1195/2000
    8/8 [==============================] - 0s 4ms/step - loss: 701365.8750 - val_loss: 701462.9375
    Epoch 1196/2000
    8/8 [==============================] - 0s 4ms/step - loss: 701906.9375 - val_loss: 700666.6875
    Epoch 1197/2000
    8/8 [==============================] - 0s 4ms/step - loss: 699354.3125 - val_loss: 700226.3750
    Epoch 1198/2000
    8/8 [==============================] - 0s 4ms/step - loss: 701191.3750 - val_loss: 701691.3125
    Epoch 1199/2000
    8/8 [==============================] - 0s 4ms/step - loss: 698301.7500 - val_loss: 702415.1250
    Epoch 1200/2000
    8/8 [==============================] - 0s 4ms/step - loss: 699257.6250 - val_loss: 703994.5000
    Epoch 1201/2000
    8/8 [==============================] - 0s 4ms/step - loss: 699748.1875 - val_loss: 699534.0000
    Epoch 1202/2000
    8/8 [==============================] - 0s 4ms/step - loss: 700017.0000 - val_loss: 702799.2500
    Epoch 1203/2000
    8/8 [==============================] - 0s 4ms/step - loss: 699754.7500 - val_loss: 698782.3125
    Epoch 1204/2000
    8/8 [==============================] - 0s 4ms/step - loss: 697001.1250 - val_loss: 698534.3750
    Epoch 1205/2000
    8/8 [==============================] - 0s 4ms/step - loss: 697635.1250 - val_loss: 698318.1875
    Epoch 1206/2000
    8/8 [==============================] - 0s 4ms/step - loss: 697945.8125 - val_loss: 699552.5000
    Epoch 1207/2000
    8/8 [==============================] - 0s 4ms/step - loss: 698807.3125 - val_loss: 698454.6250
    Epoch 1208/2000
    8/8 [==============================] - 0s 4ms/step - loss: 700317.1250 - val_loss: 701836.6250
    Epoch 1209/2000
    8/8 [==============================] - 0s 4ms/step - loss: 697382.8750 - val_loss: 699368.8750
    Epoch 1210/2000
    8/8 [==============================] - 0s 4ms/step - loss: 697949.2500 - val_loss: 697117.6875
    Epoch 1211/2000
    8/8 [==============================] - 0s 4ms/step - loss: 697508.4375 - val_loss: 700050.0000
    Epoch 1212/2000
    8/8 [==============================] - 0s 4ms/step - loss: 699253.6875 - val_loss: 697244.1875
    Epoch 1213/2000
    8/8 [==============================] - 0s 4ms/step - loss: 697797.7500 - val_loss: 698906.3750
    Epoch 1214/2000
    8/8 [==============================] - 0s 4ms/step - loss: 696107.0000 - val_loss: 696441.3750
    Epoch 1215/2000
    8/8 [==============================] - 0s 4ms/step - loss: 697546.4375 - val_loss: 698336.5625
    Epoch 1216/2000
    8/8 [==============================] - 0s 4ms/step - loss: 696228.6250 - val_loss: 694731.6250
    Epoch 1217/2000
    8/8 [==============================] - 0s 4ms/step - loss: 696415.1875 - val_loss: 694283.6250
    Epoch 1218/2000
    8/8 [==============================] - 0s 4ms/step - loss: 696248.5625 - val_loss: 695915.4375
    Epoch 1219/2000
    8/8 [==============================] - 0s 4ms/step - loss: 694339.1250 - val_loss: 695085.3125
    Epoch 1220/2000
    8/8 [==============================] - 0s 4ms/step - loss: 696833.9375 - val_loss: 692387.7500
    Epoch 1221/2000
    8/8 [==============================] - 0s 4ms/step - loss: 695115.5625 - val_loss: 695271.5000
    Epoch 1222/2000
    8/8 [==============================] - 0s 4ms/step - loss: 694826.9375 - val_loss: 695427.6250
    Epoch 1223/2000
    8/8 [==============================] - 0s 4ms/step - loss: 693892.4375 - val_loss: 693648.1250
    Epoch 1224/2000
    8/8 [==============================] - 0s 4ms/step - loss: 694260.4375 - val_loss: 693249.3750
    Epoch 1225/2000
    8/8 [==============================] - 0s 4ms/step - loss: 694075.3125 - val_loss: 693347.6875
    Epoch 1226/2000
    8/8 [==============================] - 0s 4ms/step - loss: 694671.1250 - val_loss: 693797.6250
    Epoch 1227/2000
    8/8 [==============================] - 0s 4ms/step - loss: 692445.3125 - val_loss: 692468.5625
    Epoch 1228/2000
    8/8 [==============================] - 0s 4ms/step - loss: 691839.5000 - val_loss: 695046.2500
    Epoch 1229/2000
    8/8 [==============================] - 0s 4ms/step - loss: 692325.3750 - val_loss: 689697.3750
    Epoch 1230/2000
    8/8 [==============================] - 0s 4ms/step - loss: 692770.9375 - val_loss: 691646.1875
    Epoch 1231/2000
    8/8 [==============================] - 0s 4ms/step - loss: 693447.3750 - val_loss: 694478.5000
    Epoch 1232/2000
    8/8 [==============================] - 0s 4ms/step - loss: 692118.3125 - val_loss: 691486.6875
    Epoch 1233/2000
    8/8 [==============================] - 0s 4ms/step - loss: 691124.1875 - val_loss: 692774.5625
    Epoch 1234/2000
    8/8 [==============================] - 0s 4ms/step - loss: 691591.9375 - val_loss: 691049.1250
    Epoch 1235/2000
    8/8 [==============================] - 0s 4ms/step - loss: 692105.6250 - val_loss: 691470.5000
    Epoch 1236/2000
    8/8 [==============================] - 0s 4ms/step - loss: 690856.1250 - val_loss: 690963.5000
    Epoch 1237/2000
    8/8 [==============================] - 0s 4ms/step - loss: 691068.1875 - val_loss: 693957.6250
    Epoch 1238/2000
    8/8 [==============================] - 0s 4ms/step - loss: 691307.5625 - val_loss: 688621.4375
    Epoch 1239/2000
    8/8 [==============================] - 0s 4ms/step - loss: 689872.0625 - val_loss: 691835.8125
    Epoch 1240/2000
    8/8 [==============================] - 0s 4ms/step - loss: 691493.3750 - val_loss: 689913.8750
    Epoch 1241/2000
    8/8 [==============================] - 0s 4ms/step - loss: 689385.0000 - val_loss: 690409.7500
    Epoch 1242/2000
    8/8 [==============================] - 0s 4ms/step - loss: 690157.0625 - val_loss: 692116.6250


    Epoch 1243/2000
    8/8 [==============================] - 0s 4ms/step - loss: 689746.5000 - val_loss: 690704.1875
    Epoch 1244/2000
    8/8 [==============================] - 0s 4ms/step - loss: 689588.7500 - val_loss: 687777.0000
    Epoch 1245/2000
    8/8 [==============================] - 0s 4ms/step - loss: 690047.6250 - val_loss: 690014.1250
    Epoch 1246/2000
    8/8 [==============================] - 0s 4ms/step - loss: 688976.4375 - val_loss: 687990.5625
    Epoch 1247/2000
    8/8 [==============================] - 0s 4ms/step - loss: 686939.3125 - val_loss: 688539.6875
    Epoch 1248/2000
    8/8 [==============================] - 0s 4ms/step - loss: 685912.6875 - val_loss: 684986.5625
    Epoch 1249/2000
    8/8 [==============================] - 0s 4ms/step - loss: 687881.1875 - val_loss: 688477.4375
    Epoch 1250/2000
    8/8 [==============================] - 0s 4ms/step - loss: 688180.2500 - val_loss: 688814.5625
    Epoch 1251/2000
    8/8 [==============================] - 0s 4ms/step - loss: 687167.5625 - val_loss: 687571.8125
    Epoch 1252/2000
    8/8 [==============================] - 0s 4ms/step - loss: 687351.4375 - val_loss: 686262.6875
    Epoch 1253/2000
    8/8 [==============================] - 0s 4ms/step - loss: 686188.9375 - val_loss: 687912.6250
    Epoch 1254/2000
    8/8 [==============================] - 0s 4ms/step - loss: 686580.1250 - val_loss: 689086.1250
    Epoch 1255/2000
    8/8 [==============================] - 0s 4ms/step - loss: 685840.0000 - val_loss: 688394.1875
    Epoch 1256/2000
    8/8 [==============================] - 0s 4ms/step - loss: 687973.8750 - val_loss: 687349.6250
    Epoch 1257/2000
    8/8 [==============================] - 0s 4ms/step - loss: 685589.1875 - val_loss: 684259.1250
    Epoch 1258/2000
    8/8 [==============================] - 0s 4ms/step - loss: 685605.3125 - val_loss: 686966.7500
    Epoch 1259/2000
    8/8 [==============================] - 0s 4ms/step - loss: 685273.6250 - val_loss: 682662.0625
    Epoch 1260/2000
    8/8 [==============================] - 0s 4ms/step - loss: 684689.9375 - val_loss: 684909.8750
    Epoch 1261/2000
    8/8 [==============================] - 0s 4ms/step - loss: 684375.3125 - val_loss: 683085.8125
    Epoch 1262/2000
    8/8 [==============================] - 0s 4ms/step - loss: 683526.0625 - val_loss: 683745.8750
    Epoch 1263/2000
    8/8 [==============================] - 0s 4ms/step - loss: 685615.0000 - val_loss: 683154.0000
    Epoch 1264/2000
    8/8 [==============================] - 0s 4ms/step - loss: 683883.2500 - val_loss: 684602.4375
    Epoch 1265/2000
    8/8 [==============================] - 0s 4ms/step - loss: 686279.4375 - val_loss: 680272.1250
    Epoch 1266/2000
    8/8 [==============================] - 0s 4ms/step - loss: 683291.3750 - val_loss: 686128.4375
    Epoch 1267/2000
    8/8 [==============================] - 0s 4ms/step - loss: 680832.5000 - val_loss: 684162.2500
    Epoch 1268/2000
    8/8 [==============================] - 0s 4ms/step - loss: 681881.2500 - val_loss: 682252.1250
    Epoch 1269/2000
    8/8 [==============================] - 0s 4ms/step - loss: 684071.1875 - val_loss: 684040.1875
    Epoch 1270/2000
    8/8 [==============================] - 0s 4ms/step - loss: 682356.9375 - val_loss: 680731.8750
    Epoch 1271/2000
    8/8 [==============================] - 0s 4ms/step - loss: 681783.2500 - val_loss: 683860.0000
    Epoch 1272/2000
    8/8 [==============================] - 0s 4ms/step - loss: 682715.8125 - val_loss: 684623.7500
    Epoch 1273/2000
    8/8 [==============================] - 0s 4ms/step - loss: 680165.5000 - val_loss: 682276.9375
    Epoch 1274/2000
    8/8 [==============================] - 0s 4ms/step - loss: 681341.0625 - val_loss: 680883.1250
    Epoch 1275/2000
    8/8 [==============================] - 0s 4ms/step - loss: 681557.0000 - val_loss: 680135.7500
    Epoch 1276/2000
    8/8 [==============================] - 0s 4ms/step - loss: 682149.0625 - val_loss: 681826.0625
    Epoch 1277/2000
    8/8 [==============================] - 0s 4ms/step - loss: 680267.8125 - val_loss: 680719.7500
    Epoch 1278/2000
    8/8 [==============================] - 0s 4ms/step - loss: 681107.6875 - val_loss: 682608.4375
    Epoch 1279/2000
    8/8 [==============================] - 0s 4ms/step - loss: 680006.3125 - val_loss: 680574.9375
    Epoch 1280/2000
    8/8 [==============================] - 0s 4ms/step - loss: 680250.1875 - val_loss: 679729.2500
    Epoch 1281/2000
    8/8 [==============================] - 0s 4ms/step - loss: 680220.6875 - val_loss: 679217.6875
    Epoch 1282/2000
    8/8 [==============================] - 0s 4ms/step - loss: 679438.5625 - val_loss: 678901.0000
    Epoch 1283/2000
    8/8 [==============================] - 0s 4ms/step - loss: 678334.5000 - val_loss: 680170.1875
    Epoch 1284/2000
    8/8 [==============================] - 0s 4ms/step - loss: 679727.5000 - val_loss: 678064.8125
    Epoch 1285/2000
    8/8 [==============================] - 0s 4ms/step - loss: 679616.5625 - val_loss: 679466.0000
    Epoch 1286/2000
    8/8 [==============================] - 0s 4ms/step - loss: 677677.3125 - val_loss: 681125.0625
    Epoch 1287/2000
    8/8 [==============================] - 0s 4ms/step - loss: 678081.6250 - val_loss: 679567.1250
    Epoch 1288/2000
    8/8 [==============================] - 0s 4ms/step - loss: 677950.3125 - val_loss: 679213.9375
    Epoch 1289/2000
    8/8 [==============================] - 0s 4ms/step - loss: 677857.5000 - val_loss: 678226.6250
    Epoch 1290/2000
    8/8 [==============================] - 0s 4ms/step - loss: 676427.1875 - val_loss: 676855.8750
    Epoch 1291/2000
    8/8 [==============================] - 0s 4ms/step - loss: 677548.8750 - val_loss: 680367.6250
    Epoch 1292/2000
    8/8 [==============================] - 0s 4ms/step - loss: 675201.0625 - val_loss: 677772.1875
    Epoch 1293/2000
    8/8 [==============================] - 0s 4ms/step - loss: 676532.0000 - val_loss: 678958.5625
    Epoch 1294/2000
    8/8 [==============================] - 0s 4ms/step - loss: 675712.6250 - val_loss: 674932.3125
    Epoch 1295/2000
    8/8 [==============================] - 0s 4ms/step - loss: 677489.5625 - val_loss: 676797.2500
    Epoch 1296/2000
    8/8 [==============================] - 0s 4ms/step - loss: 677742.1250 - val_loss: 675479.8125
    Epoch 1297/2000
    8/8 [==============================] - 0s 4ms/step - loss: 677354.8750 - val_loss: 676371.5625
    Epoch 1298/2000
    8/8 [==============================] - 0s 4ms/step - loss: 675823.3750 - val_loss: 674159.5625
    Epoch 1299/2000
    8/8 [==============================] - 0s 4ms/step - loss: 675676.3125 - val_loss: 675557.0625
    Epoch 1300/2000
    8/8 [==============================] - 0s 4ms/step - loss: 675389.4375 - val_loss: 673312.1875
    Epoch 1301/2000
    8/8 [==============================] - 0s 4ms/step - loss: 673695.0000 - val_loss: 674515.0625
    Epoch 1302/2000
    8/8 [==============================] - 0s 4ms/step - loss: 674808.7500 - val_loss: 674659.4375
    Epoch 1303/2000
    8/8 [==============================] - 0s 5ms/step - loss: 674442.8750 - val_loss: 673268.5625
    Epoch 1304/2000
    8/8 [==============================] - 0s 4ms/step - loss: 673726.8125 - val_loss: 674085.9375
    Epoch 1305/2000
    8/8 [==============================] - 0s 4ms/step - loss: 674621.0625 - val_loss: 673617.6250
    Epoch 1306/2000
    8/8 [==============================] - 0s 4ms/step - loss: 676162.1875 - val_loss: 674794.8750
    Epoch 1307/2000
    8/8 [==============================] - 0s 4ms/step - loss: 673793.8750 - val_loss: 675078.0000
    Epoch 1308/2000
    8/8 [==============================] - 0s 4ms/step - loss: 673394.1875 - val_loss: 671979.0625
    Epoch 1309/2000
    8/8 [==============================] - 0s 4ms/step - loss: 673249.2500 - val_loss: 671361.0625
    Epoch 1310/2000
    8/8 [==============================] - 0s 4ms/step - loss: 673191.8750 - val_loss: 673441.2500
    Epoch 1311/2000
    8/8 [==============================] - 0s 4ms/step - loss: 672766.9375 - val_loss: 671225.6250
    Epoch 1312/2000
    8/8 [==============================] - 0s 4ms/step - loss: 673505.4375 - val_loss: 671209.1875
    Epoch 1313/2000
    8/8 [==============================] - 0s 4ms/step - loss: 672216.6875 - val_loss: 672762.1875
    Epoch 1314/2000
    8/8 [==============================] - 0s 4ms/step - loss: 670905.5000 - val_loss: 669646.5000
    Epoch 1315/2000
    8/8 [==============================] - 0s 4ms/step - loss: 670332.5000 - val_loss: 671941.0000
    Epoch 1316/2000
    8/8 [==============================] - 0s 4ms/step - loss: 670376.3750 - val_loss: 670592.9375


    Epoch 1317/2000
    8/8 [==============================] - 0s 4ms/step - loss: 671153.1875 - val_loss: 674043.8125
    Epoch 1318/2000
    8/8 [==============================] - 0s 4ms/step - loss: 670714.6875 - val_loss: 671751.0625
    Epoch 1319/2000
    8/8 [==============================] - 0s 4ms/step - loss: 671698.2500 - val_loss: 671379.5000
    Epoch 1320/2000
    8/8 [==============================] - 0s 4ms/step - loss: 671379.1250 - val_loss: 667670.6250
    Epoch 1321/2000
    8/8 [==============================] - 0s 4ms/step - loss: 670396.3750 - val_loss: 668743.8125
    Epoch 1322/2000
    8/8 [==============================] - 0s 4ms/step - loss: 670336.8750 - val_loss: 671471.6250
    Epoch 1323/2000
    8/8 [==============================] - 0s 4ms/step - loss: 669397.6875 - val_loss: 667901.2500
    Epoch 1324/2000
    8/8 [==============================] - 0s 4ms/step - loss: 667884.2500 - val_loss: 671810.3125
    Epoch 1325/2000
    8/8 [==============================] - 0s 4ms/step - loss: 669272.6250 - val_loss: 668258.5000
    Epoch 1326/2000
    8/8 [==============================] - 0s 4ms/step - loss: 669620.0000 - val_loss: 670768.9375
    Epoch 1327/2000
    8/8 [==============================] - 0s 4ms/step - loss: 669918.8125 - val_loss: 666233.1250
    Epoch 1328/2000
    8/8 [==============================] - 0s 4ms/step - loss: 667989.1875 - val_loss: 667605.6250
    Epoch 1329/2000
    8/8 [==============================] - 0s 4ms/step - loss: 667174.6875 - val_loss: 666875.1875
    Epoch 1330/2000
    8/8 [==============================] - 0s 4ms/step - loss: 666378.4375 - val_loss: 668145.1250
    Epoch 1331/2000
    8/8 [==============================] - 0s 4ms/step - loss: 668243.0625 - val_loss: 668286.0000
    Epoch 1332/2000
    8/8 [==============================] - 0s 4ms/step - loss: 667041.7500 - val_loss: 668644.3125
    Epoch 1333/2000
    8/8 [==============================] - 0s 4ms/step - loss: 666630.3125 - val_loss: 664443.2500
    Epoch 1334/2000
    8/8 [==============================] - 0s 4ms/step - loss: 666297.1875 - val_loss: 666911.8750
    Epoch 1335/2000
    8/8 [==============================] - 0s 4ms/step - loss: 667656.8125 - val_loss: 666307.5000
    Epoch 1336/2000
    8/8 [==============================] - 0s 4ms/step - loss: 666250.2500 - val_loss: 666151.4375
    Epoch 1337/2000
    8/8 [==============================] - 0s 4ms/step - loss: 667215.3750 - val_loss: 667108.5625
    Epoch 1338/2000
    8/8 [==============================] - 0s 4ms/step - loss: 665202.0625 - val_loss: 666369.8750
    Epoch 1339/2000
    8/8 [==============================] - 0s 4ms/step - loss: 664345.9375 - val_loss: 665803.4375
    Epoch 1340/2000
    8/8 [==============================] - 0s 4ms/step - loss: 664966.7500 - val_loss: 665291.8125
    Epoch 1341/2000
    8/8 [==============================] - 0s 4ms/step - loss: 664580.6250 - val_loss: 665843.4375
    Epoch 1342/2000
    8/8 [==============================] - 0s 4ms/step - loss: 665131.5000 - val_loss: 663164.3750
    Epoch 1343/2000
    8/8 [==============================] - 0s 4ms/step - loss: 664557.8750 - val_loss: 664957.3125
    Epoch 1344/2000
    8/8 [==============================] - 0s 4ms/step - loss: 664367.0625 - val_loss: 667135.5625
    Epoch 1345/2000
    8/8 [==============================] - 0s 4ms/step - loss: 664174.6250 - val_loss: 663235.0625
    Epoch 1346/2000
    8/8 [==============================] - 0s 4ms/step - loss: 663956.0625 - val_loss: 665287.9375
    Epoch 1347/2000
    8/8 [==============================] - 0s 4ms/step - loss: 664856.2500 - val_loss: 660947.1875
    Epoch 1348/2000
    8/8 [==============================] - 0s 4ms/step - loss: 664165.6875 - val_loss: 664824.3125
    Epoch 1349/2000
    8/8 [==============================] - 0s 4ms/step - loss: 663358.6250 - val_loss: 662755.9375
    Epoch 1350/2000
    8/8 [==============================] - 0s 4ms/step - loss: 663096.6875 - val_loss: 662729.5625
    Epoch 1351/2000
    8/8 [==============================] - 0s 4ms/step - loss: 663576.3125 - val_loss: 660111.1875
    Epoch 1352/2000
    8/8 [==============================] - 0s 4ms/step - loss: 661563.2500 - val_loss: 663751.0000
    Epoch 1353/2000
    8/8 [==============================] - 0s 4ms/step - loss: 660184.9375 - val_loss: 662746.0000
    Epoch 1354/2000
    8/8 [==============================] - 0s 4ms/step - loss: 661786.1875 - val_loss: 663444.8125
    Epoch 1355/2000
    8/8 [==============================] - 0s 4ms/step - loss: 662015.0000 - val_loss: 662864.1875
    Epoch 1356/2000
    8/8 [==============================] - 0s 4ms/step - loss: 661965.3750 - val_loss: 661771.2500
    Epoch 1357/2000
    8/8 [==============================] - 0s 4ms/step - loss: 661555.5000 - val_loss: 659141.2500
    Epoch 1358/2000
    8/8 [==============================] - 0s 4ms/step - loss: 661896.4375 - val_loss: 661108.9375
    Epoch 1359/2000
    8/8 [==============================] - 0s 4ms/step - loss: 662775.6875 - val_loss: 661787.0000
    Epoch 1360/2000
    8/8 [==============================] - 0s 4ms/step - loss: 660423.6875 - val_loss: 660655.5625
    Epoch 1361/2000
    8/8 [==============================] - 0s 4ms/step - loss: 659218.0625 - val_loss: 660989.0625
    Epoch 1362/2000
    8/8 [==============================] - 0s 4ms/step - loss: 660585.6875 - val_loss: 661458.3750
    Epoch 1363/2000
    8/8 [==============================] - 0s 4ms/step - loss: 660643.8125 - val_loss: 661435.3125
    Epoch 1364/2000
    8/8 [==============================] - 0s 4ms/step - loss: 658769.1875 - val_loss: 660975.5625
    Epoch 1365/2000
    8/8 [==============================] - 0s 4ms/step - loss: 658775.8750 - val_loss: 658542.1875
    Epoch 1366/2000
    8/8 [==============================] - 0s 4ms/step - loss: 659958.7500 - val_loss: 658078.4375
    Epoch 1367/2000
    8/8 [==============================] - 0s 4ms/step - loss: 659577.9375 - val_loss: 658535.4375
    Epoch 1368/2000
    8/8 [==============================] - 0s 4ms/step - loss: 659027.8125 - val_loss: 659418.4375
    Epoch 1369/2000
    8/8 [==============================] - 0s 4ms/step - loss: 657289.3750 - val_loss: 656901.0000
    Epoch 1370/2000
    8/8 [==============================] - 0s 4ms/step - loss: 657131.8125 - val_loss: 660228.5625
    Epoch 1371/2000
    8/8 [==============================] - 0s 4ms/step - loss: 658356.1875 - val_loss: 655809.2500
    Epoch 1372/2000
    8/8 [==============================] - 0s 4ms/step - loss: 658544.5625 - val_loss: 659152.0000
    Epoch 1373/2000
    8/8 [==============================] - 0s 4ms/step - loss: 657413.3125 - val_loss: 661914.3750
    Epoch 1374/2000
    8/8 [==============================] - 0s 4ms/step - loss: 656225.1250 - val_loss: 658740.5000
    Epoch 1375/2000
    8/8 [==============================] - 0s 4ms/step - loss: 655872.5000 - val_loss: 657119.8125
    Epoch 1376/2000
    8/8 [==============================] - 0s 4ms/step - loss: 657794.8750 - val_loss: 656983.8125
    Epoch 1377/2000
    8/8 [==============================] - 0s 4ms/step - loss: 657019.3125 - val_loss: 654422.6250
    Epoch 1378/2000
    8/8 [==============================] - 0s 4ms/step - loss: 656373.1875 - val_loss: 656907.6250
    Epoch 1379/2000
    8/8 [==============================] - 0s 4ms/step - loss: 657854.9375 - val_loss: 657176.5625
    Epoch 1380/2000
    8/8 [==============================] - 0s 4ms/step - loss: 657528.2500 - val_loss: 655195.7500
    Epoch 1381/2000
    8/8 [==============================] - 0s 4ms/step - loss: 655408.8125 - val_loss: 655376.1875
    Epoch 1382/2000
    8/8 [==============================] - 0s 4ms/step - loss: 656919.4375 - val_loss: 656420.5625
    Epoch 1383/2000
    8/8 [==============================] - 0s 4ms/step - loss: 655852.4375 - val_loss: 655033.2500
    Epoch 1384/2000
    8/8 [==============================] - 0s 4ms/step - loss: 655512.1250 - val_loss: 656203.4375
    Epoch 1385/2000
    8/8 [==============================] - 0s 4ms/step - loss: 655232.8750 - val_loss: 656427.4375
    Epoch 1386/2000
    8/8 [==============================] - 0s 4ms/step - loss: 654640.8125 - val_loss: 655180.3750
    Epoch 1387/2000
    8/8 [==============================] - 0s 4ms/step - loss: 654626.5625 - val_loss: 653948.8750
    Epoch 1388/2000
    8/8 [==============================] - 0s 4ms/step - loss: 653715.9375 - val_loss: 653075.9375
    Epoch 1389/2000
    8/8 [==============================] - 0s 4ms/step - loss: 653894.5625 - val_loss: 654329.1250
    Epoch 1390/2000
    8/8 [==============================] - 0s 4ms/step - loss: 652989.3125 - val_loss: 655594.0000


    Epoch 1391/2000
    8/8 [==============================] - 0s 4ms/step - loss: 653367.8750 - val_loss: 652151.5625
    Epoch 1392/2000
    8/8 [==============================] - 0s 4ms/step - loss: 652317.4375 - val_loss: 652237.5625
    Epoch 1393/2000
    8/8 [==============================] - 0s 4ms/step - loss: 655852.5000 - val_loss: 652950.1875
    Epoch 1394/2000
    8/8 [==============================] - 0s 4ms/step - loss: 651222.3125 - val_loss: 652913.0000
    Epoch 1395/2000
    8/8 [==============================] - 0s 4ms/step - loss: 652086.5000 - val_loss: 653962.7500
    Epoch 1396/2000
    8/8 [==============================] - 0s 4ms/step - loss: 651383.1875 - val_loss: 652008.8750
    Epoch 1397/2000
    8/8 [==============================] - 0s 4ms/step - loss: 651682.0625 - val_loss: 652557.8750
    Epoch 1398/2000
    8/8 [==============================] - 0s 4ms/step - loss: 651591.9375 - val_loss: 650373.1875
    Epoch 1399/2000
    8/8 [==============================] - 0s 4ms/step - loss: 651791.3750 - val_loss: 651272.1875
    Epoch 1400/2000
    8/8 [==============================] - 0s 4ms/step - loss: 652020.7500 - val_loss: 651294.5000
    Epoch 1401/2000
    8/8 [==============================] - 0s 4ms/step - loss: 652324.5625 - val_loss: 648872.6875
    Epoch 1402/2000
    8/8 [==============================] - 0s 4ms/step - loss: 650799.6875 - val_loss: 650611.0625
    Epoch 1403/2000
    8/8 [==============================] - 0s 4ms/step - loss: 649801.1875 - val_loss: 651797.1250
    Epoch 1404/2000
    8/8 [==============================] - 0s 4ms/step - loss: 650644.1875 - val_loss: 651346.0625
    Epoch 1405/2000
    8/8 [==============================] - 0s 4ms/step - loss: 651136.3125 - val_loss: 651342.3750
    Epoch 1406/2000
    8/8 [==============================] - 0s 4ms/step - loss: 651045.2500 - val_loss: 646800.1875
    Epoch 1407/2000
    8/8 [==============================] - 0s 4ms/step - loss: 648927.6875 - val_loss: 649067.2500
    Epoch 1408/2000
    8/8 [==============================] - 0s 4ms/step - loss: 649543.1250 - val_loss: 648181.3125
    Epoch 1409/2000
    8/8 [==============================] - 0s 5ms/step - loss: 650337.5000 - val_loss: 649284.3125
    Epoch 1410/2000
    8/8 [==============================] - 0s 4ms/step - loss: 650075.3750 - val_loss: 649599.6250
    Epoch 1411/2000
    8/8 [==============================] - 0s 4ms/step - loss: 649723.1875 - val_loss: 650818.5625
    Epoch 1412/2000
    8/8 [==============================] - 0s 4ms/step - loss: 648096.5625 - val_loss: 648054.1250
    Epoch 1413/2000
    8/8 [==============================] - 0s 4ms/step - loss: 648259.2500 - val_loss: 648908.9375
    Epoch 1414/2000
    8/8 [==============================] - 0s 4ms/step - loss: 646315.2500 - val_loss: 646218.3750
    Epoch 1415/2000
    8/8 [==============================] - 0s 4ms/step - loss: 648226.5000 - val_loss: 647990.9375
    Epoch 1416/2000
    8/8 [==============================] - 0s 4ms/step - loss: 648195.9375 - val_loss: 647611.8750
    Epoch 1417/2000
    8/8 [==============================] - 0s 4ms/step - loss: 649176.6250 - val_loss: 649091.6250
    Epoch 1418/2000
    8/8 [==============================] - 0s 4ms/step - loss: 648466.7500 - val_loss: 645895.8125
    Epoch 1419/2000
    8/8 [==============================] - 0s 4ms/step - loss: 645268.7500 - val_loss: 643765.0000
    Epoch 1420/2000
    8/8 [==============================] - 0s 4ms/step - loss: 646186.6250 - val_loss: 645084.7500
    Epoch 1421/2000
    8/8 [==============================] - 0s 4ms/step - loss: 646823.0625 - val_loss: 645621.1250
    Epoch 1422/2000
    8/8 [==============================] - 0s 4ms/step - loss: 643918.2500 - val_loss: 644551.8125
    Epoch 1423/2000
    8/8 [==============================] - 0s 4ms/step - loss: 645571.3125 - val_loss: 645630.7500
    Epoch 1424/2000
    8/8 [==============================] - 0s 4ms/step - loss: 645719.5625 - val_loss: 646153.1875
    Epoch 1425/2000
    8/8 [==============================] - 0s 4ms/step - loss: 645837.4375 - val_loss: 645156.3125
    Epoch 1426/2000
    8/8 [==============================] - 0s 4ms/step - loss: 643681.3750 - val_loss: 642158.0000
    Epoch 1427/2000
    8/8 [==============================] - 0s 4ms/step - loss: 646093.8750 - val_loss: 646525.1875
    Epoch 1428/2000
    8/8 [==============================] - 0s 4ms/step - loss: 646863.1250 - val_loss: 645254.4375
    Epoch 1429/2000
    8/8 [==============================] - 0s 4ms/step - loss: 644600.9375 - val_loss: 645978.0000
    Epoch 1430/2000
    8/8 [==============================] - 0s 4ms/step - loss: 645120.3125 - val_loss: 644572.9375
    Epoch 1431/2000
    8/8 [==============================] - 0s 4ms/step - loss: 643702.8750 - val_loss: 643213.8750
    Epoch 1432/2000
    8/8 [==============================] - 0s 4ms/step - loss: 644886.5000 - val_loss: 644827.7500
    Epoch 1433/2000
    8/8 [==============================] - 0s 4ms/step - loss: 643905.5625 - val_loss: 648036.9375
    Epoch 1434/2000
    8/8 [==============================] - 0s 4ms/step - loss: 643792.1875 - val_loss: 641920.9375
    Epoch 1435/2000
    8/8 [==============================] - 0s 4ms/step - loss: 643399.1250 - val_loss: 644525.5000
    Epoch 1436/2000
    8/8 [==============================] - 0s 4ms/step - loss: 643508.7500 - val_loss: 644427.2500
    Epoch 1437/2000
    8/8 [==============================] - 0s 4ms/step - loss: 642944.3125 - val_loss: 643904.8125
    Epoch 1438/2000
    8/8 [==============================] - 0s 4ms/step - loss: 645149.6250 - val_loss: 641281.2500
    Epoch 1439/2000
    8/8 [==============================] - 0s 4ms/step - loss: 644794.5000 - val_loss: 643747.6250
    Epoch 1440/2000
    8/8 [==============================] - 0s 4ms/step - loss: 642158.9375 - val_loss: 643162.0000
    Epoch 1441/2000
    8/8 [==============================] - 0s 4ms/step - loss: 642786.3750 - val_loss: 642206.8750
    Epoch 1442/2000
    8/8 [==============================] - 0s 4ms/step - loss: 641885.5625 - val_loss: 638618.7500
    Epoch 1443/2000
    8/8 [==============================] - 0s 4ms/step - loss: 641421.5000 - val_loss: 639761.1250
    Epoch 1444/2000
    8/8 [==============================] - 0s 4ms/step - loss: 642381.8750 - val_loss: 641023.5000
    Epoch 1445/2000
    8/8 [==============================] - 0s 4ms/step - loss: 641620.8125 - val_loss: 641662.3125
    Epoch 1446/2000
    8/8 [==============================] - 0s 4ms/step - loss: 641010.6875 - val_loss: 643092.1875
    Epoch 1447/2000
    8/8 [==============================] - 0s 4ms/step - loss: 640470.8750 - val_loss: 643977.2500
    Epoch 1448/2000
    8/8 [==============================] - 0s 4ms/step - loss: 640354.5000 - val_loss: 641130.8125
    Epoch 1449/2000
    8/8 [==============================] - 0s 4ms/step - loss: 641033.6875 - val_loss: 640584.3750
    Epoch 1450/2000
    8/8 [==============================] - 0s 4ms/step - loss: 638377.6250 - val_loss: 642705.8125
    Epoch 1451/2000
    8/8 [==============================] - 0s 4ms/step - loss: 641198.3750 - val_loss: 640159.6250
    Epoch 1452/2000
    8/8 [==============================] - 0s 4ms/step - loss: 639473.1875 - val_loss: 639533.0000
    Epoch 1453/2000
    8/8 [==============================] - 0s 4ms/step - loss: 639207.5625 - val_loss: 640302.5625
    Epoch 1454/2000
    8/8 [==============================] - 0s 4ms/step - loss: 639804.6875 - val_loss: 637721.3125
    Epoch 1455/2000
    8/8 [==============================] - 0s 4ms/step - loss: 640179.1250 - val_loss: 637593.2500
    Epoch 1456/2000
    8/8 [==============================] - 0s 4ms/step - loss: 639820.8750 - val_loss: 637615.5000
    Epoch 1457/2000
    8/8 [==============================] - 0s 4ms/step - loss: 639456.4375 - val_loss: 635954.3750
    Epoch 1458/2000
    8/8 [==============================] - 0s 4ms/step - loss: 638128.8125 - val_loss: 638778.3750
    Epoch 1459/2000
    8/8 [==============================] - 0s 4ms/step - loss: 635931.7500 - val_loss: 638184.6250
    Epoch 1460/2000
    8/8 [==============================] - 0s 4ms/step - loss: 637833.8125 - val_loss: 634960.5625
    Epoch 1461/2000
    8/8 [==============================] - 0s 4ms/step - loss: 638372.5625 - val_loss: 638287.2500
    Epoch 1462/2000
    8/8 [==============================] - 0s 4ms/step - loss: 638620.6875 - val_loss: 638178.5625
    Epoch 1463/2000
    8/8 [==============================] - 0s 4ms/step - loss: 637266.8750 - val_loss: 636874.7500
    Epoch 1464/2000
    8/8 [==============================] - 0s 4ms/step - loss: 636826.8750 - val_loss: 634364.4375


    Epoch 1465/2000
    8/8 [==============================] - 0s 4ms/step - loss: 637386.0000 - val_loss: 635547.7500
    Epoch 1466/2000
    8/8 [==============================] - 0s 4ms/step - loss: 637163.5000 - val_loss: 640048.0625
    Epoch 1467/2000
    8/8 [==============================] - 0s 4ms/step - loss: 635369.7500 - val_loss: 638938.7500
    Epoch 1468/2000
    8/8 [==============================] - 0s 4ms/step - loss: 637350.3750 - val_loss: 636556.9375
    Epoch 1469/2000
    8/8 [==============================] - 0s 4ms/step - loss: 636892.5625 - val_loss: 636307.9375
    Epoch 1470/2000
    8/8 [==============================] - 0s 4ms/step - loss: 635735.4375 - val_loss: 637155.8125
    Epoch 1471/2000
    8/8 [==============================] - 0s 4ms/step - loss: 636653.5625 - val_loss: 633044.7500
    Epoch 1472/2000
    8/8 [==============================] - 0s 4ms/step - loss: 635520.8125 - val_loss: 637610.1250
    Epoch 1473/2000
    8/8 [==============================] - 0s 4ms/step - loss: 635624.8750 - val_loss: 634857.6250
    Epoch 1474/2000
    8/8 [==============================] - 0s 4ms/step - loss: 635380.7500 - val_loss: 631739.1875
    Epoch 1475/2000
    8/8 [==============================] - 0s 4ms/step - loss: 636060.2500 - val_loss: 633165.3125
    Epoch 1476/2000
    8/8 [==============================] - 0s 4ms/step - loss: 633886.5000 - val_loss: 632496.9375
    Epoch 1477/2000
    8/8 [==============================] - 0s 4ms/step - loss: 633110.6875 - val_loss: 635642.2500
    Epoch 1478/2000
    8/8 [==============================] - 0s 4ms/step - loss: 634252.3750 - val_loss: 633427.4375
    Epoch 1479/2000
    8/8 [==============================] - 0s 4ms/step - loss: 635557.7500 - val_loss: 632852.3750
    Epoch 1480/2000
    8/8 [==============================] - 0s 4ms/step - loss: 633665.4375 - val_loss: 633642.1875
    Epoch 1481/2000
    8/8 [==============================] - 0s 4ms/step - loss: 633612.7500 - val_loss: 631418.7500
    Epoch 1482/2000
    8/8 [==============================] - 0s 4ms/step - loss: 633020.4375 - val_loss: 633505.4375
    Epoch 1483/2000
    8/8 [==============================] - 0s 4ms/step - loss: 632086.5000 - val_loss: 632565.8750
    Epoch 1484/2000
    8/8 [==============================] - 0s 4ms/step - loss: 632912.8750 - val_loss: 632465.1875
    Epoch 1485/2000
    8/8 [==============================] - 0s 4ms/step - loss: 634521.5000 - val_loss: 630859.5000
    Epoch 1486/2000
    8/8 [==============================] - 0s 4ms/step - loss: 631732.1250 - val_loss: 631943.3750
    Epoch 1487/2000
    8/8 [==============================] - 0s 4ms/step - loss: 632496.3125 - val_loss: 632913.6250
    Epoch 1488/2000
    8/8 [==============================] - 0s 4ms/step - loss: 631604.6875 - val_loss: 630110.1250
    Epoch 1489/2000
    8/8 [==============================] - 0s 4ms/step - loss: 631371.0000 - val_loss: 628929.8125
    Epoch 1490/2000
    8/8 [==============================] - 0s 4ms/step - loss: 631290.1875 - val_loss: 632574.6250
    Epoch 1491/2000
    8/8 [==============================] - 0s 4ms/step - loss: 631580.8750 - val_loss: 630152.8125
    Epoch 1492/2000
    8/8 [==============================] - 0s 4ms/step - loss: 630929.1875 - val_loss: 632836.5000
    Epoch 1493/2000
    8/8 [==============================] - 0s 4ms/step - loss: 632030.9375 - val_loss: 630887.4375
    Epoch 1494/2000
    8/8 [==============================] - 0s 4ms/step - loss: 629464.6250 - val_loss: 631683.1875
    Epoch 1495/2000
    8/8 [==============================] - 0s 4ms/step - loss: 631049.8750 - val_loss: 631883.0000
    Epoch 1496/2000
    8/8 [==============================] - 0s 4ms/step - loss: 630172.3750 - val_loss: 630590.5625
    Epoch 1497/2000
    8/8 [==============================] - 0s 4ms/step - loss: 630182.8125 - val_loss: 630698.8125
    Epoch 1498/2000
    8/8 [==============================] - 0s 4ms/step - loss: 628518.3750 - val_loss: 629737.8125
    Epoch 1499/2000
    8/8 [==============================] - 0s 4ms/step - loss: 629642.5000 - val_loss: 628289.2500
    Epoch 1500/2000
    8/8 [==============================] - 0s 4ms/step - loss: 630576.1875 - val_loss: 627326.6875
    Epoch 1501/2000
    8/8 [==============================] - 0s 4ms/step - loss: 628819.2500 - val_loss: 624870.4375
    Epoch 1502/2000
    8/8 [==============================] - 0s 4ms/step - loss: 628114.4375 - val_loss: 625622.7500
    Epoch 1503/2000
    8/8 [==============================] - 0s 4ms/step - loss: 628554.6875 - val_loss: 629837.8125
    Epoch 1504/2000
    8/8 [==============================] - 0s 4ms/step - loss: 628028.3750 - val_loss: 628738.5000
    Epoch 1505/2000
    8/8 [==============================] - 0s 4ms/step - loss: 628103.0625 - val_loss: 627527.0625
    Epoch 1506/2000
    8/8 [==============================] - 0s 4ms/step - loss: 628376.2500 - val_loss: 626451.0000
    Epoch 1507/2000
    8/8 [==============================] - 0s 4ms/step - loss: 629307.8750 - val_loss: 627906.5000
    Epoch 1508/2000
    8/8 [==============================] - 0s 4ms/step - loss: 626344.3750 - val_loss: 624993.6250
    Epoch 1509/2000
    8/8 [==============================] - 0s 4ms/step - loss: 626572.9375 - val_loss: 628121.3750
    Epoch 1510/2000
    8/8 [==============================] - 0s 4ms/step - loss: 627043.6250 - val_loss: 629450.0625
    Epoch 1511/2000
    8/8 [==============================] - 0s 4ms/step - loss: 626835.0000 - val_loss: 624902.1875
    Epoch 1512/2000
    8/8 [==============================] - 0s 5ms/step - loss: 626850.8750 - val_loss: 624464.6875
    Epoch 1513/2000
    8/8 [==============================] - 0s 4ms/step - loss: 626382.8125 - val_loss: 627092.8750
    Epoch 1514/2000
    8/8 [==============================] - 0s 4ms/step - loss: 626790.6875 - val_loss: 625072.5000
    Epoch 1515/2000
    8/8 [==============================] - 0s 4ms/step - loss: 627571.2500 - val_loss: 627098.3750
    Epoch 1516/2000
    8/8 [==============================] - 0s 4ms/step - loss: 625727.3125 - val_loss: 624592.0625
    Epoch 1517/2000
    8/8 [==============================] - 0s 4ms/step - loss: 625235.3125 - val_loss: 625507.5000
    Epoch 1518/2000
    8/8 [==============================] - 0s 4ms/step - loss: 625058.1250 - val_loss: 626204.6875
    Epoch 1519/2000
    8/8 [==============================] - 0s 4ms/step - loss: 625097.1250 - val_loss: 627572.8750
    Epoch 1520/2000
    8/8 [==============================] - 0s 4ms/step - loss: 625357.3125 - val_loss: 622600.7500
    Epoch 1521/2000
    8/8 [==============================] - 0s 4ms/step - loss: 624208.9375 - val_loss: 627408.1875
    Epoch 1522/2000
    8/8 [==============================] - 0s 4ms/step - loss: 624961.7500 - val_loss: 623838.1250
    Epoch 1523/2000
    8/8 [==============================] - 0s 4ms/step - loss: 623440.5000 - val_loss: 624736.8750
    Epoch 1524/2000
    8/8 [==============================] - 0s 4ms/step - loss: 624368.8125 - val_loss: 622968.5000
    Epoch 1525/2000
    8/8 [==============================] - 0s 4ms/step - loss: 624565.3125 - val_loss: 623379.0000
    Epoch 1526/2000
    8/8 [==============================] - 0s 4ms/step - loss: 622534.2500 - val_loss: 622353.3750
    Epoch 1527/2000
    8/8 [==============================] - 0s 4ms/step - loss: 622522.4375 - val_loss: 623716.0000
    Epoch 1528/2000
    8/8 [==============================] - 0s 4ms/step - loss: 623912.0625 - val_loss: 619885.7500
    Epoch 1529/2000
    8/8 [==============================] - 0s 4ms/step - loss: 624569.3125 - val_loss: 624955.2500
    Epoch 1530/2000
    8/8 [==============================] - 0s 4ms/step - loss: 623559.2500 - val_loss: 626258.6875
    Epoch 1531/2000
    8/8 [==============================] - 0s 4ms/step - loss: 620449.7500 - val_loss: 621770.0625
    Epoch 1532/2000
    8/8 [==============================] - 0s 4ms/step - loss: 622968.4375 - val_loss: 622459.0000
    Epoch 1533/2000
    8/8 [==============================] - 0s 4ms/step - loss: 620936.7500 - val_loss: 622796.8750
    Epoch 1534/2000
    8/8 [==============================] - 0s 4ms/step - loss: 621913.7500 - val_loss: 622182.2500
    Epoch 1535/2000
    8/8 [==============================] - 0s 4ms/step - loss: 621475.7500 - val_loss: 623139.4375
    Epoch 1536/2000
    8/8 [==============================] - 0s 4ms/step - loss: 622049.6875 - val_loss: 621136.1875
    Epoch 1537/2000
    8/8 [==============================] - 0s 4ms/step - loss: 620664.1250 - val_loss: 621969.2500
    Epoch 1538/2000
    8/8 [==============================] - 0s 4ms/step - loss: 620743.3750 - val_loss: 620931.8750


    Epoch 1539/2000
    8/8 [==============================] - 0s 4ms/step - loss: 619249.5625 - val_loss: 623471.0625
    Epoch 1540/2000
    8/8 [==============================] - 0s 4ms/step - loss: 622468.1875 - val_loss: 620635.3125
    Epoch 1541/2000
    8/8 [==============================] - 0s 4ms/step - loss: 621292.8750 - val_loss: 621386.8750
    Epoch 1542/2000
    8/8 [==============================] - 0s 4ms/step - loss: 620376.5625 - val_loss: 620108.5625
    Epoch 1543/2000
    8/8 [==============================] - 0s 4ms/step - loss: 619203.4375 - val_loss: 618885.4375
    Epoch 1544/2000
    8/8 [==============================] - 0s 4ms/step - loss: 621108.6250 - val_loss: 620712.3125
    Epoch 1545/2000
    8/8 [==============================] - 0s 4ms/step - loss: 619300.0625 - val_loss: 620970.3125
    Epoch 1546/2000
    8/8 [==============================] - 0s 4ms/step - loss: 619666.3750 - val_loss: 618763.0000
    Epoch 1547/2000
    8/8 [==============================] - 0s 4ms/step - loss: 618669.6250 - val_loss: 618335.3750
    Epoch 1548/2000
    8/8 [==============================] - 0s 4ms/step - loss: 620346.4375 - val_loss: 618816.0625
    Epoch 1549/2000
    8/8 [==============================] - 0s 4ms/step - loss: 618977.1250 - val_loss: 617064.3125
    Epoch 1550/2000
    8/8 [==============================] - 0s 4ms/step - loss: 618363.7500 - val_loss: 618027.4375
    Epoch 1551/2000
    8/8 [==============================] - 0s 4ms/step - loss: 619461.6250 - val_loss: 617690.4375
    Epoch 1552/2000
    8/8 [==============================] - 0s 4ms/step - loss: 617996.6875 - val_loss: 618486.7500
    Epoch 1553/2000
    8/8 [==============================] - 0s 4ms/step - loss: 618341.6250 - val_loss: 614837.8125
    Epoch 1554/2000
    8/8 [==============================] - 0s 4ms/step - loss: 617679.5625 - val_loss: 617971.3125
    Epoch 1555/2000
    8/8 [==============================] - 0s 4ms/step - loss: 618337.1250 - val_loss: 615926.5000
    Epoch 1556/2000
    8/8 [==============================] - 0s 4ms/step - loss: 617023.5625 - val_loss: 616904.6250
    Epoch 1557/2000
    8/8 [==============================] - 0s 4ms/step - loss: 618474.8125 - val_loss: 615817.7500
    Epoch 1558/2000
    8/8 [==============================] - 0s 4ms/step - loss: 617322.4375 - val_loss: 616646.3750
    Epoch 1559/2000
    8/8 [==============================] - 0s 4ms/step - loss: 616462.0000 - val_loss: 617271.5000
    Epoch 1560/2000
    8/8 [==============================] - 0s 4ms/step - loss: 617142.2500 - val_loss: 618030.4375
    Epoch 1561/2000
    8/8 [==============================] - 0s 4ms/step - loss: 616810.4375 - val_loss: 616761.6875
    Epoch 1562/2000
    8/8 [==============================] - 0s 4ms/step - loss: 616363.5625 - val_loss: 614809.8750
    Epoch 1563/2000
    8/8 [==============================] - 0s 4ms/step - loss: 617544.6250 - val_loss: 617107.8125
    Epoch 1564/2000
    8/8 [==============================] - 0s 4ms/step - loss: 615857.5000 - val_loss: 613811.5000
    Epoch 1565/2000
    8/8 [==============================] - 0s 4ms/step - loss: 616836.6875 - val_loss: 616233.3125
    Epoch 1566/2000
    8/8 [==============================] - 0s 4ms/step - loss: 615750.6250 - val_loss: 616471.3125
    Epoch 1567/2000
    8/8 [==============================] - 0s 4ms/step - loss: 615106.9375 - val_loss: 613087.7500
    Epoch 1568/2000
    8/8 [==============================] - 0s 4ms/step - loss: 615457.5625 - val_loss: 616095.1875
    Epoch 1569/2000
    8/8 [==============================] - 0s 4ms/step - loss: 613695.6875 - val_loss: 617111.9375
    Epoch 1570/2000
    8/8 [==============================] - 0s 4ms/step - loss: 614959.6875 - val_loss: 614500.9375
    Epoch 1571/2000
    8/8 [==============================] - 0s 4ms/step - loss: 614507.4375 - val_loss: 612790.3750
    Epoch 1572/2000
    8/8 [==============================] - 0s 4ms/step - loss: 614611.6250 - val_loss: 613887.3750
    Epoch 1573/2000
    8/8 [==============================] - 0s 4ms/step - loss: 613321.4375 - val_loss: 612055.6875
    Epoch 1574/2000
    8/8 [==============================] - 0s 4ms/step - loss: 612692.6250 - val_loss: 610562.3750
    Epoch 1575/2000
    8/8 [==============================] - 0s 4ms/step - loss: 612609.0625 - val_loss: 614319.0625
    Epoch 1576/2000
    8/8 [==============================] - 0s 4ms/step - loss: 613756.9375 - val_loss: 611927.4375
    Epoch 1577/2000
    8/8 [==============================] - 0s 4ms/step - loss: 611928.7500 - val_loss: 610132.3750
    Epoch 1578/2000
    8/8 [==============================] - 0s 4ms/step - loss: 611200.9375 - val_loss: 614039.9375
    Epoch 1579/2000
    8/8 [==============================] - 0s 4ms/step - loss: 614217.6250 - val_loss: 610696.1875
    Epoch 1580/2000
    8/8 [==============================] - 0s 4ms/step - loss: 611719.8750 - val_loss: 610971.0000
    Epoch 1581/2000
    8/8 [==============================] - 0s 4ms/step - loss: 612773.3125 - val_loss: 612193.5000
    Epoch 1582/2000
    8/8 [==============================] - 0s 4ms/step - loss: 613408.1875 - val_loss: 611602.0000
    Epoch 1583/2000
    8/8 [==============================] - 0s 4ms/step - loss: 611797.6875 - val_loss: 613255.3750
    Epoch 1584/2000
    8/8 [==============================] - 0s 4ms/step - loss: 611012.6250 - val_loss: 610924.4375
    Epoch 1585/2000
    8/8 [==============================] - 0s 4ms/step - loss: 611665.2500 - val_loss: 612358.6875
    Epoch 1586/2000
    8/8 [==============================] - 0s 4ms/step - loss: 612003.9375 - val_loss: 610789.8125
    Epoch 1587/2000
    8/8 [==============================] - 0s 4ms/step - loss: 612347.8125 - val_loss: 609621.6250
    Epoch 1588/2000
    8/8 [==============================] - 0s 4ms/step - loss: 611254.0000 - val_loss: 609659.1875
    Epoch 1589/2000
    8/8 [==============================] - 0s 4ms/step - loss: 610469.5000 - val_loss: 609107.4375
    Epoch 1590/2000
    8/8 [==============================] - 0s 4ms/step - loss: 612053.7500 - val_loss: 611611.3750
    Epoch 1591/2000
    8/8 [==============================] - 0s 4ms/step - loss: 610331.0625 - val_loss: 609584.3125
    Epoch 1592/2000
    8/8 [==============================] - 0s 4ms/step - loss: 609844.2500 - val_loss: 610203.0000
    Epoch 1593/2000
    8/8 [==============================] - 0s 4ms/step - loss: 610043.5625 - val_loss: 611037.1875
    Epoch 1594/2000
    8/8 [==============================] - 0s 4ms/step - loss: 609132.0625 - val_loss: 609593.0625
    Epoch 1595/2000
    8/8 [==============================] - 0s 4ms/step - loss: 609604.0000 - val_loss: 609052.8125
    Epoch 1596/2000
    8/8 [==============================] - 0s 4ms/step - loss: 608388.1875 - val_loss: 606995.6250
    Epoch 1597/2000
    8/8 [==============================] - 0s 4ms/step - loss: 608091.5625 - val_loss: 608284.8125
    Epoch 1598/2000
    8/8 [==============================] - 0s 4ms/step - loss: 608085.3750 - val_loss: 610570.0000
    Epoch 1599/2000
    8/8 [==============================] - 0s 4ms/step - loss: 609274.3750 - val_loss: 608389.7500
    Epoch 1600/2000
    8/8 [==============================] - 0s 4ms/step - loss: 607415.0625 - val_loss: 609354.1250
    Epoch 1601/2000
    8/8 [==============================] - 0s 4ms/step - loss: 607465.3750 - val_loss: 608885.8125
    Epoch 1602/2000
    8/8 [==============================] - 0s 4ms/step - loss: 608158.6250 - val_loss: 608124.0000
    Epoch 1603/2000
    8/8 [==============================] - 0s 4ms/step - loss: 606811.3750 - val_loss: 608499.1875
    Epoch 1604/2000
    8/8 [==============================] - 0s 4ms/step - loss: 609144.5625 - val_loss: 607102.3750
    Epoch 1605/2000
    8/8 [==============================] - 0s 4ms/step - loss: 607724.0625 - val_loss: 607488.3125
    Epoch 1606/2000
    8/8 [==============================] - 0s 4ms/step - loss: 606962.1875 - val_loss: 604927.0625
    Epoch 1607/2000
    8/8 [==============================] - 0s 5ms/step - loss: 607446.3750 - val_loss: 607152.3125
    Epoch 1608/2000
    8/8 [==============================] - 0s 4ms/step - loss: 606021.3125 - val_loss: 606892.4375
    Epoch 1609/2000
    8/8 [==============================] - 0s 4ms/step - loss: 607289.3750 - val_loss: 607245.6250
    Epoch 1610/2000
    8/8 [==============================] - 0s 4ms/step - loss: 605810.5625 - val_loss: 606266.6875
    Epoch 1611/2000
    8/8 [==============================] - 0s 4ms/step - loss: 607989.8125 - val_loss: 602585.2500
    Epoch 1612/2000
    8/8 [==============================] - 0s 4ms/step - loss: 605970.6250 - val_loss: 606460.2500


    Epoch 1613/2000
    8/8 [==============================] - 0s 4ms/step - loss: 606600.7500 - val_loss: 603659.0625
    Epoch 1614/2000
    8/8 [==============================] - 0s 4ms/step - loss: 608369.4375 - val_loss: 606744.8125
    Epoch 1615/2000
    8/8 [==============================] - 0s 4ms/step - loss: 605320.3125 - val_loss: 602264.3125
    Epoch 1616/2000
    8/8 [==============================] - 0s 4ms/step - loss: 605480.8750 - val_loss: 606040.6875
    Epoch 1617/2000
    8/8 [==============================] - 0s 4ms/step - loss: 604412.1875 - val_loss: 603235.7500
    Epoch 1618/2000
    8/8 [==============================] - 0s 4ms/step - loss: 604028.0000 - val_loss: 606080.5625
    Epoch 1619/2000
    8/8 [==============================] - 0s 4ms/step - loss: 603964.8125 - val_loss: 606047.4375
    Epoch 1620/2000
    8/8 [==============================] - 0s 4ms/step - loss: 603783.9375 - val_loss: 606182.6875
    Epoch 1621/2000
    8/8 [==============================] - 0s 4ms/step - loss: 602772.0000 - val_loss: 605265.3125
    Epoch 1622/2000
    8/8 [==============================] - 0s 4ms/step - loss: 604375.2500 - val_loss: 604068.0625
    Epoch 1623/2000
    8/8 [==============================] - 0s 4ms/step - loss: 602184.7500 - val_loss: 603950.6875
    Epoch 1624/2000
    8/8 [==============================] - 0s 4ms/step - loss: 601774.7500 - val_loss: 600398.6250
    Epoch 1625/2000
    8/8 [==============================] - 0s 4ms/step - loss: 604303.0000 - val_loss: 604819.6875
    Epoch 1626/2000
    8/8 [==============================] - 0s 4ms/step - loss: 603895.3125 - val_loss: 604351.1250
    Epoch 1627/2000
    8/8 [==============================] - 0s 4ms/step - loss: 604001.7500 - val_loss: 599297.0625
    Epoch 1628/2000
    8/8 [==============================] - 0s 4ms/step - loss: 602315.5625 - val_loss: 601740.5625
    Epoch 1629/2000
    8/8 [==============================] - 0s 4ms/step - loss: 601910.6875 - val_loss: 602153.7500
    Epoch 1630/2000
    8/8 [==============================] - 0s 4ms/step - loss: 601621.2500 - val_loss: 603610.1250
    Epoch 1631/2000
    8/8 [==============================] - 0s 4ms/step - loss: 603017.5000 - val_loss: 602061.1875
    Epoch 1632/2000
    8/8 [==============================] - 0s 4ms/step - loss: 601387.6875 - val_loss: 601850.0000
    Epoch 1633/2000
    8/8 [==============================] - 0s 4ms/step - loss: 601761.1250 - val_loss: 603185.6250
    Epoch 1634/2000
    8/8 [==============================] - 0s 4ms/step - loss: 601734.8750 - val_loss: 600448.1875
    Epoch 1635/2000
    8/8 [==============================] - 0s 4ms/step - loss: 602731.8125 - val_loss: 600681.3125
    Epoch 1636/2000
    8/8 [==============================] - 0s 4ms/step - loss: 601745.5625 - val_loss: 602450.0625
    Epoch 1637/2000
    8/8 [==============================] - 0s 4ms/step - loss: 601441.8750 - val_loss: 603728.7500
    Epoch 1638/2000
    8/8 [==============================] - 0s 4ms/step - loss: 601843.3125 - val_loss: 601329.6250
    Epoch 1639/2000
    8/8 [==============================] - 0s 4ms/step - loss: 600390.0000 - val_loss: 598733.3125
    Epoch 1640/2000
    8/8 [==============================] - 0s 4ms/step - loss: 599521.0625 - val_loss: 601597.8750
    Epoch 1641/2000
    8/8 [==============================] - 0s 4ms/step - loss: 598829.2500 - val_loss: 600707.6875
    Epoch 1642/2000
    8/8 [==============================] - 0s 4ms/step - loss: 600577.7500 - val_loss: 599335.3750
    Epoch 1643/2000
    8/8 [==============================] - 0s 4ms/step - loss: 598758.6250 - val_loss: 600131.1250
    Epoch 1644/2000
    8/8 [==============================] - 0s 4ms/step - loss: 599310.2500 - val_loss: 598351.7500
    Epoch 1645/2000
    8/8 [==============================] - 0s 4ms/step - loss: 599308.1875 - val_loss: 599998.0000
    Epoch 1646/2000
    8/8 [==============================] - 0s 4ms/step - loss: 599656.1875 - val_loss: 601105.5625
    Epoch 1647/2000
    8/8 [==============================] - 0s 4ms/step - loss: 600370.5000 - val_loss: 598215.8125
    Epoch 1648/2000
    8/8 [==============================] - 0s 4ms/step - loss: 599004.7500 - val_loss: 599137.6875
    Epoch 1649/2000
    8/8 [==============================] - 0s 4ms/step - loss: 599311.3125 - val_loss: 598046.7500
    Epoch 1650/2000
    8/8 [==============================] - 0s 4ms/step - loss: 598722.5000 - val_loss: 598165.6250
    Epoch 1651/2000
    8/8 [==============================] - 0s 4ms/step - loss: 597791.6875 - val_loss: 598842.6875
    Epoch 1652/2000
    8/8 [==============================] - 0s 4ms/step - loss: 598118.7500 - val_loss: 598462.3125
    Epoch 1653/2000
    8/8 [==============================] - 0s 4ms/step - loss: 598680.1250 - val_loss: 597776.5625
    Epoch 1654/2000
    8/8 [==============================] - 0s 4ms/step - loss: 598541.6250 - val_loss: 599177.6875
    Epoch 1655/2000
    8/8 [==============================] - 0s 4ms/step - loss: 598608.3125 - val_loss: 598820.1250
    Epoch 1656/2000
    8/8 [==============================] - 0s 4ms/step - loss: 598091.2500 - val_loss: 598297.3125
    Epoch 1657/2000
    8/8 [==============================] - 0s 4ms/step - loss: 596032.5000 - val_loss: 597401.0625
    Epoch 1658/2000
    8/8 [==============================] - 0s 4ms/step - loss: 596201.0000 - val_loss: 596035.8125
    Epoch 1659/2000
    8/8 [==============================] - 0s 4ms/step - loss: 596902.8125 - val_loss: 594238.8750
    Epoch 1660/2000
    8/8 [==============================] - 0s 4ms/step - loss: 596542.2500 - val_loss: 597445.1250
    Epoch 1661/2000
    8/8 [==============================] - 0s 4ms/step - loss: 596865.3750 - val_loss: 596680.1875
    Epoch 1662/2000
    8/8 [==============================] - 0s 4ms/step - loss: 596387.5625 - val_loss: 593337.3750
    Epoch 1663/2000
    8/8 [==============================] - 0s 4ms/step - loss: 596359.0625 - val_loss: 596410.0625
    Epoch 1664/2000
    8/8 [==============================] - 0s 4ms/step - loss: 596713.4375 - val_loss: 596402.0000
    Epoch 1665/2000
    8/8 [==============================] - 0s 4ms/step - loss: 595127.8750 - val_loss: 595042.3750
    Epoch 1666/2000
    8/8 [==============================] - 0s 4ms/step - loss: 594903.3125 - val_loss: 597052.1875
    Epoch 1667/2000
    8/8 [==============================] - 0s 4ms/step - loss: 595495.8750 - val_loss: 595733.0000
    Epoch 1668/2000
    8/8 [==============================] - 0s 4ms/step - loss: 595182.8750 - val_loss: 594091.1875
    Epoch 1669/2000
    8/8 [==============================] - 0s 4ms/step - loss: 594950.8750 - val_loss: 596060.2500
    Epoch 1670/2000
    8/8 [==============================] - 0s 4ms/step - loss: 594971.4375 - val_loss: 595170.8750
    Epoch 1671/2000
    8/8 [==============================] - 0s 4ms/step - loss: 594219.3750 - val_loss: 595905.5625
    Epoch 1672/2000
    8/8 [==============================] - 0s 4ms/step - loss: 595051.6875 - val_loss: 595102.0000
    Epoch 1673/2000
    8/8 [==============================] - 0s 4ms/step - loss: 593506.4375 - val_loss: 596095.4375
    Epoch 1674/2000
    8/8 [==============================] - 0s 4ms/step - loss: 591995.0000 - val_loss: 595875.2500
    Epoch 1675/2000
    8/8 [==============================] - 0s 4ms/step - loss: 594801.1250 - val_loss: 594510.8750
    Epoch 1676/2000
    8/8 [==============================] - 0s 4ms/step - loss: 593685.8125 - val_loss: 590901.1875
    Epoch 1677/2000
    8/8 [==============================] - 0s 4ms/step - loss: 593802.0625 - val_loss: 592292.1875
    Epoch 1678/2000
    8/8 [==============================] - 0s 4ms/step - loss: 591547.8125 - val_loss: 591411.1875
    Epoch 1679/2000
    8/8 [==============================] - 0s 4ms/step - loss: 594139.0000 - val_loss: 591261.3125
    Epoch 1680/2000
    8/8 [==============================] - 0s 4ms/step - loss: 592889.7500 - val_loss: 591518.6875
    Epoch 1681/2000
    8/8 [==============================] - 0s 4ms/step - loss: 593420.3125 - val_loss: 591628.1875
    Epoch 1682/2000
    8/8 [==============================] - 0s 4ms/step - loss: 593807.7500 - val_loss: 593750.0000
    Epoch 1683/2000
    8/8 [==============================] - 0s 4ms/step - loss: 591582.5000 - val_loss: 595230.0000
    Epoch 1684/2000
    8/8 [==============================] - 0s 4ms/step - loss: 590027.4375 - val_loss: 591897.0000
    Epoch 1685/2000
    8/8 [==============================] - 0s 4ms/step - loss: 592347.0000 - val_loss: 592812.7500
    Epoch 1686/2000
    8/8 [==============================] - 0s 4ms/step - loss: 590760.1250 - val_loss: 592602.7500


    Epoch 1687/2000
    8/8 [==============================] - 0s 4ms/step - loss: 590786.1250 - val_loss: 590266.6250
    Epoch 1688/2000
    8/8 [==============================] - 0s 4ms/step - loss: 591106.1875 - val_loss: 590366.6875
    Epoch 1689/2000
    8/8 [==============================] - 0s 4ms/step - loss: 590575.4375 - val_loss: 593435.5625
    Epoch 1690/2000
    8/8 [==============================] - 0s 4ms/step - loss: 592015.0000 - val_loss: 590436.6250
    Epoch 1691/2000
    8/8 [==============================] - 0s 4ms/step - loss: 589963.0625 - val_loss: 590862.8750
    Epoch 1692/2000
    8/8 [==============================] - 0s 4ms/step - loss: 589451.5000 - val_loss: 588554.8125
    Epoch 1693/2000
    8/8 [==============================] - 0s 4ms/step - loss: 589950.3125 - val_loss: 592740.1250
    Epoch 1694/2000
    8/8 [==============================] - 0s 4ms/step - loss: 589885.8750 - val_loss: 590096.8750
    Epoch 1695/2000
    8/8 [==============================] - 0s 4ms/step - loss: 589655.8125 - val_loss: 587964.1875
    Epoch 1696/2000
    8/8 [==============================] - 0s 4ms/step - loss: 588994.0625 - val_loss: 590078.3750
    Epoch 1697/2000
    8/8 [==============================] - 0s 4ms/step - loss: 589759.1875 - val_loss: 587879.4375
    Epoch 1698/2000
    8/8 [==============================] - 0s 4ms/step - loss: 590774.6250 - val_loss: 588957.1250
    Epoch 1699/2000
    8/8 [==============================] - 0s 4ms/step - loss: 591316.3125 - val_loss: 588569.9375
    Epoch 1700/2000
    8/8 [==============================] - 0s 4ms/step - loss: 588397.5625 - val_loss: 588578.1875
    Epoch 1701/2000
    8/8 [==============================] - 0s 4ms/step - loss: 589075.8750 - val_loss: 590451.7500
    Epoch 1702/2000
    8/8 [==============================] - 0s 4ms/step - loss: 587469.3125 - val_loss: 589725.7500
    Epoch 1703/2000
    8/8 [==============================] - 0s 4ms/step - loss: 588838.3125 - val_loss: 587985.5000
    Epoch 1704/2000
    8/8 [==============================] - 0s 4ms/step - loss: 586973.5000 - val_loss: 587758.9375
    Epoch 1705/2000
    8/8 [==============================] - 0s 4ms/step - loss: 587409.0000 - val_loss: 588466.5625
    Epoch 1706/2000
    8/8 [==============================] - 0s 4ms/step - loss: 588055.6250 - val_loss: 589406.3125
    Epoch 1707/2000
    8/8 [==============================] - 0s 4ms/step - loss: 589016.1875 - val_loss: 590432.5625
    Epoch 1708/2000
    8/8 [==============================] - 0s 4ms/step - loss: 587440.0000 - val_loss: 587955.0625
    Epoch 1709/2000
    8/8 [==============================] - 0s 4ms/step - loss: 587294.6250 - val_loss: 587347.9375
    Epoch 1710/2000
    8/8 [==============================] - 0s 4ms/step - loss: 586592.5625 - val_loss: 586013.6875
    Epoch 1711/2000
    8/8 [==============================] - 0s 4ms/step - loss: 586281.5000 - val_loss: 585934.6875
    Epoch 1712/2000
    8/8 [==============================] - 0s 4ms/step - loss: 585119.3750 - val_loss: 588241.9375
    Epoch 1713/2000
    8/8 [==============================] - 0s 4ms/step - loss: 586381.0000 - val_loss: 584306.1250
    Epoch 1714/2000
    8/8 [==============================] - 0s 4ms/step - loss: 585681.6875 - val_loss: 585440.1875
    Epoch 1715/2000
    8/8 [==============================] - 0s 4ms/step - loss: 585899.1875 - val_loss: 585548.1250
    Epoch 1716/2000
    8/8 [==============================] - 0s 4ms/step - loss: 584775.3125 - val_loss: 584944.0000
    Epoch 1717/2000
    8/8 [==============================] - 0s 4ms/step - loss: 586054.1875 - val_loss: 586078.8750
    Epoch 1718/2000
    8/8 [==============================] - 0s 4ms/step - loss: 583728.6875 - val_loss: 587392.8125
    Epoch 1719/2000
    8/8 [==============================] - 0s 4ms/step - loss: 583956.4375 - val_loss: 586847.8125
    Epoch 1720/2000
    8/8 [==============================] - 0s 4ms/step - loss: 585520.3750 - val_loss: 585799.2500
    Epoch 1721/2000
    8/8 [==============================] - 0s 4ms/step - loss: 585863.3750 - val_loss: 583699.1250
    Epoch 1722/2000
    8/8 [==============================] - 0s 4ms/step - loss: 585814.2500 - val_loss: 587062.8125
    Epoch 1723/2000
    8/8 [==============================] - 0s 4ms/step - loss: 585386.8125 - val_loss: 583429.1875
    Epoch 1724/2000
    8/8 [==============================] - 0s 4ms/step - loss: 583501.1250 - val_loss: 585997.6250
    Epoch 1725/2000
    8/8 [==============================] - 0s 4ms/step - loss: 583377.4375 - val_loss: 587302.9375
    Epoch 1726/2000
    8/8 [==============================] - 0s 4ms/step - loss: 583607.1875 - val_loss: 584515.3750
    Epoch 1727/2000
    8/8 [==============================] - 0s 4ms/step - loss: 582906.5000 - val_loss: 586866.5000
    Epoch 1728/2000
    8/8 [==============================] - 0s 4ms/step - loss: 584487.0000 - val_loss: 584267.3125
    Epoch 1729/2000
    8/8 [==============================] - 0s 4ms/step - loss: 584112.0625 - val_loss: 582967.1875
    Epoch 1730/2000
    8/8 [==============================] - 0s 4ms/step - loss: 582134.1250 - val_loss: 585669.5625
    Epoch 1731/2000
    8/8 [==============================] - 0s 4ms/step - loss: 583460.1875 - val_loss: 581721.2500
    Epoch 1732/2000
    8/8 [==============================] - 0s 4ms/step - loss: 582598.2500 - val_loss: 579993.0625
    Epoch 1733/2000
    8/8 [==============================] - 0s 4ms/step - loss: 583097.0000 - val_loss: 581238.3125
    Epoch 1734/2000
    8/8 [==============================] - 0s 4ms/step - loss: 581596.0625 - val_loss: 580511.1250
    Epoch 1735/2000
    8/8 [==============================] - 0s 4ms/step - loss: 582128.1250 - val_loss: 583766.7500
    Epoch 1736/2000
    8/8 [==============================] - 0s 4ms/step - loss: 583274.3750 - val_loss: 582211.8125
    Epoch 1737/2000
    8/8 [==============================] - 0s 4ms/step - loss: 580368.1250 - val_loss: 581534.9375
    Epoch 1738/2000
    8/8 [==============================] - 0s 4ms/step - loss: 582192.6250 - val_loss: 578569.0625
    Epoch 1739/2000
    8/8 [==============================] - 0s 4ms/step - loss: 582296.9375 - val_loss: 583263.8125
    Epoch 1740/2000
    8/8 [==============================] - 0s 4ms/step - loss: 581353.6250 - val_loss: 581560.3750
    Epoch 1741/2000
    8/8 [==============================] - 0s 4ms/step - loss: 580922.3125 - val_loss: 580376.0000
    Epoch 1742/2000
    8/8 [==============================] - 0s 4ms/step - loss: 580337.2500 - val_loss: 581247.8125
    Epoch 1743/2000
    8/8 [==============================] - 0s 4ms/step - loss: 581012.0625 - val_loss: 582314.0000
    Epoch 1744/2000
    8/8 [==============================] - 0s 4ms/step - loss: 581195.8125 - val_loss: 582721.2500
    Epoch 1745/2000
    8/8 [==============================] - 0s 4ms/step - loss: 579616.8125 - val_loss: 578651.1250
    Epoch 1746/2000
    8/8 [==============================] - 0s 4ms/step - loss: 580554.8750 - val_loss: 579603.0625
    Epoch 1747/2000
    8/8 [==============================] - 0s 4ms/step - loss: 580238.9375 - val_loss: 579360.8750
    Epoch 1748/2000
    8/8 [==============================] - 0s 4ms/step - loss: 581766.1250 - val_loss: 579184.6875
    Epoch 1749/2000
    8/8 [==============================] - 0s 4ms/step - loss: 581282.3750 - val_loss: 579599.2500
    Epoch 1750/2000
    8/8 [==============================] - 0s 4ms/step - loss: 580744.8125 - val_loss: 580037.1250
    Epoch 1751/2000
    8/8 [==============================] - 0s 4ms/step - loss: 578244.5000 - val_loss: 579011.0000
    Epoch 1752/2000
    8/8 [==============================] - 0s 4ms/step - loss: 579434.5000 - val_loss: 582529.2500
    Epoch 1753/2000
    8/8 [==============================] - 0s 4ms/step - loss: 578132.4375 - val_loss: 577276.3750
    Epoch 1754/2000
    8/8 [==============================] - 0s 4ms/step - loss: 578275.9375 - val_loss: 578404.4375
    Epoch 1755/2000
    8/8 [==============================] - 0s 4ms/step - loss: 578574.5000 - val_loss: 573786.7500
    Epoch 1756/2000
    8/8 [==============================] - 0s 4ms/step - loss: 579235.1250 - val_loss: 577008.7500
    Epoch 1757/2000
    8/8 [==============================] - 0s 4ms/step - loss: 579224.1875 - val_loss: 578622.8125
    Epoch 1758/2000
    8/8 [==============================] - 0s 4ms/step - loss: 575996.3125 - val_loss: 578213.5000
    Epoch 1759/2000
    8/8 [==============================] - 0s 4ms/step - loss: 578480.9375 - val_loss: 576937.0625
    Epoch 1760/2000
    8/8 [==============================] - 0s 4ms/step - loss: 578842.0000 - val_loss: 577578.7500


    Epoch 1761/2000
    8/8 [==============================] - 0s 4ms/step - loss: 576306.6250 - val_loss: 576664.1250
    Epoch 1762/2000
    8/8 [==============================] - 0s 4ms/step - loss: 577105.4375 - val_loss: 577484.8125
    Epoch 1763/2000
    8/8 [==============================] - 0s 4ms/step - loss: 576837.5000 - val_loss: 576796.6250
    Epoch 1764/2000
    8/8 [==============================] - 0s 4ms/step - loss: 576990.6250 - val_loss: 577447.9375
    Epoch 1765/2000
    8/8 [==============================] - 0s 4ms/step - loss: 576647.7500 - val_loss: 577576.1875
    Epoch 1766/2000
    8/8 [==============================] - 0s 4ms/step - loss: 576674.5000 - val_loss: 575671.0000
    Epoch 1767/2000
    8/8 [==============================] - 0s 4ms/step - loss: 575940.5625 - val_loss: 575026.7500
    Epoch 1768/2000
    8/8 [==============================] - 0s 4ms/step - loss: 575004.9375 - val_loss: 573340.6875
    Epoch 1769/2000
    8/8 [==============================] - 0s 4ms/step - loss: 575943.4375 - val_loss: 574569.6250
    Epoch 1770/2000
    8/8 [==============================] - 0s 4ms/step - loss: 577299.5000 - val_loss: 576154.8125
    Epoch 1771/2000
    8/8 [==============================] - 0s 4ms/step - loss: 574884.3750 - val_loss: 577309.6250
    Epoch 1772/2000
    8/8 [==============================] - 0s 4ms/step - loss: 575465.1875 - val_loss: 575818.5625
    Epoch 1773/2000
    8/8 [==============================] - 0s 4ms/step - loss: 575228.1250 - val_loss: 575867.5000
    Epoch 1774/2000
    8/8 [==============================] - 0s 4ms/step - loss: 575429.5000 - val_loss: 574828.5000
    Epoch 1775/2000
    8/8 [==============================] - 0s 4ms/step - loss: 573878.8750 - val_loss: 572640.1875
    Epoch 1776/2000
    8/8 [==============================] - 0s 4ms/step - loss: 572860.5000 - val_loss: 574765.2500
    Epoch 1777/2000
    8/8 [==============================] - 0s 4ms/step - loss: 576051.5625 - val_loss: 571408.7500
    Epoch 1778/2000
    8/8 [==============================] - 0s 4ms/step - loss: 573116.6875 - val_loss: 572373.7500
    Epoch 1779/2000
    8/8 [==============================] - 0s 4ms/step - loss: 574815.3750 - val_loss: 575136.5625
    Epoch 1780/2000
    8/8 [==============================] - 0s 4ms/step - loss: 574440.2500 - val_loss: 571134.6250
    Epoch 1781/2000
    8/8 [==============================] - 0s 4ms/step - loss: 574941.5625 - val_loss: 573477.6250
    Epoch 1782/2000
    8/8 [==============================] - 0s 4ms/step - loss: 575486.1250 - val_loss: 573555.0000
    Epoch 1783/2000
    8/8 [==============================] - 0s 4ms/step - loss: 573749.8125 - val_loss: 575710.3125
    Epoch 1784/2000
    8/8 [==============================] - 0s 4ms/step - loss: 573104.3750 - val_loss: 572838.5625
    Epoch 1785/2000
    8/8 [==============================] - 0s 4ms/step - loss: 572705.3750 - val_loss: 575265.5000
    Epoch 1786/2000
    8/8 [==============================] - 0s 4ms/step - loss: 572558.4375 - val_loss: 574866.6250
    Epoch 1787/2000
    8/8 [==============================] - 0s 4ms/step - loss: 571882.9375 - val_loss: 571317.7500
    Epoch 1788/2000
    8/8 [==============================] - 0s 4ms/step - loss: 574034.0000 - val_loss: 569719.5625
    Epoch 1789/2000
    8/8 [==============================] - 0s 4ms/step - loss: 572804.9375 - val_loss: 570501.7500
    Epoch 1790/2000
    8/8 [==============================] - 0s 4ms/step - loss: 573535.5625 - val_loss: 573356.0625
    Epoch 1791/2000
    8/8 [==============================] - 0s 4ms/step - loss: 571643.6875 - val_loss: 571280.7500
    Epoch 1792/2000
    8/8 [==============================] - 0s 4ms/step - loss: 571731.0000 - val_loss: 571517.6250
    Epoch 1793/2000
    8/8 [==============================] - 0s 4ms/step - loss: 571740.0625 - val_loss: 570028.5625
    Epoch 1794/2000
    8/8 [==============================] - 0s 4ms/step - loss: 570566.3125 - val_loss: 572977.0625
    Epoch 1795/2000
    8/8 [==============================] - 0s 4ms/step - loss: 569291.6875 - val_loss: 572547.0625
    Epoch 1796/2000
    8/8 [==============================] - 0s 4ms/step - loss: 569016.8750 - val_loss: 570242.8750
    Epoch 1797/2000
    8/8 [==============================] - 0s 4ms/step - loss: 571190.7500 - val_loss: 571057.7500
    Epoch 1798/2000
    8/8 [==============================] - 0s 4ms/step - loss: 571590.0000 - val_loss: 571304.0625
    Epoch 1799/2000
    8/8 [==============================] - 0s 4ms/step - loss: 571698.7500 - val_loss: 569626.6875
    Epoch 1800/2000
    8/8 [==============================] - 0s 4ms/step - loss: 571131.7500 - val_loss: 569817.5000
    Epoch 1801/2000
    8/8 [==============================] - 0s 4ms/step - loss: 571247.2500 - val_loss: 571107.8750
    Epoch 1802/2000
    8/8 [==============================] - 0s 4ms/step - loss: 569309.1875 - val_loss: 569582.7500
    Epoch 1803/2000
    8/8 [==============================] - 0s 4ms/step - loss: 568725.8125 - val_loss: 571482.7500
    Epoch 1804/2000
    8/8 [==============================] - 0s 4ms/step - loss: 570894.6250 - val_loss: 569187.8125
    Epoch 1805/2000
    8/8 [==============================] - 0s 4ms/step - loss: 568042.4375 - val_loss: 569871.0625
    Epoch 1806/2000
    8/8 [==============================] - 0s 4ms/step - loss: 570241.1250 - val_loss: 570485.2500
    Epoch 1807/2000
    8/8 [==============================] - 0s 4ms/step - loss: 568420.3750 - val_loss: 567839.3125
    Epoch 1808/2000
    8/8 [==============================] - 0s 4ms/step - loss: 568088.9375 - val_loss: 566593.6250
    Epoch 1809/2000
    8/8 [==============================] - 0s 4ms/step - loss: 568248.0000 - val_loss: 567115.0625
    Epoch 1810/2000
    8/8 [==============================] - 0s 4ms/step - loss: 567931.6875 - val_loss: 570964.9375
    Epoch 1811/2000
    8/8 [==============================] - 0s 4ms/step - loss: 568160.0000 - val_loss: 566990.8125
    Epoch 1812/2000
    8/8 [==============================] - 0s 4ms/step - loss: 568020.2500 - val_loss: 567898.3125
    Epoch 1813/2000
    8/8 [==============================] - 0s 4ms/step - loss: 568136.5625 - val_loss: 570572.7500
    Epoch 1814/2000
    8/8 [==============================] - 0s 4ms/step - loss: 569293.6250 - val_loss: 568257.5000
    Epoch 1815/2000
    8/8 [==============================] - 0s 4ms/step - loss: 568184.5625 - val_loss: 567850.5625
    Epoch 1816/2000
    8/8 [==============================] - 0s 4ms/step - loss: 567828.9375 - val_loss: 566604.7500
    Epoch 1817/2000
    8/8 [==============================] - 0s 5ms/step - loss: 568832.6250 - val_loss: 567837.8750
    Epoch 1818/2000
    8/8 [==============================] - 0s 4ms/step - loss: 568225.6875 - val_loss: 565813.3750
    Epoch 1819/2000
    8/8 [==============================] - 0s 4ms/step - loss: 565970.2500 - val_loss: 568178.0000
    Epoch 1820/2000
    8/8 [==============================] - 0s 4ms/step - loss: 565812.5625 - val_loss: 566821.1250
    Epoch 1821/2000
    8/8 [==============================] - 0s 4ms/step - loss: 566740.6250 - val_loss: 569189.3750
    Epoch 1822/2000
    8/8 [==============================] - 0s 4ms/step - loss: 565201.9375 - val_loss: 568731.2500
    Epoch 1823/2000
    8/8 [==============================] - 0s 4ms/step - loss: 566209.5000 - val_loss: 565417.8750
    Epoch 1824/2000
    8/8 [==============================] - 0s 4ms/step - loss: 566436.6250 - val_loss: 565749.2500
    Epoch 1825/2000
    8/8 [==============================] - 0s 4ms/step - loss: 565608.8125 - val_loss: 564124.2500
    Epoch 1826/2000
    8/8 [==============================] - 0s 4ms/step - loss: 565981.5625 - val_loss: 564789.5625
    Epoch 1827/2000
    8/8 [==============================] - 0s 4ms/step - loss: 564934.1250 - val_loss: 562808.6875
    Epoch 1828/2000
    8/8 [==============================] - 0s 4ms/step - loss: 565113.5000 - val_loss: 565051.6875
    Epoch 1829/2000
    8/8 [==============================] - 0s 4ms/step - loss: 564388.8125 - val_loss: 563219.1250
    Epoch 1830/2000
    8/8 [==============================] - 0s 4ms/step - loss: 565824.2500 - val_loss: 563013.6250
    Epoch 1831/2000
    8/8 [==============================] - 0s 4ms/step - loss: 566100.1875 - val_loss: 563542.9375
    Epoch 1832/2000
    8/8 [==============================] - 0s 4ms/step - loss: 563318.4375 - val_loss: 563396.5625
    Epoch 1833/2000
    8/8 [==============================] - 0s 4ms/step - loss: 564031.8125 - val_loss: 564192.2500
    Epoch 1834/2000
    8/8 [==============================] - 0s 4ms/step - loss: 563444.8125 - val_loss: 562670.9375


    Epoch 1835/2000
    8/8 [==============================] - 0s 4ms/step - loss: 563635.5625 - val_loss: 567140.1875
    Epoch 1836/2000
    8/8 [==============================] - 0s 4ms/step - loss: 563307.9375 - val_loss: 562122.7500
    Epoch 1837/2000
    8/8 [==============================] - 0s 4ms/step - loss: 563907.0625 - val_loss: 559996.8125
    Epoch 1838/2000
    8/8 [==============================] - 0s 4ms/step - loss: 564540.1875 - val_loss: 564405.5000
    Epoch 1839/2000
    8/8 [==============================] - 0s 4ms/step - loss: 564496.8750 - val_loss: 564952.8125
    Epoch 1840/2000
    8/8 [==============================] - 0s 4ms/step - loss: 564269.3750 - val_loss: 563882.7500
    Epoch 1841/2000
    8/8 [==============================] - 0s 4ms/step - loss: 563042.2500 - val_loss: 561661.2500
    Epoch 1842/2000
    8/8 [==============================] - 0s 4ms/step - loss: 563169.8750 - val_loss: 564137.9375
    Epoch 1843/2000
    8/8 [==============================] - 0s 4ms/step - loss: 563015.3750 - val_loss: 561858.2500
    Epoch 1844/2000
    8/8 [==============================] - 0s 4ms/step - loss: 561696.9375 - val_loss: 560710.0625
    Epoch 1845/2000
    8/8 [==============================] - 0s 4ms/step - loss: 562733.5625 - val_loss: 564064.0000
    Epoch 1846/2000
    8/8 [==============================] - 0s 4ms/step - loss: 561960.0625 - val_loss: 562992.5625
    Epoch 1847/2000
    8/8 [==============================] - 0s 4ms/step - loss: 561932.2500 - val_loss: 560237.8750
    Epoch 1848/2000
    8/8 [==============================] - 0s 4ms/step - loss: 560123.5000 - val_loss: 561200.0625
    Epoch 1849/2000
    8/8 [==============================] - 0s 4ms/step - loss: 561477.3750 - val_loss: 563939.5625
    Epoch 1850/2000
    8/8 [==============================] - 0s 4ms/step - loss: 561269.0625 - val_loss: 560094.1250
    Epoch 1851/2000
    8/8 [==============================] - 0s 4ms/step - loss: 562043.3125 - val_loss: 562606.2500
    Epoch 1852/2000
    8/8 [==============================] - 0s 4ms/step - loss: 561051.1250 - val_loss: 559950.8125
    Epoch 1853/2000
    8/8 [==============================] - 0s 4ms/step - loss: 560683.8125 - val_loss: 559476.4375
    Epoch 1854/2000
    8/8 [==============================] - 0s 4ms/step - loss: 560360.1250 - val_loss: 562015.5625
    Epoch 1855/2000
    8/8 [==============================] - 0s 4ms/step - loss: 560357.3125 - val_loss: 560995.1250
    Epoch 1856/2000
    8/8 [==============================] - 0s 4ms/step - loss: 560785.0000 - val_loss: 559633.8125
    Epoch 1857/2000
    8/8 [==============================] - 0s 4ms/step - loss: 562706.1875 - val_loss: 560320.1875
    Epoch 1858/2000
    8/8 [==============================] - 0s 4ms/step - loss: 560156.0000 - val_loss: 559536.5000
    Epoch 1859/2000
    8/8 [==============================] - 0s 4ms/step - loss: 560874.9375 - val_loss: 559598.0000
    Epoch 1860/2000
    8/8 [==============================] - 0s 4ms/step - loss: 559620.5625 - val_loss: 555955.0000
    Epoch 1861/2000
    8/8 [==============================] - 0s 4ms/step - loss: 559509.1875 - val_loss: 559009.0000
    Epoch 1862/2000
    8/8 [==============================] - 0s 4ms/step - loss: 558354.7500 - val_loss: 557040.7500
    Epoch 1863/2000
    8/8 [==============================] - 0s 4ms/step - loss: 560337.9375 - val_loss: 556371.4375
    Epoch 1864/2000
    8/8 [==============================] - 0s 4ms/step - loss: 558395.9375 - val_loss: 558436.2500
    Epoch 1865/2000
    8/8 [==============================] - 0s 4ms/step - loss: 559513.5000 - val_loss: 557625.2500
    Epoch 1866/2000
    8/8 [==============================] - 0s 4ms/step - loss: 560314.1250 - val_loss: 557840.1875
    Epoch 1867/2000
    8/8 [==============================] - 0s 4ms/step - loss: 556746.5000 - val_loss: 557739.4375
    Epoch 1868/2000
    8/8 [==============================] - 0s 4ms/step - loss: 557926.3125 - val_loss: 557152.3125
    Epoch 1869/2000
    8/8 [==============================] - 0s 4ms/step - loss: 558757.0625 - val_loss: 557508.7500
    Epoch 1870/2000
    8/8 [==============================] - 0s 4ms/step - loss: 557143.1250 - val_loss: 557429.5000
    Epoch 1871/2000
    8/8 [==============================] - 0s 4ms/step - loss: 557752.5625 - val_loss: 557376.1250
    Epoch 1872/2000
    8/8 [==============================] - 0s 4ms/step - loss: 558065.6875 - val_loss: 557574.1250
    Epoch 1873/2000
    8/8 [==============================] - 0s 4ms/step - loss: 557457.2500 - val_loss: 559742.3750
    Epoch 1874/2000
    8/8 [==============================] - 0s 4ms/step - loss: 555501.6250 - val_loss: 555757.1875
    Epoch 1875/2000
    8/8 [==============================] - 0s 4ms/step - loss: 557618.1875 - val_loss: 558508.3125
    Epoch 1876/2000
    8/8 [==============================] - 0s 4ms/step - loss: 557428.4375 - val_loss: 558944.1250
    Epoch 1877/2000
    8/8 [==============================] - 0s 4ms/step - loss: 556570.5625 - val_loss: 555517.2500
    Epoch 1878/2000
    8/8 [==============================] - 0s 4ms/step - loss: 556894.0000 - val_loss: 557755.7500
    Epoch 1879/2000
    8/8 [==============================] - 0s 4ms/step - loss: 555659.3125 - val_loss: 557135.5000
    Epoch 1880/2000
    8/8 [==============================] - 0s 4ms/step - loss: 556563.6875 - val_loss: 557119.9375
    Epoch 1881/2000
    8/8 [==============================] - 0s 4ms/step - loss: 556169.1250 - val_loss: 556720.9375
    Epoch 1882/2000
    8/8 [==============================] - 0s 4ms/step - loss: 555700.0000 - val_loss: 557295.6250
    Epoch 1883/2000
    8/8 [==============================] - 0s 4ms/step - loss: 555337.6875 - val_loss: 554152.1250
    Epoch 1884/2000
    8/8 [==============================] - 0s 4ms/step - loss: 554252.8125 - val_loss: 554961.3750
    Epoch 1885/2000
    8/8 [==============================] - 0s 4ms/step - loss: 555796.5000 - val_loss: 554827.1250
    Epoch 1886/2000
    8/8 [==============================] - 0s 4ms/step - loss: 555911.8125 - val_loss: 555673.6875
    Epoch 1887/2000
    8/8 [==============================] - 0s 4ms/step - loss: 555028.6875 - val_loss: 553768.3125
    Epoch 1888/2000
    8/8 [==============================] - 0s 4ms/step - loss: 556241.0000 - val_loss: 553469.1875
    Epoch 1889/2000
    8/8 [==============================] - 0s 4ms/step - loss: 555937.6875 - val_loss: 552899.6875
    Epoch 1890/2000
    8/8 [==============================] - 0s 4ms/step - loss: 554604.1875 - val_loss: 551743.2500
    Epoch 1891/2000
    8/8 [==============================] - 0s 4ms/step - loss: 553492.5625 - val_loss: 554607.4375
    Epoch 1892/2000
    8/8 [==============================] - 0s 4ms/step - loss: 554348.7500 - val_loss: 555459.6250
    Epoch 1893/2000
    8/8 [==============================] - 0s 4ms/step - loss: 554260.6875 - val_loss: 550601.8125
    Epoch 1894/2000
    8/8 [==============================] - 0s 4ms/step - loss: 553820.5625 - val_loss: 553509.6875
    Epoch 1895/2000
    8/8 [==============================] - 0s 4ms/step - loss: 552302.2500 - val_loss: 553764.0000
    Epoch 1896/2000
    8/8 [==============================] - 0s 4ms/step - loss: 553565.8125 - val_loss: 555733.7500
    Epoch 1897/2000
    8/8 [==============================] - 0s 4ms/step - loss: 552810.4375 - val_loss: 551738.3125
    Epoch 1898/2000
    8/8 [==============================] - 0s 4ms/step - loss: 551972.6875 - val_loss: 556945.0625
    Epoch 1899/2000
    8/8 [==============================] - 0s 4ms/step - loss: 554061.1250 - val_loss: 551812.9375
    Epoch 1900/2000
    8/8 [==============================] - 0s 4ms/step - loss: 552986.8125 - val_loss: 552584.0625
    Epoch 1901/2000
    8/8 [==============================] - 0s 4ms/step - loss: 551544.1875 - val_loss: 554713.6250
    Epoch 1902/2000
    8/8 [==============================] - 0s 4ms/step - loss: 552694.6875 - val_loss: 553281.3750
    Epoch 1903/2000
    8/8 [==============================] - 0s 4ms/step - loss: 552612.9375 - val_loss: 552883.5625
    Epoch 1904/2000
    8/8 [==============================] - 0s 4ms/step - loss: 552506.1250 - val_loss: 551116.4375
    Epoch 1905/2000
    8/8 [==============================] - 0s 4ms/step - loss: 552563.8750 - val_loss: 552652.5625
    Epoch 1906/2000
    8/8 [==============================] - 0s 4ms/step - loss: 553176.9375 - val_loss: 552995.8750
    Epoch 1907/2000
    8/8 [==============================] - 0s 4ms/step - loss: 552323.3125 - val_loss: 551060.6250
    Epoch 1908/2000
    8/8 [==============================] - 0s 4ms/step - loss: 551104.1250 - val_loss: 551228.8750


    Epoch 1909/2000
    8/8 [==============================] - 0s 4ms/step - loss: 553002.6250 - val_loss: 553562.5000
    Epoch 1910/2000
    8/8 [==============================] - 0s 4ms/step - loss: 550072.6875 - val_loss: 552572.6875
    Epoch 1911/2000
    8/8 [==============================] - 0s 4ms/step - loss: 551170.6875 - val_loss: 549317.1875
    Epoch 1912/2000
    8/8 [==============================] - 0s 4ms/step - loss: 550275.8125 - val_loss: 549150.9375
    Epoch 1913/2000
    8/8 [==============================] - 0s 4ms/step - loss: 550419.6875 - val_loss: 551873.2500
    Epoch 1914/2000
    8/8 [==============================] - 0s 4ms/step - loss: 551007.1875 - val_loss: 549316.8750
    Epoch 1915/2000
    8/8 [==============================] - 0s 4ms/step - loss: 550401.7500 - val_loss: 548997.8125
    Epoch 1916/2000
    8/8 [==============================] - 0s 4ms/step - loss: 549887.4375 - val_loss: 548378.5625
    Epoch 1917/2000
    8/8 [==============================] - 0s 4ms/step - loss: 549509.5625 - val_loss: 550903.8125
    Epoch 1918/2000
    8/8 [==============================] - 0s 4ms/step - loss: 549569.2500 - val_loss: 549619.6250
    Epoch 1919/2000
    8/8 [==============================] - 0s 4ms/step - loss: 549490.0000 - val_loss: 548722.6250
    Epoch 1920/2000
    8/8 [==============================] - 0s 4ms/step - loss: 549944.3125 - val_loss: 546362.3750
    Epoch 1921/2000
    8/8 [==============================] - 0s 4ms/step - loss: 548884.9375 - val_loss: 548674.8125
    Epoch 1922/2000
    8/8 [==============================] - 0s 4ms/step - loss: 548091.4375 - val_loss: 549012.8750
    Epoch 1923/2000
    8/8 [==============================] - 0s 4ms/step - loss: 548321.9375 - val_loss: 547775.4375
    Epoch 1924/2000
    8/8 [==============================] - 0s 4ms/step - loss: 548882.3125 - val_loss: 546630.0625
    Epoch 1925/2000
    8/8 [==============================] - 0s 4ms/step - loss: 549220.4375 - val_loss: 548322.4375
    Epoch 1926/2000
    8/8 [==============================] - 0s 4ms/step - loss: 548667.0625 - val_loss: 547735.6875
    Epoch 1927/2000
    8/8 [==============================] - 0s 4ms/step - loss: 548853.1875 - val_loss: 548360.1250
    Epoch 1928/2000
    8/8 [==============================] - 0s 4ms/step - loss: 545887.3125 - val_loss: 550687.3750
    Epoch 1929/2000
    8/8 [==============================] - 0s 4ms/step - loss: 547717.8125 - val_loss: 546970.7500
    Epoch 1930/2000
    8/8 [==============================] - 0s 4ms/step - loss: 548326.7500 - val_loss: 547419.6250
    Epoch 1931/2000
    8/8 [==============================] - 0s 4ms/step - loss: 546417.3750 - val_loss: 546847.5625
    Epoch 1932/2000
    8/8 [==============================] - 0s 4ms/step - loss: 548769.0000 - val_loss: 544262.9375
    Epoch 1933/2000
    8/8 [==============================] - 0s 4ms/step - loss: 547556.0000 - val_loss: 544998.5000
    Epoch 1934/2000
    8/8 [==============================] - 0s 4ms/step - loss: 547938.9375 - val_loss: 546954.3125
    Epoch 1935/2000
    8/8 [==============================] - 0s 4ms/step - loss: 547009.0625 - val_loss: 547842.3750
    Epoch 1936/2000
    8/8 [==============================] - 0s 4ms/step - loss: 545643.3125 - val_loss: 546546.0625
    Epoch 1937/2000
    8/8 [==============================] - 0s 4ms/step - loss: 547559.3125 - val_loss: 544767.6250
    Epoch 1938/2000
    8/8 [==============================] - 0s 4ms/step - loss: 545886.8750 - val_loss: 545222.8750
    Epoch 1939/2000
    8/8 [==============================] - 0s 4ms/step - loss: 546066.9375 - val_loss: 544349.9375
    Epoch 1940/2000
    8/8 [==============================] - 0s 4ms/step - loss: 545693.0000 - val_loss: 547891.4375
    Epoch 1941/2000
    8/8 [==============================] - 0s 4ms/step - loss: 545888.3125 - val_loss: 545706.5000
    Epoch 1942/2000
    8/8 [==============================] - 0s 4ms/step - loss: 545776.0625 - val_loss: 544732.5625
    Epoch 1943/2000
    8/8 [==============================] - 0s 4ms/step - loss: 545932.3125 - val_loss: 544317.6250
    Epoch 1944/2000
    8/8 [==============================] - 0s 4ms/step - loss: 544082.9375 - val_loss: 545592.3125
    Epoch 1945/2000
    8/8 [==============================] - 0s 4ms/step - loss: 543703.5000 - val_loss: 546136.0625
    Epoch 1946/2000
    8/8 [==============================] - 0s 4ms/step - loss: 543777.0000 - val_loss: 544884.3750
    Epoch 1947/2000
    8/8 [==============================] - 0s 4ms/step - loss: 545962.9375 - val_loss: 544982.3750
    Epoch 1948/2000
    8/8 [==============================] - 0s 4ms/step - loss: 546810.6875 - val_loss: 544638.2500
    Epoch 1949/2000
    8/8 [==============================] - 0s 4ms/step - loss: 544809.1250 - val_loss: 545043.6250
    Epoch 1950/2000
    8/8 [==============================] - 0s 4ms/step - loss: 545431.6875 - val_loss: 541958.9375
    Epoch 1951/2000
    8/8 [==============================] - 0s 4ms/step - loss: 542181.3125 - val_loss: 545670.0625
    Epoch 1952/2000
    8/8 [==============================] - 0s 4ms/step - loss: 543629.7500 - val_loss: 543971.6250
    Epoch 1953/2000
    8/8 [==============================] - 0s 4ms/step - loss: 544374.8750 - val_loss: 544881.0625
    Epoch 1954/2000
    8/8 [==============================] - 0s 4ms/step - loss: 543759.1250 - val_loss: 545575.0625
    Epoch 1955/2000
    8/8 [==============================] - 0s 4ms/step - loss: 542542.1250 - val_loss: 544366.7500
    Epoch 1956/2000
    8/8 [==============================] - 0s 4ms/step - loss: 544522.3125 - val_loss: 541707.8125
    Epoch 1957/2000
    8/8 [==============================] - 0s 4ms/step - loss: 543464.3750 - val_loss: 543302.6250
    Epoch 1958/2000
    8/8 [==============================] - 0s 4ms/step - loss: 542013.2500 - val_loss: 541388.5625
    Epoch 1959/2000
    8/8 [==============================] - 0s 4ms/step - loss: 543681.8125 - val_loss: 542855.6250
    Epoch 1960/2000
    8/8 [==============================] - 0s 4ms/step - loss: 542513.2500 - val_loss: 540834.1875
    Epoch 1961/2000
    8/8 [==============================] - 0s 4ms/step - loss: 542284.8125 - val_loss: 544326.3125
    Epoch 1962/2000
    8/8 [==============================] - 0s 4ms/step - loss: 543006.5000 - val_loss: 541047.9375
    Epoch 1963/2000
    8/8 [==============================] - 0s 4ms/step - loss: 540512.4375 - val_loss: 542358.5000
    Epoch 1964/2000
    8/8 [==============================] - 0s 4ms/step - loss: 541284.8750 - val_loss: 541934.0000
    Epoch 1965/2000
    8/8 [==============================] - 0s 4ms/step - loss: 542709.0000 - val_loss: 542559.8125
    Epoch 1966/2000
    8/8 [==============================] - 0s 4ms/step - loss: 542780.6875 - val_loss: 543445.4375
    Epoch 1967/2000
    8/8 [==============================] - 0s 4ms/step - loss: 541156.5625 - val_loss: 541584.1875
    Epoch 1968/2000
    8/8 [==============================] - 0s 4ms/step - loss: 542489.6250 - val_loss: 539056.4375
    Epoch 1969/2000
    8/8 [==============================] - 0s 4ms/step - loss: 541582.7500 - val_loss: 544484.0000
    Epoch 1970/2000
    8/8 [==============================] - 0s 4ms/step - loss: 539640.5000 - val_loss: 540399.6875
    Epoch 1971/2000
    8/8 [==============================] - 0s 4ms/step - loss: 540412.2500 - val_loss: 541135.6250
    Epoch 1972/2000
    8/8 [==============================] - 0s 4ms/step - loss: 540849.0000 - val_loss: 538645.6250
    Epoch 1973/2000
    8/8 [==============================] - 0s 4ms/step - loss: 540238.2500 - val_loss: 541008.7500
    Epoch 1974/2000
    8/8 [==============================] - 0s 4ms/step - loss: 539539.9375 - val_loss: 541824.9375
    Epoch 1975/2000
    8/8 [==============================] - 0s 4ms/step - loss: 539765.5000 - val_loss: 540264.6875
    Epoch 1976/2000
    8/8 [==============================] - 0s 4ms/step - loss: 539684.6875 - val_loss: 541717.6250
    Epoch 1977/2000
    8/8 [==============================] - 0s 4ms/step - loss: 538689.6875 - val_loss: 540888.8750
    Epoch 1978/2000
    8/8 [==============================] - 0s 4ms/step - loss: 538894.5625 - val_loss: 541141.8750
    Epoch 1979/2000
    8/8 [==============================] - 0s 4ms/step - loss: 538563.5625 - val_loss: 536048.6250
    Epoch 1980/2000
    8/8 [==============================] - 0s 4ms/step - loss: 540998.8125 - val_loss: 537510.3750
    Epoch 1981/2000
    8/8 [==============================] - 0s 4ms/step - loss: 537288.0625 - val_loss: 542431.9375
    Epoch 1982/2000
    8/8 [==============================] - 0s 4ms/step - loss: 539615.3125 - val_loss: 540102.2500


    Epoch 1983/2000
    8/8 [==============================] - 0s 4ms/step - loss: 539521.8750 - val_loss: 540367.8125
    Epoch 1984/2000
    8/8 [==============================] - 0s 4ms/step - loss: 538468.9375 - val_loss: 539752.5625
    Epoch 1985/2000
    8/8 [==============================] - 0s 4ms/step - loss: 537164.2500 - val_loss: 538776.0000
    Epoch 1986/2000
    8/8 [==============================] - 0s 4ms/step - loss: 539270.0625 - val_loss: 539925.4375
    Epoch 1987/2000
    8/8 [==============================] - 0s 4ms/step - loss: 538661.3750 - val_loss: 537635.8125
    Epoch 1988/2000
    8/8 [==============================] - 0s 4ms/step - loss: 539571.7500 - val_loss: 539656.6875
    Epoch 1989/2000
    8/8 [==============================] - 0s 4ms/step - loss: 537331.1875 - val_loss: 537959.0625
    Epoch 1990/2000
    8/8 [==============================] - 0s 4ms/step - loss: 537211.8125 - val_loss: 537631.8125
    Epoch 1991/2000
    8/8 [==============================] - 0s 4ms/step - loss: 537210.3125 - val_loss: 535850.0000
    Epoch 1992/2000
    8/8 [==============================] - 0s 4ms/step - loss: 537562.3750 - val_loss: 539526.8125
    Epoch 1993/2000
    8/8 [==============================] - 0s 4ms/step - loss: 536139.4375 - val_loss: 535676.8125
    Epoch 1994/2000
    8/8 [==============================] - 0s 4ms/step - loss: 536837.0000 - val_loss: 537322.5000
    Epoch 1995/2000
    8/8 [==============================] - 0s 4ms/step - loss: 536383.3125 - val_loss: 536216.5625
    Epoch 1996/2000
    8/8 [==============================] - 0s 4ms/step - loss: 538088.3125 - val_loss: 538706.1250
    Epoch 1997/2000
    8/8 [==============================] - 0s 4ms/step - loss: 537721.4375 - val_loss: 534537.4375
    Epoch 1998/2000
    8/8 [==============================] - 0s 4ms/step - loss: 538214.3750 - val_loss: 536923.5000
    Epoch 1999/2000
    8/8 [==============================] - 0s 4ms/step - loss: 536320.0625 - val_loss: 536013.5625
    Epoch 2000/2000
    8/8 [==============================] - 0s 4ms/step - loss: 535522.5000 - val_loss: 535591.5000



```python
print(regressor(tf.convert_to_tensor([1])))
```

    tf.Tensor([[0.]], shape=(1, 1), dtype=float32)



```python
plt.figure(figsize=(15, 10))
plt.scatter(X, y, marker='o', color='gray', alpha=0.3)
y_pred = 0
for _ in range(5):
  y_pred +=  regressor(X)
  plt.plot(X, regressor(X), color='red', alpha=0.6)

plt.plot(X, y_pred/5, color='green', label='mean')
plt.legend()
plt.title('Epistemic uncertainty in regression')
plt.savefig('figs/epistemic_uncertainty_rgn.png')
```


    
![png](output_57_0.png)
    

