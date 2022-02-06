# Constructing a Convolutional Neural Network
It is possible to train a CNN to learn how to interpret a singlet yield signal and predict the heading direction corresponding to that signal. The topology of the network and hyperparameters are pre-determined, and the only parameter the user has to specify is the size of the representation of the signal across the retina
```python
CNN = spin_dynamics.ConvolutionalModel(receptor_grid, path=None, domain=1.0)
```
```
receptor_grid : int
    Number of receptors in each dimension in a square grid (Note, this
    must match the number of receptors used to define the RetinaSignal                               
    object used to generate the training data)
path : str, optional
    Directory from which to read in model, if None the model will be built
    from scratch (Default: None)
domain : float, optional
    Fraction of domain of angles to sample over (Default: 1.0)
```
If you want to know the details of the network, such as layers and number of parameters, you can call the `summary()` function
```python
CNN.summary()
```

## Generate training, development and test data sets
In order to generate the data sets needed for training and evaluating the network, we need to have first specified the representation of the signal using a RetinaSignal object (see heading_accuracy docs for more info). Then it is simply a matter of specifying the size of the three data sets
```python
CNN.training_data(
	retina_signal, Ntrain=10000, Ndev=1000,                                      
        ntrajectories=None, batch_size=64
	)
```
```
retina_signal : RetinaSignal                                                                     
    Class object for generating the signal of the singlet yield
    across the retina
Ntrain : int, optional                                                                           
    Size of the training set (Default: 10000)                                                    
Ndev : int, optional
    Size of the development set (Default: 1000)
ntrajectories : int, optional                                                                    
    Number of photocycles over which to average the signal, if                                   
    None give the infinite limit (Default: None)                                                 
batch_size : int, optional
    Batch size for training the CNN (Default: 64)
```
```python
CNN.test_data(retina_signal, Ntest=1000, ntrajectories=None)
```
```
retina_signal : RetinaSignal
    Class object for generating the signal of the singlet yield
    across the retina
Ntest : int, optional
    Size of the test set (Default: 1000)
ntrajectories : int, optional
    Number of photocycles over which to average the signal, if
    None give the infinite limit (Default: None)
```

## Training and evaluating the model
The final step is to train the model and evaluate its performance (NOTE: if you're trying to train the model with a very noisy signal, it is advisable to iteratively train the weights with fewer and fewer photocycles)
```python
CNN.train_model(epochs=4):
```
```
epochs : int, optional
    Number of epochs over which to train (Default: 4)
```
```python
CNN.evalute()
```
The result of the `evaluate()` function is the root mean squared error of the network predicting the heading direction, in degrees.

## Saving and restarting the model
The model can be saved using
```python
CNN.save_model('my_model')
```
```
path : str
    Directory to which to write the model parameters
```
This can then be read into a new instantiation of the ConvolutionalModel object
```python
CNN_new = spin_dynamics.ConvolutionalModel(receptor_grid, path='my_model')
```
