import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

class ConvolutionalModel:
    '''Build a convolutional neural network to learn the mapping between the
    singlet yield signal and the corresponding heading direction.

    Parameters
    ----------
    receptor_grid : int
        Number of receptors in each dimension in a square grid (Note, this 
        must match the number of receptors used to define the RetinaSignal 
        object used to generate the training data)
    path : str, optional
        Directory from which to read in model, if None the model will be built
        from scratch (Default: None)
    domain : float, optional
        Fraction of domain of angles to sample over (Default: 1.0)
'''

    def __init__(self, receptor_grid, path=None, domain=1.0):

        self.xdim = receptor_grid
        self.domain = domain

        if path is not None:
            self.model = tf.keras.models.load_model(path)
        else:
            self.model = self.conv_model((self.xdim,self.xdim,1))

        self.model.compile(optimizer='adam',
                                loss=self.wrapped_loss,
                                metrics=[self.wrapped_loss])

    def summary(self):
        '''Return the model summary
'''
        return self.model.summary()

    def training_data(self, retina_signal, Ntrain=10000, Ndev=1000,
                      ntrajectories=None, batch_size=64):
        '''Get training and development data sets for number of photo
        cycles given by Ntrajectories. Each data point is the singlet yield
        signal for a different heading direction

        Parameters
        ----------
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
'''
        xtrain = np.zeros((Ntrain, self.xdim, self.xdim))
        ytrain = np.zeros(Ntrain)

        for i in range(Ntrain):
            ytrain[i] = np.random.uniform(-1.0, 1.0)
            xtrain[i] = retina_signal.generate_signal(
                                heading=ytrain[i]*np.pi*self.domain,
                                normalize=True,
                                ntrajectories=ntrajectories)

        xdev = np.zeros((Ndev, self.xdim, self.xdim))
        ydev = np.zeros(Ndev)

        for i in range(Ndev):
            ydev[i] = np.random.uniform(-1.0, 1.0)
            xdev[i] = retina_signal.generate_signal(
                                heading=ydev[i]*np.pi*self.domain,
                                normalize=True,
                                ntrajectories=ntrajectories)

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
                                (xtrain, ytrain)).batch(batch_size)
        self.dev_dataset = tf.data.Dataset.from_tensor_slices(
                                (xdev, ydev)).batch(batch_size)

    def test_data(self, retina_signal, Ntest=1000, ntrajectories=None):
        '''Get test data set for number of photo cycles given by Ntrajectories.
        Each data point is the singlet yield signal for a different heading
        direction

        Parameters
        ----------
        retina_signal : RetinaSignal
            Class object for generating the signal of the singlet yield
            across the retina
        Ntest : int, optional
            Size of the test set (Default: 1000)
        ntrajectories : int, optional
            Number of photocycles over which to average the signal, if
            None give the infinite limit (Default: None)
'''
        self.xtest = np.zeros((Ntest, self.xdim, self.xdim))
        self.ytest = np.zeros(Ntest)

        for i in range(Ntest):
            self.ytest[i] = np.random.uniform(-1.0, 1.0)
            self.xtest[i] = retina_signal.generate_signal(
                                heading=self.ytest[i]*np.pi*self.domain,
                                normalize=True,
                                ntrajectories=ntrajectories)

    def train_model(self, epochs=4):
        '''Train the CNN

        Parameters
        ----------
        epochs : int, optional
            Number of epochs over which to train (Default: 4)
'''
        self.history = self.model.fit(self.train_dataset, epochs=epochs,
                                      validation_data=self.dev_dataset)

    def evaluate(self):
        '''Evaluate accuracy of model on test set to give the mean
        squared error in degrees
'''
        test_scores = self.model.evaluate(self.xtest, self.ytest, verbose=2)
        return np.sqrt(test_scores[1]*180*180)

    def save_model(self, path):
        '''Save the model to file

        Parameters
        ----------
        path : str
            Directory to which to write the model parameters
'''

        self.model.save(path, save_traces=False, include_optimizer=False)

    @staticmethod
    def wrapped_loss(y, yhat):
        '''Define loss function to account for the symmetry in angles'''
        ydiff = y-yhat
        y1 = ydiff - 2.0*float(int(ydiff))
        return y1*y1

    @staticmethod
    def conv_model(input_shape, training=True):
        '''Construct a convolutional neural network consisting of two convolutional
        layers each followed by a batch normalization and ReLU activation, and a
        max pool layer before the final dense layer with a single node

        Input -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool -> Dense
'''
        input_img = tf.keras.Input(shape=input_shape)
        Z1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=3, 
             kernel_regularizer='l2')(input_img)
        B1 = tf.keras.layers.BatchNormalization(axis=3)(Z1, training=training)
        A1 = tf.keras.layers.ReLU()(B1)

        Z2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid')(A1)
        B2 = tf.keras.layers.BatchNormalization(axis=3)(Z2, training=training)
        A2 = tf.keras.layers.ReLU()(B2)
        P2 = tf.keras.layers.MaxPool2D(pool_size=3, strides=3)(A2)

        F = tf.keras.layers.Flatten()(P2)
        outputs = tf.keras.layers.Dense(units=1, kernel_regularizer='l2',
                                        activation='tanh')(F)

        conv_model = tf.keras.Model(inputs=input_img, outputs=outputs)
        return conv_model

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

def read_in_model(json):
    return tf.keras.models.model_from_json(json)

#-----------------------------------------------------------------------------#

#Allow GPU memory to grow
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
