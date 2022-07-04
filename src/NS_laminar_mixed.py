import tensorflow as tf
from tensorflow import keras
import numpy as np
from NS_model import NSModelPinn

strategy = tf.distribute.MirroredStrategy()

class NSlaminar(NSModelPinn):
  def __init__(self, 
               filters= [16,32,128],
               kernel = 3,
               **kwargs):

    super(NSlaminar, self).__init__(**kwargs)
    self.encoder = []
    self.decoder = []
    self.activation_enc = [] 
    self.activation_dec = [] 
    self.batchnorm_enc = []
    self.batchnorm_dec = []
    self.kernel = kernel
    self.leaky = tf.keras.layers.Activation(tf.nn.leaky_relu,dtype=tf.float32)

    for f in filters:
        self.activation_enc.append(tf.keras.layers.Activation('tanh'))
      #  self.batchnorm_enc.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.encoder.append(tf.keras.layers.Conv2D(filters=f,
							kernel_size = kernel,
							padding = 'same',
							strides=1))

    for f in reversed(filters):
        self.activation_dec.append(tf.keras.layers.Activation('tanh'))
       # self.batchnorm_dec.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.decoder.append(tf.keras.layers.Conv2DTranspose(filters=f,
								kernel_size = kernel,
								padding = 'same',
								strides=1))

    self.output_layer = tf.keras.layers.Conv2DTranspose(filters=3,
								kernel_size = kernel,
								padding = 'same',
								strides=1)
  def call(self, inputs):

    flowvar = inputs[0]
    xz = inputs[1]
   
    x = tf.concat([flowvar,xz],axis=-1)

    for i in range(len(self.encoder)):
      x = self.encoder[i](x)
      x = self.activation_enc[i](x)

    for i in range(len(self.decoder)):
      x = self.decoder[i](x)
      x = self.activation_dec[i](x)

    x = self.output_layer(x)
    x = self.leaky(x)

    return x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]

  def train_step(self, data):

    inputs = data[0]
    labels = data[1]

    flowvar = inputs[:,:,:,0:4]
    xz = inputs[:,:,:,4:] 

    with tf.GradientTape(persistent=True) as tape0:
        
      uMse, vMse, pMse, cont_loss = self.compute_loss(flowvar, labels, xz)

      data_loss  = (1/3)*(uMse   + vMse + pMse)
     
      loss = data_loss + self.beta[0]*cont_loss
      loss = loss / strategy.num_replicas_in_sync
      scaled_loss = self.optimizer.get_scaled_loss(loss)

    lossGrad = tape0.gradient(scaled_loss, self.trainable_variables)
    lossGrad = self.optimizer.get_unscaled_gradients(lossGrad)
    del tape0

    # ---- update parameters ---- #
    self.optimizer.apply_gradients(zip(lossGrad, self.trainable_variables))

    # ---- update metrics and statistics ---- #
    # track loss and mae
    self.trainMetrics['loss'].update_state(loss*strategy.num_replicas_in_sync)
    self.trainMetrics['data_loss'].update_state(data_loss)
    self.trainMetrics['cont_loss'].update_state(cont_loss)
    self.trainMetrics['uMse'].update_state(uMse)
    self.trainMetrics['vMse'].update_state(vMse)
    self.trainMetrics['pMse'].update_state(pMse)
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()
    return self.trainStat

  def compute_loss(self, flowvar, labels, xz):

    xz = xz    

    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
      tape1.watch(xz)
      upred, vpred, ppred = self([flowvar,xz])

    u_grad = tape1.gradient(upred,xz)
    v_grad = tape1.gradient(vpred,xz)
    p_grad = tape1.gradient(ppred,xz)

    u_x = u_grad[:,:,:,0]
    v_z = v_grad[:,:,:,1]

    del tape1

    uMse, vMse, pMse = self.compute_data_loss(labels,upred,vpred,ppred)

    contMse  = self.compute_pde_loss(u_x,v_z)

    return uMse, vMse, pMse, contMse

  def compute_data_loss(self,labels,upred,vpred,ppred):


    uMse = tf.reduce_mean(tf.square(labels[:,:,:,0]-upred))
    vMse = tf.reduce_mean(tf.square(labels[:,:,:,1]-vpred))
    pMse = tf.reduce_mean(tf.square(labels[:,:,:,2]-ppred))

    return uMse, vMse, pMse

  def compute_pde_loss(self,u_x,v_z):

      contMse = tf.reduce_mean(tf.square(u_x+v_z))

      return contMse

  def test_step(self, data):

    inputs = data[0]
    flowvar = inputs[:,:,:,:4]
    xz = inputs[:,:,:,4:]

    labels = data[1] 

    uMse, vMse, pMse, cont_loss = self.compute_loss(flowvar,labels,xz)

    data_loss  = (1/3)*(uMse   + vMse + pMse)

    loss = data_loss + self.beta[0]*cont_loss
    
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['data_loss'].update_state(data_loss)
    self.validMetrics['cont_loss'].update_state(cont_loss)
    self.validMetrics['uMse'].update_state(uMse)
    self.validMetrics['vMse'].update_state(vMse)
    self.validMetrics['pMse'].update_state(pMse)

    for key in self.validMetrics:
      self.validStat[key] = self.validMetrics[key].result()
    return self.validStat

