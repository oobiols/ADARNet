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
    self.kernel = kernel
 
    for f in filters:
        self.encoder.append(tf.keras.layers.Conv2D(filters=f,
							kernel_size = kernel,
							activation = tf.nn.leaky_relu,
							padding = 'same',
							strides=1))

    for f in reversed(filters):
        self.decoder.append(tf.keras.layers.Conv2DTranspose(filters=f,
								kernel_size = kernel,
								activation = tf.nn.leaky_relu,
								padding = 'same',
								strides=1))

    self.output_layer = tf.keras.layers.Conv2DTranspose(filters=3,
								kernel_size = kernel,
								activation = tf.nn.leaky_relu,
								padding = 'same',
								strides=1)
  def call(self, inputs):

    flowvar = inputs[0]
    xz = inputs[1]
   
    x = tf.concat([flowvar,xz],axis=-1)


    for e in self.encoder:
      x = e(x)

    for d in self.decoder:
      x = d(x)

    x = self.output_layer(x)

    return x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]

  def train_step(self, data):

    inputs = data[0]
    labels = data[1]

    flowvar = inputs[:,:,:,0:3]
    xz = inputs[:,:,:,3:] 

    with tf.GradientTape(persistent=True) as tape0:
        
      uMse, vMse, pMse, cont_loss, momx_loss, momy_loss = self.compute_loss(flowvar, labels, xz)

      data_loss  = (1/3)*(uMse   + vMse + pMse)
     
      loss = data_loss + self.beta[0]*cont_loss + self.beta[1]* momx_loss + self.beta[2] * momy_loss
      loss = loss / strategy.num_replicas_in_sync

    lossGrad = tape0.gradient(loss, self.trainable_variables)
    del tape0

    # ---- update parameters ---- #
    self.optimizer.apply_gradients(zip(lossGrad, self.trainable_variables))

    # ---- update metrics and statistics ---- #
    # track loss and mae
    self.trainMetrics['loss'].update_state(loss*strategy.num_replicas_in_sync)
    self.trainMetrics['data_loss'].update_state(data_loss)
    self.trainMetrics['cont_loss'].update_state(cont_loss)
    self.trainMetrics['momx_loss'].update_state(momx_loss)
    self.trainMetrics['momy_loss'].update_state(momy_loss)
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()
    return self.trainStat

  def compute_loss(self, flowvar, labels, xz):

    xz = xz    
    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(xz)

      with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(xz)
        upred, vpred, ppred = self([flowvar,xz])

      u_grad = tape1.gradient(upred,xz)
      v_grad = tape1.gradient(vpred,xz)
      p_grad = tape1.gradient(ppred,xz)

      u_x, u_z = u_grad[:,:,:,0], u_grad[:,:,:,1]
      v_x, v_z = v_grad[:,:,:,0], v_grad[:,:,:,1]
      p_x, p_z = p_grad[:,:,:,0], p_grad[:,:,:,1]

      del tape1

    u_xx = tape2.gradient(u_x, xz)[:,:,:,0]
    u_zz = tape2.gradient(u_z, xz)[:,:,:,1]
    v_xx = tape2.gradient(v_x, xz)[:,:,:,0]
    v_zz = tape2.gradient(v_z, xz)[:,:,:,1]

    del tape2

    uMse, vMse, pMse = self.compute_data_loss(labels,upred,vpred,ppred)

    contMse, momxMse, momzMse = self.compute_pde_loss(upred,vpred, u_x,u_z,v_x,v_z,p_x,p_z,u_xx,u_zz,v_xx,v_zz)

    return uMse, vMse, pMse, contMse, momxMse, momzMse

  def compute_data_loss(self,labels,upred,vpred,ppred):

    uMse = tf.reduce_mean(tf.square(labels[:,:,:,0]-upred))
    vMse = tf.reduce_mean(tf.square(labels[:,:,:,1]-vpred))
    pMse = tf.reduce_mean(tf.square(labels[:,:,:,2]-ppred))

    return uMse, vMse, pMse

  def compute_pde_loss(self,upred,vpred, u_x,u_z,v_x,v_z,p_x,p_z,u_xx,u_zz,v_xx,v_zz):

      re = 0.1
      contMse = tf.reduce_mean(tf.square(u_x+v_z))
      momx    = upred * u_x + vpred*u_z + p_x - re*(u_xx +u_zz)
      momxMse = tf.reduce_mean(tf.square(momx))
     
      momy = upred * v_x + vpred*v_z + p_z - re*(v_xx + v_zz)
      momyMse = tf.reduce_mean(tf.square(momy))

      return contMse, momxMse, momyMse

  def test_step(self, data):

    inputs = data[0]
    flowvar = inputs[:,:,:,:3]
    xz = inputs[:,:,:,3:]

    labels = data[1] 

    uMse, vMse, pMse, cont_loss, momx_loss, momy_loss = self.compute_loss(flowvar,labels,xz)

    data_loss  = (1/3)*(uMse   + vMse + pMse)

    loss = data_loss + self.beta[0]*cont_loss + self.beta[1]*momx_loss + self.beta[2]*momy_loss
    
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['data_loss'].update_state(data_loss)
    self.validMetrics['cont_loss'].update_state(cont_loss)
    self.validMetrics['momx_loss'].update_state(momx_loss)
    self.validMetrics['momy_loss'].update_state(momy_loss)
 #   self.validMetrics['nuMse'].update_state(nuMse)

    for key in self.validMetrics:
      self.validStat[key] = self.validMetrics[key].result()
    return self.validStat

