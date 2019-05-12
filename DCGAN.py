from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Dropout, LeakyReLU, Activation, Reshape, BatchNormalization, ReLU, UpSampling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import picLoader as pl
import os
import time


def get_noise(nsample=1, nlatent_dim=100):
    noise = np.random.normal(0, 1, (nsample, nlatent_dim))
    return(noise)

def threshold_binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)



class DCGAN():
    
    
    def __init__(self):
        self.D = None
        self.G = None
        optimizer = Adam(0.00005, 0.5)
        
        self.build_discriminator()
        self.D.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics   = [threshold_binary_accuracy])
        
        self.build_generator()
                
        # The generator takes noise as input and generates imgs
        noiseInput = Input(shape=(100,))
        genOutput = self.G(noiseInput)

        # For the combined model we will only train the generator
        self.D.trainable = False

        # The discriminator takes generated images as input and determines validity
        validityOutput = self.D(genOutput)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.GAN = Model(noiseInput, validityOutput)
        self.GAN.summary()
        self.GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        
    def build_discriminator(self):
        dropout = 0.4
    
        model = Sequential()
        
        input_shape = (64, 64, 3)
        
        # model.add(Conv2D(32, 5, strides=2, input_shape=input_shape, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        # model.add(BatchNormalization(momentum=0.9))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(dropout))
        
        model.add(Conv2D(64, 5, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        
        model.add(Conv2D(128, 5, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        
        model.add(Conv2D(256, 5, strides=1, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        
        # Out: 1-dim probability
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        imageInput = Input(shape=input_shape)
        validityOutput = model(imageInput)
        
        self.D = Model(imageInput,validityOutput)
        
    def build_generator(self):
        dropout = 0.4
    
        model = Sequential()
        
        model.add(Dense(64 * 32 * 32, kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        model.add(Reshape((32, 32, 64)))
        
        # model.add(UpSampling2D(interpolation='bilinear'))
        # # model.add(Conv2DTranspose(512, 5, 2, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        # model.add(Conv2D(256, 5, 1, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        # model.add(BatchNormalization(momentum=0.9))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(dropout))
        
        # model.add(UpSampling2D(interpolation='bilinear'))
        # model.add(Conv2DTranspose(128, 5, 2, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(Conv2D(64, 5, 1, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        
        # model.add(UpSampling2D(interpolation='bilinear'))
        # model.add(Conv2DTranspose(64, 5, 2, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(Conv2DTranspose(64, 5, 2, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(Conv2D(64, 5, 1, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        
        # model.add(UpSampling2D(interpolation='bilinear'))
        model.add(Conv2D(32, 5, 1, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        
        model.add(Conv2DTranspose(3, 5, 1, padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(Activation('tanh'))
        
        noiseInput = Input(shape = (100,))
        imageOutput = model(noiseInput)
        
        self.G = Model(noiseInput, imageOutput)
        print('==============Generator=============')
        model.summary()
        print('====================================')
    
    def train(self, epochs, batch_size=10, save_interval = 5):
        real_imgs = pl.loadData()
        real_imgs = real_imgs / 127.5 - 1
        currEpoch = 0
        halfBatch = int(batch_size/2) # Half of the batch to train D and G equally
        noiseForPlot = np.random.normal(0, 1, (25, 100))
        for epoch in range(epochs):
            currEpoch = epoch # For saving models
        
            noiseForDiscriminator = np.random.normal(0, 1, (halfBatch, 100))
            
            valid = np.random.uniform(0.9, 1.0, (halfBatch,1))
            fake = np.random.uniform(0.0, 0.1, (halfBatch,1))
            
            mask = np.random.choice(2,(halfBatch,1), p=[0.95,0.05]).astype(bool)
            temp=np.copy(valid)
            valid[mask] = fake[mask]
            fake[mask] = temp[mask]
            
            idx = np.random.randint(0, real_imgs.shape[0], halfBatch)
            real_imgs_batch = real_imgs[idx]
            gen_imgs_batch = self.G.predict(noiseForDiscriminator)

            d_loss_real = self.D.train_on_batch(real_imgs_batch, fake)
            d_loss_fake = self.D.train_on_batch(gen_imgs_batch, valid)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            noiseForCombined = np.random.normal(0, 1, (batch_size, 100))

            mask = np.random.choice(2,(batch_size,1), p=[0.95,0.05]).astype(bool)
            valid_for_gen =  np.random.uniform(0.0, 0.1, (batch_size,1))
            valid_for_gen[mask] = np.random.uniform(0.9, 1.0, len(valid_for_gen[mask]))
            
            g_loss = self.GAN.train_on_batch(noiseForCombined, valid_for_gen)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            
            if epoch % save_interval == 0:
                # noise = noiseVector #np.random.normal(0, 1, (r * v, c.NOISE_SIZE))
                # gen_imgs = self.G.predict(noise)
                self.save_imgs(currEpoch, noiseForPlot)
            if os.path.exists('stop.txt'):
                break
        model_json = self.G.to_json()
        with open(str(currEpoch) + "model" + str(time.time()) + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.G.save_weights(str(currEpoch) + "model" + str(time.time()) + ".h5")        
        self.G.save(str(currEpoch) + "modelSave" + str(time.time()) + ".h5")
    
    
    
    def save_imgs(self, epoch, noiseVector):
        r, v = 5, 5
        noise = noiseVector #np.random.normal(0, 1, (r * v, c.NOISE_SIZE))
        gen_imgs = self.G.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, v)
        cnt = 0
        for i in range(r):
            for j in range(v):
                # axs[i,j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%06d_gen.png" % epoch)
        plt.close()
    

if __name__ == '__main__':
    a = DCGAN()
    # c = a.G.predict(get_noise())
    # print(c)
    # print(c.shape)
    a.train(999999, 128, 10)    
        
        
        
        
        
        
        
        
        
        
        
        
        