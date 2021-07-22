from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard

import noisy as ns

# Loading Dataset
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# plt.imshow(x_train[0], cmap = "gray")
# plt.show()


# Applying the noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Show the noisy digits
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()
plt.savefig('noisy_digits.png')

#  Loading Model
input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# At this point the representation is (7, 7, 32)

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Starting the training of the autoencoder

""" training_history = autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))
                #callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)]) """

# autoencoder.save('autoencoder.h5')

x_test_noise_poisson = np.copy(x_test)
for i in range(len(x_test_noise_poisson)):
    img_noise = ns.noisy('poisson', x_test[i])
    x_test_noise_poisson[i] = img_noise

x_test_noise_speckle = np.copy(x_test)
for i in range(len(x_test_noise_speckle)):
    img_noise = ns.noisy('speckle', x_test[i])
    x_test_noise_speckle[i] = img_noise

x_test_noise_SP = np.copy(x_test)
for i in range(len(x_test_noise_SP)):
    img_noise = ns.noisy('s&p', x_test[i])
    x_test_noise_SP[i] = img_noise

autoencoder.load_weights('autoencoder.h5')

# gaussian noise
test_data_denoised_G = autoencoder.predict(x_test_noisy)
# print(np.shape(x_test_noisy))
# print(np.shape(test_data_denoised_G))

# poisson noise
# img_poisson_noised = ns.noisy('poisson',x_test[0])
test_data_denoised_P = autoencoder.predict(x_test_noise_poisson)
# print(np.shape(test_data_denoised_P))

# speckle noise
# img_speckle_noised = ns.noisy('speckle',x_test[0])
test_data_denoised_S = autoencoder.predict(x_test_noise_speckle)

# s&p noise
# img_sp_noised = ns.noisy('s&p',x_test[0])
test_data_denoised_SP = autoencoder.predict(x_test_noise_SP)
# print(np.shape(img_sp_noised))


# We will calculate the mean squared error of the whole test set.
# First we will calculate the mse between our clean data and the data with added noise.
# Next we check how well our DAE (denoising autoencoder) denoised the data.

def mse(data_clean, data_noisy, data_denoised):
    noisy_clean_mse = np.square(np.subtract(data_clean, data_noisy)).mean()
    denoised_clean_mse = np.square(np.subtract(data_denoised, data_clean)).mean()
    if denoised_clean_mse < noisy_clean_mse:
        percent_diff = (100 - ((denoised_clean_mse / noisy_clean_mse) * 100))
        print(noisy_clean_mse, denoised_clean_mse)
        print("DAE decreases the noise around {:.0f}%".format(percent_diff))
    else:
        percent_diff = (100 - ((noisy_clean_mse / denoised_clean_mse) * 100))
        print(noisy_clean_mse, denoised_clean_mse)
        # the noisy image is closer to the clean image than the DAE reconstructed image
        print("DAE shows a deterioration in image reconstruction of about {:.0f}%".format(percent_diff))


mse(x_test, x_test_noisy, test_data_denoised_G)
mse(x_test, x_test_noise_poisson, test_data_denoised_P)
mse(x_test, x_test_noise_speckle, test_data_denoised_S)
mse(x_test, x_test_noise_SP, test_data_denoised_SP)


idx = 88
################################
plt.subplot(1, 3, 1)
plt.imshow(x_test[idx])
plt.title('original')

plt.subplot(1, 3, 2)
plt.imshow(x_test_noisy[idx])
plt.title('gaussian_noise')

plt.subplot(1, 3, 3)
plt.imshow(test_data_denoised_G[idx])
plt.title('denoised_G')

plt.tight_layout()
plt.savefig('noise_denoised_G')
#################################

##################################
plt.subplot(1, 3, 1)
plt.imshow(x_test[idx])
plt.title('original')

plt.subplot(1, 3, 2)
plt.imshow(x_test_noise_poisson[idx])
plt.title('poisson_noise')

plt.subplot(1, 3, 3)
plt.imshow(test_data_denoised_P[idx])
plt.title('denoised_P')

plt.tight_layout()
plt.savefig('noise_denoised_P')
######################################


######################################
plt.subplot(1, 3, 1)
plt.imshow(x_test[idx])
plt.title('original')

plt.subplot(1, 3, 2)
plt.imshow(x_test_noise_speckle[idx])
plt.title('speckle_noise')

plt.subplot(1, 3, 3)
plt.imshow(test_data_denoised_S[idx])
plt.title('denoised_S')

plt.tight_layout()
plt.savefig('noise_denoised_S')

#######################################

#######################################
plt.subplot(1, 3, 1)
plt.imshow(x_test[idx])
plt.title('original')

plt.subplot(1, 3, 2)
plt.imshow(x_test_noise_SP[idx])
plt.title('salt_pepper_noise')

plt.subplot(1, 3, 3)
plt.imshow(test_data_denoised_SP[idx])
plt.title('denoised_SP')

plt.tight_layout()
plt.savefig('noise_denoised_SP')

########################################
