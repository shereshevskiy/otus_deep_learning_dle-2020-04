import numpy as np
from keras import backend as K
from keras.activations import softmax
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from keras.objectives import binary_crossentropy as bce

from gumbel_trick.show_images import plot_epoch

batch_size = 100
data_dim = 784
categorical_dim = 10
latent_dim = 30
nb_epoch = 300
epsilon_std = 0.01

anneal_rate = 0.0003
min_temperature = 0.5


def build_encoder(init_temp=5.0,
                  categorical_dim=categorical_dim,
                  latent_dim=latent_dim,
                  batch_size=batch_size,
                  data_dim=data_dim
                  ):
    tau = K.variable(init_temp, name="temperature")
    x = Input(batch_shape=(batch_size, data_dim))
    h = Dense(256, activation='relu')(Dense(512, activation='relu')(x))
    logits_y = Dense(categorical_dim * latent_dim)(h)
    return x, logits_y, tau


def sampling(logits_y, tau):
    U = K.random_uniform(K.shape(logits_y), 0, 1)
    y = logits_y - K.log(-K.log(U + K.epsilon()) + K.epsilon())  # logits + gumbel noise
    y = softmax(K.reshape(y, (-1, latent_dim, categorical_dim)) / tau)
    y = K.reshape(y, (-1, latent_dim * categorical_dim))
    return y


def build_generator(logits_y, tau):
    z = Lambda(lambda x: sampling(x, tau),
               output_shape=(categorical_dim * latent_dim,))(logits_y)
    generator = Sequential()
    generator.add(Dense(256, activation='relu', input_shape=(latent_dim * categorical_dim,)))
    generator.add(Dense(512, activation='relu'))
    generator.add(Dense(data_dim, activation='sigmoid'))
    x_hat = generator(z)
    return generator, x_hat


def build_vae(x, x_hat, logits_y):
    def gumbel_loss(x, x_hat):
        q_y = K.reshape(logits_y, (-1, latent_dim, categorical_dim))
        q_y = softmax(q_y)
        log_q_y = K.log(q_y + 1e-20)
        kl_tmp = q_y * (log_q_y - K.log(1.0 / categorical_dim))
        KL = K.sum(kl_tmp, axis=(1, 2))
        elbo = data_dim * bce(x, x_hat) - KL
        return elbo

    vae = Model(x, x_hat)
    vae.compile(optimizer='adam', loss=gumbel_loss)
    return vae


x, logits_y, tau = build_encoder(5.0)
generator, x_hat = build_generator(logits_y, tau)

vae = build_vae(x, x_hat, logits_y)


def prepare_data_vae():
    def ssl_fix():
        import requests
        import ssl
        requests.packages.urllib3.disable_warnings()

        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            # Legacy Python that doesn't verify HTTPS certificates by default
            pass
        else:
            # Handle target environment that doesn't support HTTPS verification
            ssl._create_default_https_context = _create_unverified_https_context
    ssl_fix()


    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return (x_train, x_test), (y_train, y_test)


(x_train, x_test), _ = prepare_data_vae()

if __name__ == '__main__':
    for e in range(nb_epoch):
        vae.fit(x_train, x_train,
                shuffle=True,
                nb_epoch=1,
                batch_size=batch_size,
                validation_data=(x_test, x_test))
        K.set_value(tau, np.max([K.get_value(tau) * np.exp(-anneal_rate * e), min_temperature]))
        if e % 10 == 0:
            plot_epoch(e, generator, categorical_dim, latent_dim)

    # argmax_y = K.max(K.reshape(logits_y, (-1, latent_dim, categorical_dim)), axis=-1, keepdims=True)
    # argmax_y = K.equal(K.reshape(logits_y, (-1, latent_dim, categorical_dim)), argmax_y)
    # encoder = K.function([x], [argmax_y, x_hat])
    # code, x_hat_test = encoder([x_test[:100]])
