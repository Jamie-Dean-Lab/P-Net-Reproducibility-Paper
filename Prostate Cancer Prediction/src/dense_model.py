import logging
from keras import Input, Model
from keras.layers import Dense
import numpy as np

def compile_dense(optimizer, loss, h_reg, data, n_weights, h_activation, o_activation, n_hidden_layers):
    x = data.xs
    y = data.ys
    info = data.ids
    features = data.get_features()

    logging.info(
        "x shape {} , y shape {} info {} genes {}".format(
            x.shape, y.shape, len(info), len(features)
        )
    )

    feature_names = []
    feature_names.append(features)

    n_features = x.shape[1]

    ins = Input(shape=(n_features,), dtype="float32", name="inputs")
    n = np.ceil(float(n_weights) / float(n_features))

    layer1 = Dense(
        units=int(n), activation=h_activation, kernel_regularizer=h_reg, name="h0"
    )
    outcome = layer1(ins)
    outcome = Dense(1, activation=o_activation, name="output")(outcome)
    model = Model(inputs=[ins], outputs=outcome)

    model.compile(optimizer=optimizer, loss=loss)
    logging.info("done compiling")

    logging.info(model.summary())
    logging.info("# of trainable params of the model is %s" % model.count_params())
    return model, feature_names