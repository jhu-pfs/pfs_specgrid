from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras import optimizers


def train_dnn(nn_input, nn_labels, model, epochs=1, steps=1, learning_rate=0, loss='mean_squared_error', patience=0):
    sgd = optimizers.SGD(lr=learning_rate, decay=0, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=sgd, metrics=[loss])
    print(model.summary())

    best_model = ModelCheckpoint('best_model', save_best_only=True, verbose=1)
    #early_stop = EarlyStopping(patience=patience, verbose=1)
    #learning_rate = LearningRateScheduler(sdecay)

    #model.fit_generator(dg_train, nb_epoch=args.n_epochs,
    #                    steps_per_epoch=n_steps,
    #                    validation_data=dg_test,
    #                    validation_steps=dg_test.n_steps,
    #                    callbacks=[best_model],
    #                    verbose=2)

    callback_hist = model.fit(nn_input, nn_labels,
                              epochs=epochs,
                              validation_split=0.3,
                              shuffle=True,
                              callbacks=[best_model],
                              verbose=1)

    #model.save_weights('results/cnn_model_' + RUN_NAME + '.p')  # save the model
    #model.load_weights('results/cnn_model_' + RUN_NAME + '.p')  # load the model