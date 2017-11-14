from tf_utils import ResNet50
from resnets_utils import load_dataset, preprocess_data, plot_model_history

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

# load dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# preprocess dataset
X_train, Y_train, X_test, Y_test = preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)

# generate model
model = ResNet50(input_shape = (64, 64, 3), classes = len(classes))

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit model
model_history = model.fit(X_train, Y_train, epochs = 20, batch_size = 32)

# plot accuracy/loss
plot_model_history(model_history)

# evaluate model
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))