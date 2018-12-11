from keras.models import load_model
import numpy as np

model = load_model('savedModel.h5')
y = model.predict(np.array([500]))
np.savetxt('output.dat', y)
