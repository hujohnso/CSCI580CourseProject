from keras.models import load_model
import numpy as np

model = load_model('savedModel.h5')
y = model.predict(np.array([2]))
np.savetxt('output.dat', y)
