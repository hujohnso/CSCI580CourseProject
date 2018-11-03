import autograd.numpy as np
import tensorflow as tf

class analyticalODE1:
    constant = None
    def __init__ (self, x_test, x_train, constant = 1):
        self.x_test = x_test
        self.x_train = x_train
        self.constant = constant
        
    #I didn't feel like repeating the logic for all of the seperate functions
    #so here is a generic evaluator for all of the sub classes to use :)
    def performAnalyticalFunction(self, functionToPerform ,x , constants):
        y = np.linspace(0,0,num = len(x))
        y.shape = len(x), 1
        for i in range(0, len(x)):
            y[i] = functionToPerform(x[i],constants)
        return y

class x2(analyticalODE1):
    def __init__(self, x_test, x_train,constant = 1):
        super().__init__(x_test, x_train,constant)
                         
    def func(self,x):
        return super(x2, self).performAnalyticalFunction(lambda value, constant: constant * (value ** 2 / 2), x, self.constant)
              
    @staticmethod
    def custom_activation(x):
        return x ** 2

class x4(analyticalODE1):
    def __init__(self, x_test, x_train, constant = np.array([1,1,1])):
        super().__init__(x_test, x_train, constant)
    
    def func(self,x):
        return super(x4, self).performAnalyticalFunction(lambda value ,constant:(constant[0] * (value ** 4 / 4)) + constant[1] * (value ** 3 / 3) + constant[2] * (value ** 2 / 2), x, self.constant)
    
    @staticmethod
    def custom_activation(x):
        return x ** 4


class ex(analyticalODE1):
    def __init__(self, x_test, x_train, constant = 1):
        super().__init__(x_test, x_train, constant)

    def func(self, x):
        return super(ex, self).performAnalyticalFunction(lambda value, constant: constant * np.exp(value), x, self.constant)

    @staticmethod
    def custom_activation(x):
        return tf.math.exp(x)


class log(analyticalODE1):
    def __init__(self, x_test, x_train,constant = 1):
        super().__init__(x_test, x_train,constant)

    def func(self, x):
        return super(log, self).performAnalyticalFunction(lambda value,constant: constant * log(value), x, self.constant)
    
    @staticmethod
    def custom_activation(x):
        return tf.log(x)
