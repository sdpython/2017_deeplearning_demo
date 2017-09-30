"""
Defines user functions to add a custom gradient.
"""
import numpy
import os
from PIL import Image
from cntk.ops.functions import UserFunction
from cntk import output_variable


class PinkActivation(UserFunction):
    def __init__(self, arg, image, channels, width, height, beta=0.001, name='PinkActivation'):
        super(PinkActivation, self).__init__([arg], name=name)

        self.myname = name
        self.mysize = (width, height)
        self.img = Image.open(image)
        self.img = self.img.resize((width,height))
        self.array = numpy.array(self.img.getdata()).reshape(self.img.size[0], self.img.size[1], channels)        
        self.rgbArray = numpy.zeros((channels, height,width), 'float32')
        for ic in range(channels):
            self.rgbArray[2-ic,:,:] = self.array[:,:, ic] / 255.0
        self.rgbArray *= 0.0090625            
        self.rgbArray = self.rgbArray.reshape((1, channels, width,height))
        self.beta = beta

    def forward(self, argument, device=None, outputs_to_retain=None):
        # sigmoid_x = 1 / (1 + np.exp(-argument))
        # return sigmoid_x, sigmoid_x
        return argument, argument

    def backward(self, state, root_gradients):
        # sigmoid_x = state
        # return root_gradients * sigmoid_x * (1 - sigmoid_x)
        # We add our custom gradient.
        for x, grad in zip(state, root_gradients):
            if x.shape != grad.shape:
                raise ValueError("Not the same dimension {0} != {1}".format(x.shape, grad.shape))
            if x.shape != self.rgbArray.shape:
                raise ValueError("Not the same dimension {0} != {1}".format(x.shape, self.rgbArray.shape))
            diff = (x - self.rgbArray)
            grad -= diff * self.beta
        return root_gradients        

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype,
            self.inputs[0].dynamic_axes)]

    @staticmethod
    def deserialize(inputs, name, state):
        return PinkActivation(inputs[0], name, self.myname, self.mysize[0], self.mysize[1])

        
if __name__ == "__main__":
    name = os.path.join(os.path.dirname(__file__), "data", "pink_elephant.jpg")
    fct = PinkActivation(None, name, 200, 160, "pink")
    print(fct.mysize)
    