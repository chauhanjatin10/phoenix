import numpy as np
import sys

def generate_dataset(output_dim=10,examples=10000):
	def int2vec(x,dim=output_dim):
		out = np.zeros(dim)
		binrep = np.array(list(np.binary_repr(x))).astype('int')
		out[-len(binrep):] = binrep
		#print(out)
		return out

	x_left_int = (np.random.rand(examples)*2**(output_dim-1)).astype('int')
	x_right_int = (np.random.rand(examples)*2**(output_dim-1)).astype('int')
	y_int = x_left_int+x_right_int
	#print(x_left_int.shape)
	x = list()
	for i in range(len(x_left_int)):
		x.append(np.concatenate((int2vec(x_left_int[i]),int2vec(x_right_int[i]))))

	y = list()
	for i in range(len(y_int)):
		y.append(int2vec(y_int[i]))

	x = np.array(x)
	y = np.array(y)

	return x,y

def sigmoid(a):
	return 1/(1+np.exp(-a))

def sigmoid_derivative(a):
	return a*(1-a)

np.random.seed(1)

examples=10000
output_dim = 10
iterations = 1000

x,y = generate_dataset(output_dim,examples)
#print(x.shape,y.shape)

class Layer(object):
	def __init__(self,input_dim,output_dim,nonlin,nonlin_deriv):
		self.weights = np.random.randn(input_dim,output_dim)*0.2 - 0.1
		self.nonlin = nonlin
		self.nonlin_deriv = nonlin_deriv

	def forward(self,input):
		self.input = input
		self.output = self.nonlin(self.input.dot(self.weights))
		return self.output

	def backward(self,output_delta):
		self.weights_output_delta = output_delta * self.nonlin_deriv(self.output)
		return self.weights_output_delta.dot(self.weights.T)

	def update(self,alpha=0.05):
		self.weights -= self.input.T.dot(self.weights_output_delta)

batch_size = 10
alpha = 0.02

'''input_dim = len(x[0])
layer1_dim = 128
layer2_dim = 64
output_dim = len(y[0])

layer_1 = Layer(input_dim,layer1_dim,sigmoid,sigmoid_derivative)
layer_2 = Layer(layer1_dim,layer2_dim,sigmoid,sigmoid_derivative)
layer_3 = Layer(layer2_dim,output_dim,sigmoid,sigmoid_derivative)

for iter in range(iterations):
	error = 0

	for batch_i in range(int(len(x) / batch_size)):
		batch_x = x[(batch_i*batch_size):(batch_i+1*batch_size)]
		batch_y = y[(batch_i*batch_size):(batch_i+1*batch_size)]

		layer1_out = layer_1.forward(batch_x)
		layer2_out = layer_2.forward(layer1_out)
		layer3_out = layer_3.forward(layer2_out)

		layer3_delta = layer3_out - batch_y
		layer2_delta = layer_3.backward(layer3_delta)
		layer1_delta = layer_2.backward(layer2_delta)
		layer_1.backward(layer1_delta)

		layer_1.update()
		layer_2.update()
		layer_3.update()

		error += np.sum(np.abs(layer3_delta*layer3_out*(1-layer3_out)))

	print(error)
	if(iter%100==99):
		print("")'''

def sigmoid_out2deriv(a):
	return a*(1-a)

# DNI stands for decoupled neural interface
class DNI(object):
	def __init__(self,input_dim,output_dim,nonlin,nonlin_deriv,alpha=0.02):
		self.weights = np.random.randn(input_dim,output_dim)*2 - 0.1
		self.bias = np.random.randn(output_dim)*0.001 

		self.weights_0_1_synthetic_grads = (np.random.randn(output_dim,output_dim)*0.0) - 0.0
		self.bias_0_1_synthetic_grads = np.random.randn(output_dim)*0.0 - 0.0
		self.nonlin_deriv = nonlin_deriv
		self.nonlin = nonlin
		self.alpha = alpha

	def forward_and_synthetic_update(self,inputs,update=True):
		self.inputs = inputs
		self.output = self.nonlin(self.inputs.dot(self.weights)) # + self.bias)

		if not update:
			return self.output		
		else:
			self.synthetic_gradient = (self.output.dot(self.weights_0_1_synthetic_grads)) # + self.bias_0_1_synthetic_grads)
			self.weights_synthetic_gradient = self.synthetic_gradient * self.nonlin_deriv(self.output)

			self.weights -= self.inputs.T.dot(self.weights_synthetic_gradient)*self.alpha
			#self.bias -= np.average(self.weights_synthetic_gradient,axis=0)*self.alpha

		return self.weights_synthetic_gradient.dot(self.weights.T), self.output	    

	def normal_update(self,true_gradient):
		grad = true_gradient * self.nonlin_deriv(self.output)
		self.weights -= self.inputs.T.dot(grad) * self.alpha
		#self.bias -= np.average(grad,axis=0) * self.alpha
		return grad.dot(self.weights.T)

	def update_synthetic_weights(self,true_gradient):
		self.synthetic_gradient_delta = (self.synthetic_gradient - true_gradient)
		self.weights_0_1_synthetic_grads -= self.output.T.dot(self.synthetic_gradient_delta) * self.alpha
		#self.bias_0_1_synthetic_grads -= np.average(self.synthetic_gradient_delta,axis=0) * self.alpha


input_dim = len(x[0])
layer_1_dim = 64
layer_2_dim = 32
output_dim = len(y[0])

layer_1 = DNI(input_dim,layer_1_dim,sigmoid,sigmoid_out2deriv,alpha)
layer_2 = DNI(layer_1_dim,layer_2_dim,sigmoid,sigmoid_out2deriv,alpha)
layer_3 = DNI(layer_2_dim, output_dim,sigmoid, sigmoid_out2deriv,alpha)

for iter in range(iterations):
	error = 0
	synthetic_error = 0

	for batch_i in range(int(len(x) / batch_size)):
		batch_x = x[(batch_i*batch_size):(batch_i+1*batch_size)]
		batch_y = y[(batch_i*batch_size):(batch_i+1*batch_size)] 
		
		_, layer_1_out = layer_1.forward_and_synthetic_update(batch_x)
		layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
		layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out,False)

		layer_3_delta = layer_3_out - batch_y
		layer_2_delta = layer_3.normal_update(layer_3_delta)
		layer_2.update_synthetic_weights(layer_2_delta)
		layer_1.update_synthetic_weights(layer_1_delta)

		error += np.sum(np.abs(layer_3_delta))
		synthetic_error += (np.sum(np.abs(layer_2_delta - layer_2.synthetic_gradient)))

	if iter%100==0:
		print(error,"  ",synthetic_error)
