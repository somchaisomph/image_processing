import numpy as np
from skimage import io,color
import matplotlib.pyplot as plt
	
'''generate a 5x5 kernel'''	
def generating_kernel(a):
	w_1d = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
	return np.outer(w_1d, w_1d)
 
'''reduce image by 1/2'''
def ireduce(image):
	out = None
	image = image.reshape((1,image.shape[0],image.shape[1],1))
	kernel = generating_kernel(0.4)
	kernel = kernel.reshape((kernel.shape[0],kernel.shape[1],1,1))
	b = np.random.randn(1,1,1,1)
	hparameters = {"pad" : 1,"stride": 1}
	Z, cache_conv = conv_forward(image, kernel, b, hparameters)
	out = Z[0,:,:,0][::2,::2]
	return out
 
'''expand image by factor of 2'''
	
def iexpand(image):
	out = None
	kernel = generating_kernel(0.4)	
	kernel = kernel.reshape((kernel.shape[0],kernel.shape[1],1,1))
	
	outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
	outimage[::2,::2]=image[:,:]

	outimage = outimage.reshape((1,outimage.shape[0],outimage.shape[1],1))
	b = np.random.randn(1,1,1,1)
	hparameters = {"pad" : 1,"stride": 1}
	Z, cache_conv = conv_forward(outimage, kernel, b, hparameters)

	out = 4*Z[0,:,:,0] 
	return out
	
'''create a gaussain pyramid of a given image'''
def gauss_pyramid(image, levels):
	output = []
	output.append(image)
	tmp = image
	for i in range(0,levels):
		tmp = ireduce(tmp)
		output.append(tmp)
	return output

'''build a laplacian pyramid'''
def lapl_pyramid(gauss_pyr):
	output = []
	k = len(gauss_pyr)
	for i in range(0,k-1):
		gu = gauss_pyr[i]
		egu = iexpand(gauss_pyr[i+1])
		if egu.shape[0] > gu.shape[0]:
			egu = np.delete(egu,(-1),axis=0)
		if egu.shape[1] > gu.shape[1]:
			egu = np.delete(egu,(-1),axis=1)
		output.append(gu - egu)
	output.append(gauss_pyr.pop())
	return output


'''Blend the two laplacian pyramids by weighting them according to the mask.'''
def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
	blended_pyr = []
	k = len(gauss_pyr_mask)
	for i in range(0,k):
		p1= gauss_pyr_mask[i]*lapl_pyr_white[i]
		p2=(1 - gauss_pyr_mask[i])*lapl_pyr_black[i]
		blended_pyr.append(p1 + p2)
	return blended_pyr

'''Reconstruct the image based on its laplacian pyramid.'''
def collapse(lapl_pyr):
	output = None
	output = np.zeros((lapl_pyr[0].shape[0],lapl_pyr[0].shape[1]), dtype=np.float64)
	for i in range(len(lapl_pyr)-1,0,-1):
		lap = iexpand(lapl_pyr[i])
		lapb = lapl_pyr[i-1]
		if lap.shape[0] > lapb.shape[0]:
			lap = np.delete(lap,(-1),axis=0)
		if lap.shape[1] > lapb.shape[1]:
			lap = np.delete(lap,(-1),axis=1)
		tmp = lap + lapb
		lapl_pyr.pop()
		lapl_pyr.pop()
		lapl_pyr.append(tmp)
		output = tmp
	return output
	
def zero_pad(X, pad):
	"""
	Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
	as illustrated in Figure 1.
	
	Argument:
	X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
	pad -- integer, amount of padding around each image on vertical and horizontal dimensions
	
	Returns:
	X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
	"""
	
	### START CODE HERE ### (≈ 1 line)
	sp = np.shape(X)
	
	X_pad = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values = (0,0))
	### END CODE HERE ###
	
	return X_pad

def conv_single_step(a_slice_prev, W, b):
	"""
	Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
	of the previous layer.
	
	Arguments:
	a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
	W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
	b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
	
	Returns:
	Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
	"""

	### START CODE HERE ### (≈ 2 lines of code)
	# Element-wise product between a_slice and W. Do not add the bias yet.
	s = a_slice_prev * W
	# Sum over all entries of the volume s.
	Z = np.sum(s)
	# Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
	Z = float(Z + b)
	### END CODE HERE ###

	return Z

def conv_forward(A_prev, W, b, hparameters):
	"""
	Implements the forward propagation for a convolution function
	
	Arguments:
	A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
	where
	n_C_prev = number of channel of previous layer
	n_H_prev = number of vertical dimension of previous layer
	n_W_prev = number of horizontal dimension of previous layer
	
	W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
	b -- Biases, numpy array of shape (1, 1, 1, n_C)
	hparameters -- python dictionary containing "stride" and "pad"
		
	Returns:
	Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
	cache -- cache of values needed for the conv_backward() function
	"""
	
	### START CODE HERE ###
	# Retrieve dimensions from A_prev's shape (≈1 line)  
	(m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
	
	# Retrieve dimensions from W's shape (≈1 line)
	(f, f, n_C_prev, n_C) = np.shape(W)
	
	# Retrieve information from "hparameters" (≈2 lines)
	stride = hparameters['stride']
	pad = hparameters['pad']
	
	# Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
	n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
	n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
	
	# Initialize the output volume Z with zeros. (≈1 line)

	Z = np.zeros((m, n_H, n_W, n_C))
	
	# Create A_prev_pad by padding A_prev
	A_prev_pad = zero_pad(A_prev, pad)

	for i in range(m):			                      # loop over the batch of training examples
		a_prev_pad = A_prev_pad[i,:,:,:]                # Select ith training example's padded activation
		for h in range(n_H):						   # loop over vertical axis of the output volume
			for w in range(n_W):					   # loop over horizontal axis of the output volume
				for c in range(n_C):				   # loop over channels (= #filters) of the output volume
					
					# Find the corners of the current "slice" (≈4 lines)
					vert_start = h * stride
					vert_end = h * stride+ f
					horiz_start = w * stride
					horiz_end = w * stride + f
					
					# Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
					a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
					
					# Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
					Z[i, h, w, c] = conv_single_step(a_slice_prev,W[:,:,:,c], b[:,:,:,c])
			 
	### END CODE HERE ###
	
	# Making sure your output shape is correct
	assert(Z.shape == (m, n_H, n_W, n_C))
	
	return Z

def test_pyramid_down(image):
	gp_img = gauss_pyramid(image,levels=2)
	fig=plt.figure(figsize=(8, 8))
	rows = 3
	columns = 1
	for i in range(len(gp_img)):
		fig.add_subplot(rows, columns, i)
    		plt.imshow(gp_img[i])
	plt.show()
	
if __name__ == "__main__":	
	
	img = io.imread('somchai.png')    # Load the image
	img = color.rgb2gray(img) 	
	
	
	#lapl_pyr_image1r  = lapl_pyramid(_img[0])
	
	#for m in _img :
	#	print(m.shape)
	ex_img = iexpand(img)
	
	plt.imshow(ex_img, cmap=plt.cm.gray)
	#plt.axis('off')
	plt.show()
