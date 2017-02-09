"""
functions to create instances of opengm graphical models for various useful layer graphs
"""
import opengm


def segment_overlap_graph(pixel_unaries, pixel_regularizer, segment_map, segment_unaries, segment_regularizer, inter_layer_regularizer):
	"""
	greates a graphical model comprised of two layers. 
		- The first layer is a pixel lattice
		- The second layer is a region adjacency graph over segments
		- Connections exist between pixels where they are overlapped by a segment

	Parameters:
		- pixel_unaries - a 3D array of shape (width, height, n_labels). 
		- pixel_regularizer - a pairwise opengm function e.g. opengm.PottsFunction([2,2],0.0,beta)
		- segment_map - a 2d array of shape (width, height). Each element >= 0 is a segment id that maps the corresponding pixel to that id. -1 represents no segment.
		- segment_unaries - a 2d arry of shape (n_segments, n_labels)
		- segment_regularizer - a pairwise opengm function, same properties as pixel_regularizer
		- inter_layer_regularizer - a pairwise opengm function, same properties as pixel_regularizer
	"""

	# calculate how many variables and factors will be required
	n_pixels = pixel_unaries.shape[0]*pixel_unaries.shape[1]
	n_segments = segment_unaries.shape[0]

	n_labels_pixels = pixel_unaries.shape[2]
	n_labels_segments = segment_unaries.shape[1]

	# allocate space for the model and all its variables
	gm = opengm.graphicalModel([n_labels_pixels]*n_pixels + [n_labels_segments]*n_segments)

	gm.reserveFunctions(,'explicit')
	gm.reserveFactors()
