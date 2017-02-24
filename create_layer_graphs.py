"""
functions to create instances of opengm graphical models for various useful layer graphs


"""
import numpy as np
import opengm
from skimage.future import graph

def _remove_rows_with_negative(arr):
	return arr[np.greater_equal(arr, 0).all(axis=1)]

def _check_valid_segment_map(segment_map):
	if not np.array_equal( np.unique(segment_map[segment_map >= 0]) , np.arange(np.max(segment_map+1)) ):
		raise ValueError('Segment map is not valid. Segment identifiers must have all values from 0 to max(segment_map)')

def _check_compatible_segment_map_unaries(segment_map, unaries):
	if not unaries.shape[0] == np.max(segment_map)+1:
		raise ValueError('segment map and segment unaries are not compatible')

def calc_n_pixel_edges(image_shape):
	return (image_shape[0]-1)*image_shape[1] + (image_shape[1]-1)*image_shape[0]

def segment_map_to_rag_edges(segment_map):
	rag = graph.rag_mean_color(np.zeros_like(segment_map), segment_map)
	rag_edges = np.array(rag.edges()) 
	rag_edges = _remove_rows_with_negative(rag_edges) #remove all edges connection to segment labels < 0 as they represent no segment
	return rag_edges


##########################################################################################
## Functions for adding layers to existing graphs
##########################################################################################


def add_layer(gm, unaries=None, edges=None, pairwise=None, offset=0):
	"""
	adds a layer to a GM. All variables in a layer must share the same label space

	The layer may have unary and/or pairwise functions

	unaries are given as shape (n_vars, n_labels)
	edges are given as shape (n_edges, 2)
	pairwise are given as either: 
		pairwise opengm function e.g. opengm.PottsFunction([n_labels,n_labels],0.0,beta)
		nd.array of shape (n_labels, n_labels)
		list of pairwise opengm functions of length n_edges
		list of nd.array of length n_edges where each array has shape (n_labels, n_labels)

	"""

	n_vars = unaries.shape[0]

	if unaries is not None:
		# add unary functions and factors
		fids = gm.addFunctions(unaries)
		vids = np.arange(n_vars) + offset
		gm.addFactors(fids, vids, finalize=False)


	if pairwise is not None and edges is not None:
		if isinstance(pairwise, list):
			fids = gm.addFunctions(pairwise)
		else:
			# add pairwise functions and factors
			fids = gm.addFunction(pairwise)
		vids = edges + offset
		gm.addFactors(fids, vids, finalize=False)

	return gm

def add_lattice_layer(gm, pixel_unaries, pixel_regularizer=None, offset=0):
	"""
	Adds a lattice (pixel grid) layer to a GM where all edges share the same regularizer

	offset is the variable index of the first new variable. Make this the count of all variables already added

	Parameters:
		- pixel_unaries - a 3D array of shape (width, height, n_labels). 
		- pixel_regularizer (optional) - a pairwise opengm function e.g. opengm.PottsFunction([2,2],0.0,beta)
	"""
	n_labels = pixel_unaries.shape[-1]
	n_pixels = pixel_unaries.shape[0]*pixel_unaries.shape[1]

	edges = np.array(opengm.secondOrderGridVis(pixel_unaries.shape[0],pixel_unaries.shape[1]))

	unaries = pixel_unaries.reshape([n_pixels,n_labels])

	gm = add_layer(gm, unaries, edges, pairwise=pixel_regularizer, offset=offset)

	return gm

def add_potts_lattice_layer(gm, pixel_unaries, beta, offset=0):
	"""
	Adds a lattice layer to a gm where all pairwise functions are Potts model

	Parameters:
		- pixel_unaries - a 3D array of shape (width, height, n_labels). 
		- beta - the smoothing parameter for the Potts model. Penalty for label dissimilarity
	"""
	n_labels = pixel_unaries.shape[-1]
	add_lattice_layer(gm, pixel_unaries, 
		pixel_regularizer=opengm.PottsFunction([n_labels,n_labels],0.0,beta), offset=offset)



##########################################################################################
## Functions for constructing graphs
##########################################################################################

def pixel_lattice_graph(pixel_unaries, pixel_regularizer):
	n_vars = pixel_unaries.shape[0]*pixel_unaries.shape[1]
	n_labels = pixel_unaries.shape[-1]
	n_edges = calc_n_pixel_edges(pixel_unaries.shape)

	gm = opengm.graphicalModel([n_labels]*n_vars)
	gm.reserveFunctions(n_vars + 1,'explicit') # the unary functions plus the 1 type of regularizer
	gm.reserveFactors(n_vars + n_edges)

	gm = add_lattice_layer(gm, pixel_unaries, pixel_regularizer)

	gm.finalize()
	return gm

def segment_adjacency_graph(segment_unaries, segment_map, segment_regularizer=None):
	"""
	Creates a region adjacency graph. Each segment has a variable and adjacent segments are linked by edges
	"""
	_check_valid_segment_map(segment_map)
	_check_compatible_segment_map_unaries(segment_map, segment_unaries)

	n_vars = segment_unaries.shape[0]
	n_labels = segment_unaries.shape[-1]
	edges = segment_map_to_rag_edges(segment_map)
	n_edges = edges.shape[0]

	# allocate space for the model and all its variables
	gm = opengm.graphicalModel([n_labels]*n_vars)

	gm.reserveFunctions(n_vars + n_edges,'explicit') # the unary functions plus the 3 types of regularizer
	gm.reserveFactors(n_vars + n_edges)

	gm = add_layer(gm, segment_unaries, edges, segment_regularizer)
	gm.finalize()
	return gm


def segment_overlap_graph(pixel_unaries, segment_map, segment_unaries, pixel_regularizer=None, segment_regularizer=None, inter_layer_regularizer=None):
	"""
	greates a graphical model comprised of two layers. 
		- The first layer is a pixel lattice
		- The second layer is a region adjacency graph over segments
		- Connections exist between pixels where they are overlapped by a segment

	Parameters:
		- pixel_unaries - a 3D array of shape (width, height, n_labels). 
		- segment_map - a 2d array of shape (width, height). Each element >= 0 is a segment id that maps the corresponding pixel to that id. -1 represents no segment and no corresponding node will be added
		- segment_unaries - a 2d arry of shape (n_segments, n_labels)
		- pixel_regularizer (optional) - a pairwise opengm function e.g. opengm.PottsFunction([2,2],0.0,beta) or list of opengm functions of length n_pixels
		- segment_regularizer (optional) - a pairwise opengm function, same requirements as pixel_regularizer
		- inter_layer_regularizer (optional) - a pairwise opengm function, same requirements as pixel_regularizer
	"""
	_check_valid_segment_map(segment_map)
	_check_compatible_segment_map_unaries(segment_map, unaries)

	# calculate how many variables and factors will be required
	n_pixels = pixel_unaries.shape[0]*pixel_unaries.shape[1]
	n_segments = segment_unaries.shape[0]
	n_variables = n_pixels + n_segments

	n_labels_pixels = pixel_unaries.shape[-1]
	n_labels_segments = segment_unaries.shape[-1]

	# calculate the region adjacency graph for the segments
	rag_edges = segment_map_to_rag_edges(segment_map)
	rag_edges += n_pixels #segment indices start at n_pixels remember!

	n_pixel_edges = (pixel_unaries.shape[0]-1)*pixel_unaries.shape[1] + (pixel_unaries.shape[1]-1)*pixel_unaries.shape[0]
	n_segment_edges = rag_edges.shape[0] #check this is right
	n_inter_edges = n_pixels
	n_edges = n_pixel_edges + n_segment_edges + n_inter_edges


	# allocate space for the model and all its variables
	gm = opengm.graphicalModel([n_labels_pixels]*n_pixels + [n_labels_segments]*n_segments)

	gm.reserveFunctions(n_variables + 3,'explicit') # the unary functions plus the 3 types of regularizer
	gm.reserveFactors(n_variables + n_edges)

	# add unary functions and factors
	fids = gm.addFunctions(pixel_unaries.reshape([n_pixels,n_labels_pixels]))
	gm.addFactors(fids, np.arange(n_pixels), finalize=False)

	fids = gm.addFunctions(segment_unaries)
	gm.addFactors(fids, n_pixels + np.arange(n_segments), finalize=False)

	## add pairwise functions
	# pixel lattice
	if pixel_regularizer is not None:
		fid = gm.addFunction(pixel_regularizer)
		vis = opengm.secondOrderGridVis(pixel_unaries.shape[0],pixel_unaries.shape[1])
		gm.addFactors(fid,vis, finalize=False)

	# segment rag
	if segment_regularizer is not None:
		fid = gm.addFunction(segment_regularizer)
		gm.addFactors(fid, np.sort(rag_edges, axis=1), finalize=False)

	# inter-layer
	if inter_layer_regularizer is not None:
		fid = gm.addFunction(inter_layer_regularizer)
		vis = np.dstack([np.arange(n_pixels).reshape(pixel_unaries.shape[:2]), segment_map]).reshape((-1,2))
		vis = _remove_rows_with_negative(vis)
		vis[:,1] += n_pixels
		gm.addFactors(fid, vis, finalize=False)

	gm.finalize()

	return gm



##########################################################################################
## Tests
##########################################################################################


if __name__ == '__main__':
	import time

	import networkx 
	from networkx.drawing.nx_agraph import graphviz_layout
	networkx.graphviz_layout = graphviz_layout

	# run some tests on the package
	shape = (4,4)
	n_pixels = shape[0]*shape[1]

	n_pixel_labels = 2
	n_segment_labels = 2
	pixels = np.random.random((shape+(n_pixel_labels,)))

	segment_map = np.arange(n_pixels).reshape(shape)
	segment_values = np.random.random((np.max(segment_map)+1,n_segment_labels))


	t0 = time.time()
	gm = pixel_lattice_graph(pixels, opengm.pottsFunction([n_pixel_labels,n_pixel_labels], 0.0, 0.5))
	t1 = time.time()

	opengm.visualizeGm(gm)
	print "graph build in", t1-t0, "seconds"


	t0 = time.time()
	gm  = segment_adjacency_graph(segment_values, segment_map, segment_regularizer=opengm.pottsFunction([n_segment_labels,n_segment_labels], 0.0, 0.5))	
	t1 = time.time()
	opengm.visualizeGm(gm)
	print "graph build in", t1-t0, "seconds"


	t0 = time.time()
	gm = segment_overlap_graph(
		pixels, 
		segment_map,
		segment_values,
		opengm.pottsFunction([n_pixel_labels,n_pixel_labels], 0.0, 0.5),
		opengm.pottsFunction([n_segment_labels,n_segment_labels], 0.0, 0.5),
		opengm.pottsFunction([n_pixel_labels, n_segment_labels], 0.0, 0.5),
		)
	t1 = time.time()

	print "graph build in", t1-t0, "seconds"


	# assert gm.numberOfVariables == n_pixels + np.max(segment_map)+1
	# assert gm.numberOfLabels(0) == n_pixel_labels
	# assert gm.numberOfLabels(n_pixels) == n_segment_labels

	opengm.visualizeGm(gm)

	# test inference is possible 
	inference = opengm.inference.GraphCut(gm=gm)
	t0 = time.time()
	inference.infer()
	t1 = time.time()

	print "inference completed in", t1-t0, "seconds"

# For a binary model of size (1000,1000) with only single pixel segments (e.g. 2 million nodes)
# graph build in 66.5958359241 seconds
# inference completed in 131.123669863 seconds

