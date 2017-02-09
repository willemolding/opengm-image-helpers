"""
functions to create instances of opengm graphical models for various useful layer graphs
"""
import numpy as np
import opengm
from skimage.future import graph

import networkx 
from networkx.drawing.nx_agraph import graphviz_layout
networkx.graphviz_layout = graphviz_layout

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
	n_variables = n_pixels + n_segments

	n_labels_pixels = pixel_unaries.shape[-1]
	n_labels_segments = segment_unaries.shape[-1]

	# calculate the region adjacency graph for the segments
	rag = graph.rag_mean_color(np.zeros_like(segment_map), segment_map)
	rag_edges = np.array(rag.edges()) + n_pixels #segment indices start at n_pixels remember!

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
	fid = gm.addFunction(pixel_regularizer)
	vis = opengm.secondOrderGridVis(pixel_unaries.shape[0],pixel_unaries.shape[1])
	gm.addFactors(fid,vis, finalize=False)

	# segment rag
	fid = gm.addFunction(segment_regularizer)
	gm.addFactors(fid, rag_edges, finalize=False)

	# inter-layer
	fid = gm.addFunction(inter_layer_regularizer)
	vis = np.dstack([np.arange(n_pixels).reshape(pixel_unaries.shape[:2]), segment_map+n_pixels]).reshape((-1,2))
	gm.addFactors(fid, vis, finalize=False)

	gm.finalize()

	return gm


import time

if __name__ == '__main__':
	# run some tests on the package
	shape = (100,100)
	n_pixels = shape[0]*shape[1]

	n_pixel_labels = 3
	n_segment_labels = 4
	pixels = np.random.random((shape+(n_pixel_labels,)))
	segment_map = np.arange(n_pixels).reshape(shape)
	segment_values = np.random.random((np.unique(segment_map).size,n_segment_labels))

	t0 = time.time()
	gm = segment_overlap_graph(
		pixels, 
		opengm.pottsFunction([n_pixel_labels,n_pixel_labels], 0.0, 0.5),
		segment_map,
		segment_values,
		opengm.pottsFunction([n_segment_labels,n_segment_labels], 0.0, 0.5),
		opengm.pottsFunction([n_pixel_labels, n_segment_labels], 0.0, 0.5),
		)
	t1 = time.time()

	print "graph build in", t1-t0, "seconds"

	assert gm.numberOfVariables == n_pixels + np.unique(segment_map).size
	assert gm.numberOfLabels(0) == n_pixel_labels
	assert gm.numberOfLabels(n_pixels) == n_segment_labels

	# test inference is possible 
	inference = opengm.inference.BeliefPropagation(gm=gm)
	t0 = time.time()
	inference.infer()
	t1 = time.time()

	print "belief propagation completed in", t1-t0, "seconds"

