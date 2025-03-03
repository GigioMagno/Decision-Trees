import numpy as np
import random

# Entropy measurement:

def Kullback_Leibler(hist_1, hist_2, epsilon=1e-6):
	
	if np.sum(hist_1) == 0 or np.sum(hist_2) == 0:
		return np.inf

	p = hist_1 / (np.sum(hist_1) + epsilon)
	q = hist_2 / (np.sum(hist_2) + epsilon)
	
	return np.sum(p * np.log((p + epsilon) / (q + epsilon)))

def Jensen_Shannon_divergence(hist_1, hist_2, epsilon=1e-6):
	
	if np.sum(hist_1) == 0 or np.sum(hist_2) == 0:
		return np.inf
	
	p = hist_1 / (np.sum(hist_1) + epsilon)
	q = hist_2 / (np.sum(hist_2) + epsilon)
	avg = 0.5 * (p + q)
	
	return 0.5 * Kullback_Leibler(p, avg, epsilon) + 0.5 * Kullback_Leibler(q, avg, epsilon)

# Class Node

class Node:
	def __init__(self, features, hyperplanes, classes, ID):

		#Binary mask selection
		self.feature_subset_mask = np.random.randint(0, 2, features)

		#Data structures for left/right histograms and separating hyperplanes 
		self.histograms_yes = np.zeros((hyperplanes, classes))
		self.histograms_no = np.zeros((hyperplanes, classes))
		self.separating_hyperplanes = np.zeros((hyperplanes, features))

		#Best hyperplane
		self.best_hyperplane = None

		#Node ID
		self.ID = ID

		#Check to select at least one feature
		while np.sum(self.feature_subset_mask) == 0:
			self.feature_subset_mask = np.random.randint(0, 2, features)

		# Random initialization for hyperplanes
		for hyperplane in range(hyperplanes):
			for feature in range(features):
				if self.feature_subset_mask[feature] == 1:
					self.separating_hyperplanes[hyperplane, feature] = random.uniform(-1, 1)

	def train_node(self, X, Y, Signature):

		#Select the samples that this node has to process
		bitmask = np.where(Signature == self.ID)[0]
		X_node = X[bitmask]
		Y_node = Y[bitmask]
		score_hyperplane = np.zeros(self.separating_hyperplanes.shape[0])

		for idx_hyperplane, hyperplane in enumerate(self.separating_hyperplanes):
			for i, x in enumerate(X_node):
				if np.dot(hyperplane, x) > 0:  # Yes
					self.histograms_yes[idx_hyperplane, Y_node[i]] += 1
					Signature[bitmask[i]] = 2 * (self.ID + 1)
				else:  # No
					self.histograms_no[idx_hyperplane, Y_node[i]] += 1
					Signature[bitmask[i]] = 2 * (self.ID + 1) - 1

			# If one of the two distribution is empty then penalize the hyperplane, otherwise compute the divergence
			if np.sum(self.histograms_yes[idx_hyperplane]) == 0 or np.sum(self.histograms_no[idx_hyperplane]) == 0:
				score_hyperplane[idx_hyperplane] = -np.inf
			else:
				score_hyperplane[idx_hyperplane] = Jensen_Shannon_divergence(self.histograms_yes[idx_hyperplane], self.histograms_no[idx_hyperplane])

		# Select the best hyperplane
		if np.all(score_hyperplane == -np.inf):
			self.best_hyperplane = None
		else:	
			idx_best_hyperplane = np.argmax(score_hyperplane)
			self.best_hyperplane = self.separating_hyperplanes[idx_best_hyperplane]


	def inference():
		pass


# Signature deve essere inizializzato a 0 inizialmente per il nodo principale




