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
		self.histogram_node = None
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

	def train_node(self, X, Y, Signature, maximum_depth=10, min_samples_split=5):

		# Seleziona i campioni associati al nodo
		bitmask = np.where(Signature == self.ID)[0]
		X_node = X[bitmask]
		Y_node = Y[bitmask]

		# Se il nodo soddisfa un criterio di stop, diventa una foglia
		if self.ID >= maximum_depth or len(Y_node) < min_samples_split or np.unique(Y_node).size == 1:
			self.best_hyperplane = None  # Indica che il nodo è una foglia
			self.histogram_node = np.bincount(Y_node, minlength=self.histograms_no.shape[1])
			return  # Stoppa il training per questo nodo

		# Calcola l'istogramma del nodo
		self.histogram_node = np.bincount(Y_node, minlength=self.histograms_no.shape[1])
		score_hyperplane = np.zeros(self.separating_hyperplanes.shape[0])

		for idx_hyperplane, hyperplane in enumerate(self.separating_hyperplanes):
			for i, x in enumerate(X_node):
				if np.dot(hyperplane, x) > 0:  # Yes
					self.histograms_yes[idx_hyperplane, Y_node[i]] += 1
					Signature[bitmask[i]] = 2 * (self.ID + 1)
				else:  # No
					self.histograms_no[idx_hyperplane, Y_node[i]] += 1
					Signature[bitmask[i]] = 2 * (self.ID + 1) - 1

			# Penalizza gli iperpiani che non separano i dati
			if np.sum(self.histograms_yes[idx_hyperplane]) == 0 or np.sum(self.histograms_no[idx_hyperplane]) == 0:
				score_hyperplane[idx_hyperplane] = -np.inf
			else:
				score_hyperplane[idx_hyperplane] = Jensen_Shannon_divergence(self.histograms_yes[idx_hyperplane], self.histograms_no[idx_hyperplane])

		# Se nessun iperpiano è valido, il nodo diventa una foglia
		if np.all(score_hyperplane == -np.inf):
			self.best_hyperplane = None
		else:	
			idx_best_hyperplane = np.argmax(score_hyperplane)
			self.best_hyperplane = self.separating_hyperplanes[idx_best_hyperplane]





	def inference(self, X, Signature, maximum_depth=10):

		# Seleziona i campioni che questo nodo deve processare
		bitmask = np.where(Signature == self.ID)[0]
		X_node = X[bitmask]
		Y_pred = np.zeros(X_node.shape[0], dtype=int)

		# Se il nodo è una foglia, assegna la classe più frequente
		if self.best_hyperplane is None or self.ID >= maximum_depth:
			most_frequent_class = np.argmax(self.histogram_node)
			Y_pred[:] = most_frequent_class  # Tutti gli elementi prendono la stessa classe
			return Y_pred, Signature

	# Se il nodo è interno, inoltra i campioni in base all'iperpiano
		for i, x in enumerate(X_node):
			if np.dot(self.best_hyperplane, x) > 0:
				Signature[bitmask[i]] = 2 * (self.ID + 1)  # Va a destra
			else:
				Signature[bitmask[i]] = 2 * (self.ID + 1) - 1  # Va a sinistra

		return Y_pred, Signature


# Signature deve essere inizializzato a 0 inizialmente per il nodo principale




