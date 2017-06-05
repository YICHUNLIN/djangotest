from django.shortcuts import render
import numpy as np
import configparser
import math
# Create your views here.

def qrindex(request):
	return render(request, 'QR/index.html')

def doQR(request):
	#print(request.GET)
	#print("Ori matrix")
	A = DataPaser().matrixFromRequest(request.GET['matrix'], request.GET['dim'])
	#print(test[1])
	#A = DataPaser().matrixParser("0551283_1.ini")
	#A = DataPaser().makeadata()
	#print(A[1])
	decomper = Decomposition()
	U = np.eye(A[2])
	for i in range(100):
		(Q,R) = decomper.QR(A)
		ra = np.dot(R,Q)
		A = ([],ra,A[2])
		U = np.dot(U,Q)
	print("Eigen vector:")
	print(U)
	print("Lambdas:")
	Lds = []
	Eigs = []
	for i in range(A[2]):
		v = ("EigV%d" % (i+1), str(U[i,:]))
		Eigs.append(v)
		d = ("lambda%d" % (i+1), str(A[1][i][i]))
		Lds.append(d)
		print("lambda%d = %lf " %(i+1, A[1][i][i]))
	return render(request, 'QR/index.html')

class DataPaser:
	def matrixParser(self, inifile):
		file=configparser.ConfigParser()
		file.read(inifile)
		matrixdatatitle = file.options('data')
		dim = int(file['data']['n'])
		matrix = []
		for title in matrixdatatitle:
			dataraw = file['data'][title]
			if title[0] == 'r':
				raw = dataraw.split(',')
				row = []
				for r in raw:
					row.append(int(r))
				matrix.append(row)
		orimatrix = matrix

		npmatrix = np.array(matrix)
		dim = dim
		return (matrix,npmatrix,dim)

	def makeadata(self):
		matrix = [[ 1,-2,4,7],[-2,3,5,8],[4,5,6,7],[7,8,7,4]]
		dim = 4
		npmatrix = np.array(matrix)

		return (matrix, npmatrix, dim)

	def matrixFromRequest(self, raw, dim):
		rawrows = raw.split("\r\n")
		matrix = []
		for row in rawrows:
			stringrow = row.split(',')
			newrow = []
			for r in stringrow:
				newrow.append(int(r))
			matrix.append(newrow)
		return (matrix, np.array(matrix), int(dim))

class Decomposition:
	def __init__(self):
		pass
	def QR(self, data):
		A = data[1].T
		dim = data[2]
		R = np.zeros((dim, dim))
		Q = np.zeros((dim, dim))
		Y = np.zeros((dim, dim))
		for j in range(0, dim):
			Y[:,j] = A[:, j]
			for k in range(0, j):
				Y[:,j] = Y[:,j] - np.dot(np.dot(Y[:, k], A[:,j]), Y[:, k]) / (np.linalg.norm(Y[:,k])*np.linalg.norm(Y[:,k]))
			R[j][j] = np.linalg.norm(Y[:, j])
			Q[:,j] = Y[:,j]/R[j][j]
		for j in range(0, dim):
			for k in range(j, dim):
				if j != k:
					R[j][k] = np.dot(Q[:,j], A[:, k]) 
		return (Q,R)
