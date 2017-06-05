# encoding = utf-8
import copy 
import numpy as np
import configparser
import math
import matplotlib.pyplot as plt

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

	def EquParser(self, inifile):
		file=configparser.ConfigParser()
		file.read(inifile)
		return (file['data']['m'], file['data']['c'], file['data']['k'], file['data']['p'], file['data']['inity'], file['data']['initv'])

	def writeQR(self, outfile, lds, egvs):
		f = open(outfile, "a")
		cp = configparser.ConfigParser()

		cp.add_section("solution")

		for d in lds:
			cp.set("solution",d[0],d[1])


		for d in egvs:
			cp.set("solution",d[0],d[1])

		cp.write(f)
		f.close()

	def writeEQU(self, outfile, lds):
		f = open(outfile, "a")
		cp = configparser.ConfigParser()

		cp.add_section("solution")

		for d in lds:
			cp.set("solution",d[0],d[1])

		cp.write(f)
		f.close()



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

class SolveEqu:
	def RungeKutta(self, f, P, fn, C, u, dt, sec, M, K):
		plt.title("%.2lfy`` + %.2lfy` + %.2lfy=0, y(0)=%.2lf, y(1)=%2.lf" %(M,C,K,u[0],u[1]))
		X = []
		Y = []
		for i in range(int(sec/dt)):
			t = i * dt
			X.append(t)
			Y.append(u[0])
			tM = t+dt/2
			k1 = f(t, u, P(fn,t), K, C, M)
			uM = [u[0]+dt/2*k1[0],u[1]+dt/2*k1[1]]
			k2 = f(tM, uM, P(fn,t), K, C, M)
			uM = [u[0]+dt/2*k2[0],u[1]+dt/2*k2[1]]
			k3 = f(tM, uM, P(fn,t), K, C, M)
			ul = [u[0]+dt*k3[0],u[1]+dt*k3[1]]
			k4 = f(t+dt, ul, P(fn,t), K, C, M)
			u = [u[0]+dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]), u[1]+dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])]
		plt.xlabel('t')
		plt.ylabel('y')
		plt.plot(X, Y)
		plt.show()
		return Y

def findEigen():
	print("Ori matrix")
	A = DataPaser().matrixParser("0551283_1.ini")
	print(A[1])

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
	DataPaser().writeQR("0551283_1.ini", Lds, Eigs)

def drawEqu():
	config = DataPaser().EquParser("0551283_2.ini")
	print(config)

	def f2(t, u, P, K, C, M):
		return [u[1], (P-K*u[0]-C*u[1])/M]

	def Pf(fn,t):
		return int(config[3])


	def fn(wn):
		return wn / (2*math.pi)

	def C(xi, wn, M):
		return 2 * xi * wn* M

	def wn(K,M):
		return math.sqrt(K/M)

	K = int(config[2])
	M = int(config[0])
	u = [int(config[4]), int(config[5])]
	print(u)
	dt = 0.01
	sec = 100
	Y = SolveEqu().RungeKutta(f2, Pf,fn(wn(K,M)), int(config[1]), u, dt, sec, M, K)

	ys = []
	for i in range(len(Y)):
		v = ("y%d"%i, str(Y[i]))
		ys.append(v)

	DataPaser().writeEQU("0551283_2.ini", ys)






def main():
	findEigen()
	drawEqu()

	





if __name__ == "__main__":
	main()



