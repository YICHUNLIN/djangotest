from django.shortcuts import render
from django.shortcuts import render_to_response
import datetime


def RKindex(request):
	return render(request, 'RK/index.html', {'currentTime':"2017 06 07", 'dept':'土木碩一', 'title':'Vic Lin 2017 06 07'})

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
