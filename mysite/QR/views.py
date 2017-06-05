from django.shortcuts import render

# Create your views here.

def qrindex(request):
	return render(request, 'QR/index.html')

def doQR(request):
	
