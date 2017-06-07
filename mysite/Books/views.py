from django.shortcuts import render
from django.shortcuts import render_to_response
from .models import Boook
# Create your views here.

def bookindex(request):
	return render(request, 'books/index.html')

def search(request):
	r = None
	if 'q' in request.GET:
		message = "You are search %r" %(request.GET['q'])
		try:
			r = Boook.objects.get(title=str(request.GET['q']))
		except Exception as e:
			print('not found' + request.GET['q'])
	else:
		message = "You are summit empty form."
	return render(request, 'books/index.html', {'message':message, 'obj':''})