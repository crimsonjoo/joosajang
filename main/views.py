from django.shortcuts import render,redirect




def index(request):
    return render(request,'main/index.html')



def error400(request,exception):
    return render(request,'main/400.html',status=400)

def error404(request,exception):
    return render(request,'main/404.html',status=404)

def error500(request):
    return render(request,'main/500.html',status=500)