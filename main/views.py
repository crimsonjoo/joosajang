from django.shortcuts import render

def index(request):
    return render(request,'main/index.html')


def bad_request(request,exception):  #400 page not found
    return render(request,'main/400.html')

def page_not_found(request,exception):  #404 page not found
    return render(request,'main/404.html')

def server_error(request):  #500 page not found
    return render(request,'main/500.html')