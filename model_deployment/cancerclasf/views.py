from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import *


# def home(request):
#     return HttpResponse("<h1>Home Page<h1>")


def cancer_image_view(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = ImageForm()
    return render(request, 'image_upload.html', {'form': form})


def success(request):
    return HttpResponse('successfully uploaded')



