from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy

from .forms import BookForm
from .models import Book

import subprocess
from subprocess import Popen, PIPE

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torchvision import transforms


from .model import EncoderCNN, DecoderRNN
from PIL import Image
import easydict
import sys

import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
import easydict
import time
import json
from . import voc
#from .build_vocab import Vocabulary
sys.modules["voc"] = voc
#sys.modules['Vocabulary'] = Vocabulary
#import pickleversion
#from django.views.decorators.csrf import csrf_exempt

class Home(TemplateView):
    template_name = 'home.html'






# @csrf_exempt
# def register_user(request):
#     context = {}
#     if request.method == 'POST':
#         form = RegisterForm(request.POST)
#         if form.is_valid():
#             pass
#     else:
#         form = RegisterForm(auto_id=False)
#     context['form'] = form
#     return render_to_response('templates/register.htm',context, context_instance=RequestContext(request))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    print('main 호출')
    # Load vocabulary wrapper
    print("파일정보:", args["vocab_path"])

    # aa = Vocabulary()
    # print("Vocabulary:", aa)



    with open(args.vocab_path, "rb") as f:
    #vocab = pickle.load(open('D:/dev/django-upload-example/mysite/core/data/vocab.pkl','r', encoding="latin1"))
        #print("파일열기:", f)
        # import pickletools
        # pickletools.dis(f)

        vocab=pickle.load(f)
        #vocab = pickle.load(f)
            #vocab = f.readlines()
            #print(vocab)

    print('pickle 열기')
    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    print('load_image_image/////////////////////////////////////////////////////////////////')
    image_tensor = image.to(device)
    print('loadimage image tensor',image_tensor)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    print('image_tensor encoder feartures',feature)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    print(sampled_caption)
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        print(word)
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    #Print out the image and the generated caption
    return sentence




def upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        print("uploaded_file:", uploaded_file)
        fs = FileSystemStorage()
        #name = fs.save(uploaded_file.name, uploaded_file)
        #name = uploaded_file.name
        #context['url'] = fs.url(name)
        context['uploaded_file'] = uploaded_file


        # Device configuration

        #MYTEST = "/home/encore/notebook/pytorch-tutorial-master/tutorials/03-advanced/image_captioning"

        #    image = Image.open(args.image)
        #    plt.imshow(np.asarray(image))
        #if __name__ == '__main__':
        args = easydict.EasyDict({
                            #'image': sys.argv[1],
            'image': uploaded_file,
            'encoder_path': 'D:/ml/test/django-upload-example/mysite/core/models/encoder-5-2000.ckpt', # MYTEST + '/models/encoder-5-1000.ckpt'
            'decoder_path': 'D:/ml/test/django-upload-example/mysite/core/models/decoder-5-2000.ckpt', # MYTEST + '/models/decoder-5-1000.ckpt
            'vocab_path' : 'D:/ml/test/django-upload-example/mysite/core/data/vocab.pkl', #MYTEST + '/data/vocab.pkl'
            'embed_size' : 256,
            'hidden_size' : 512,
            'num_layers' : 1
            })

        sentence = main(args) #문장출력
        context['sentence'] = sentence
        return render(request, 'upload_result.html', context)
    return render(request, 'upload.html', context)


def book_list(request):
    books = Book.objects.all()
    return render(request, 'book_list.html', {
        'books': books
    })


def upload_book(request):
    if request.method == 'POST':
        form = BookForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('book_list')
    else:
        form = BookForm()
    return render(request, 'upload_book.html', {
        'form': form
    })


def delete_book(request, pk):
    if request.method == 'POST':
        book = Book.objects.get(pk=pk)
        book.delete()
    return redirect('book_list')


class BookListView(ListView):
    model = Book
    template_name = 'class_book_list.html'
    context_object_name = 'books'


class UploadBookView(CreateView):
    model = Book
    form_class = BookForm
    success_url = reverse_lazy('class_book_list')
    template_name = 'upload_book.html'
