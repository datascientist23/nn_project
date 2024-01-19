import streamlit as st
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
from torchvision import transforms as T
from torchvision import io as io_t
import torch
import torchvision
import torch.nn as nn
from PIL import Image

device = 'cpu'
model = models.resnet50(pretrained=True)
model_2 = models.resnext50_32x4d(pretrained=True)

def predict(model, img, labels):
    resize = T.Resize((224, 224))
    img_pil = Image.fromarray(img)
    img = resize(T.ToTensor()(img_pil)/255)
    decode = lambda x: labels[x]
    return decode(model(img.to(device).unsqueeze(0)).argmax().item())

selected_model = st.sidebar.radio("Выбери предсказание", ("предсказание 1", "предсказание 2"))

# Модель 1
if selected_model == "предсказание 1":
    st.write("""
    # Приложение предсказывает что изображено на картинке:
    * Строения, 
    * Лес, 
    * Ледник,
    * Горы,
    * Море,
    * Улица.
    """)
    uploaded_img = st.sidebar.file_uploader('Добавь свое изображение', type=['jpeg'])
    if uploaded_img is not None:
        input_img = io.imread(uploaded_img)
    else:
        input_img = io.imread('https://big-altay.ru/wp-content/uploads/2022/07/81e19fb4b2d131280729655756878285.jpg')

    labels1 = {0: 'кажется здесь строения', 1: 'кажется это лес', 2: 'похоже на ледник', 
               3: 'якобы горы', 4: 'кажется это море', 5: 'вроде улица'}
    model_1 = model
    model_1.fc = nn.Linear(2048, 6, bias=True)
    model_1.load_state_dict(torch.load('/Users/id/Documents/strlit/08_05_strlit/resnet50_model.pt', map_location=device))
    model_1.to(device)
    model_1.eval()

    prediction = predict(model_1, input_img, labels1)

    st.image(np.array(input_img), caption=f"Хммм-мм {prediction}", use_column_width=True)

# Модель 2
elif selected_model == "предсказание 2":
    st.write("""
    # Классификатор новообразований на коже по фото:
    * доброкачественное
    * злокачественное
    """)

    st.write("""
    Рак кожи – это злокачественная эпителиальная опухоль, поражающая дерму и склонная к метастазированию.\n
    Главной причиной рака кожи считается длительное воздействие на эпителиальные клетки агрессивных доз ультрафиолета. \n
    Роль этого причинного фактора подтверждается тем, что почти в 90% случаев раковые кожные опухоли развиваются на открытых кожных покровах, \n
    в частности на лице и шее. Особенно рискую люди со светлым типом кожи.\n
    Помимо этого, значимыми факторами канцерогенеза являются некоторые химические вещества, длительно воздействующие на кожу.\n
    Частое травматическое повреждение рубцовых зон или доброкачественных невусов также может стать причиной рака кожи.\n
    У некоторых пациентов карциномы кожи могут иметь наследственную природу.\n
    """)
    uploaded_img = st.sidebar.file_uploader('Загрузите фотографию своей родинки и узнайте, есть ли повод волноваться', type=['jpeg'])
    if uploaded_img is not None:
        input_img = io.imread(uploaded_img)
    else:
        input_img = io.imread('https://cosmetologia-ufa.ru/upload/medialibrary/099/0999c7de833b5b8831c01b85e555c007.jpg')



    labels2 = {0: 'доброкачественная пупырка', 1: 'злокачественная штучка'}

    model_2.fc = nn.Linear(2048, 1, bias=True)
    model_2.load_state_dict(torch.load('/Users/id/Documents/strlit/08_05_strlit/resnext50_trained.pt', map_location=device))
    model_2.to(device)
    model_2.eval()

    resize = T.Resize((224, 224))
    img_pil = Image.fromarray(input_img)
    img = resize(T.ToTensor()(img_pil)/255)
    if model_2(img.to(device).unsqueeze(0)).item() > 0:
        prediction = 'злокачественная штучка'
    else:
        prediction = 'доброкачественная пупырка'

    st.image(np.array(input_img), caption=f"Хммм-мм {prediction}", use_column_width=True)