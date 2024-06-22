import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import streamlit as st

# Загрузка модели
model = load_model('mnist_cnn_mod.h5')

# Функция для предобработки изображения
def preprocess_image(img):
    img = img.resize((28, 28))  # Изменение размера до 28x28
    img = img.convert('L')  # Преобразование в оттенки серого
    
    # Удаление фона
    img_np = np.array(img)
    _, binary_img = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Поиск контуров
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Создание маски для заливки контуров
    mask = np.zeros_like(binary_img)
    
    # Закрашивание контуров
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # Применение маски к исходному изображению
    img_filled = cv2.bitwise_and(binary_img, mask)
    
    img_filled_pil = Image.fromarray(img_filled)
    
    # Нормализация изображения
    img_array = np.array(img_filled_pil) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))  # Добавление размерности канала
    return img, img_filled_pil, img_array

# Функция для загрузки и предсказания
def predict_digit(img):
    img, img_filled, img_array = preprocess_image(img)
    
    # Предсказание класса с использованием модели
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    return img, img_filled, predicted_class

# Streamlit приложение
st.title('Рисование цифр и предсказание')
st.write('Нарисуйте цифру внизу, затем нажмите кнопку "Предсказать".')

# Виджет для рисования
canvas_result = st.canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
    img = img.convert('L')  # Преобразование в оттенки серого
    
    if st.button('Предсказать'):
        img, img_filled, predicted_digit = predict_digit(img)
        
        # Показ исходного и предобработанного изображений
        st.image(img, caption='Исходное изображение', use_column_width=True)
        st.image(img_filled, caption='Изображение без фона и закрашенное', use_column_width=True)

        st.write(f'Предсказанная цифра: {predicted_digit}')
