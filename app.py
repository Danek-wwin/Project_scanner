import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2
from streamlit_cropper import st_cropper

# ===== 1. АРХИТЕКТУРА НЕЙРОСЕТИ =====
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ===== 2. ОБРАБОТКА ОДНОЙ ЦИФРЫ В КЛЕТКЕ =====
def process_single_digit(cell_image):
    img = cell_image.convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)
    
    # Бинаризация
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Убираем мелкий мусор
    kernel = np.ones((2,2), np.uint8)
    arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "_"
        
    # Берем самый большой контур в клетке
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    if w < 5 or h < 5:
        return "_"
        
    digit_arr = arr[y:y+h, x:x+w]
    
    # Вписываем в квадрат 28x28
    size = max(h, w)
    pad = int(size * 0.2)
    square = np.zeros((size + 2*pad, size + 2*pad), dtype=np.uint8)
    y_off = (size + 2*pad - h) // 2
    x_off = (size + 2*pad - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = digit_arr

    img_digit = Image.fromarray(square).resize((28, 28), Image.LANCZOS)
    tensor = transforms.ToTensor()(img_digit)
    tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)
    
    # Предсказание
    with torch.no_grad():
        t = tensor.unsqueeze(0)
        output = model(t)
        pred = output.argmax(dim=1).item()
        return str(pred)

# ===== 3. ИНТЕРФЕЙС =====
st.set_page_config(page_title="Сканер ОГЭ", page_icon="📝")
st.title("📝 Прототип сканера бланков ОГЭ")

st.write("Загрузите фото бланка или сделайте снимок. Программа автоматически разделит выделенную область на строки и клетки.")
st.info("⚠️ Для лучшего результата: фотографируйте бланк ровно, без теней, и выделяйте рамкой только сами строки с ответами (задания 1-12).")

# Фиксированные настройки для бланка ОГЭ
NUM_ROWS = 12
NUM_CELLS = 17

option = st.radio("Способ ввода:", ["📁 Загрузить файл", "📷 Сделать фото"], horizontal=True)
image = None

if option == "📁 Загрузить файл":
    uploaded_file = st.file_uploader("Выберите фото бланка...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    picture = st.camera_input("Сделайте снимок")
    if picture:
        image = Image.open(picture)

if image is not None:
    st.write("Шаг 1: Выделите рамкой блок с ответами (с 1 по 12 задание).")
    
    # Интерактивная обрезка
    cropped_image = st_cropper(
        image, 
        realtime_update=True, 
        box_color='#FF0000', 
        aspect_ratio=None
    )
    
    st.image(cropped_image, caption='Область для сканирования', use_container_width=True)
    
    if st.button("🚀 Начать сканирование"):
        with st.spinner('Распознаем ответы...'):
            width, height = cropped_image.size
            
            # Вычисляем размер одной строки и одной клетки
            row_height = height / NUM_ROWS
            cell_width = width / NUM_CELLS
            
            results = []
            
            # Проходим по всем строкам (сверху вниз)
            for r in range(NUM_ROWS):
                row_str = ""
                y_start = int(r * row_height)
                y_end = int((r + 1) * row_height)
                
                # Проходим по всем клеткам в строке (слева направо)
                for c in range(NUM_CELLS):
                    x_start = int(c * cell_width)
                    x_end = int((c + 1) * cell_width)
                    
                    # Вырезаем клетку
                    cell = cropped_image.crop((x_start, y_start, x_end, y_end))
                    
                    # Распознаем цифру
                    digit = process_single_digit(cell)
                    row_str += digit
                
                results.append(row_str)
            
            # Вывод результатов
            st.success("Сканирование завершено!")
            st.write("### Результаты по заданиям:")
            
            for i, res in enumerate(results):
                # Заменяем пустые клетки (прочерки) на пустоту для красоты
                clean_res = res.replace("_", "") 
                if clean_res == "":
                    st.write(f"**Задание {i+1}:** *нет ответа*")
                else:
                    st.write(f"**Задание {i+1}:** {clean_res}")
                    
            # Показываем сетку (для отладки)
            with st.expander("Показать, как программа нарезала бланк"):
                draw_img = cropped_image.copy()
                draw = ImageDraw.Draw(draw_img)
                
                # Рисуем горизонтальные линии строк
                for r in range(NUM_ROWS + 1):
                    y = int(r * row_height)
                    draw.line([(0, y), (width, y)], fill="red", width=2)
                
                # Рисуем вертикальные линии клеток
                for c in range(NUM_CELLS + 1):
                    x = int(c * cell_width)
                    draw.line([(x, 0), (x, height)], fill="blue", width=1)
                    
                st.image(draw_img, caption="Сетка нарезки", use_container_width=True)
