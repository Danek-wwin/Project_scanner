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

# ===== 2. ОБРАБОТКА ОДНОЙ ЦИФРЫ =====
def process_single_digit(cell_image):
    img = cell_image.convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)
    
    # Бинаризация
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "_"
        
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
    
    with torch.no_grad():
        t = tensor.unsqueeze(0)
        output = model(t)
        pred = output.argmax(dim=1).item()
        return str(pred)

# ===== 3. УМНЫЙ ПОИСК КЛЕТОК (Игнорирует разные пробелы) =====
def find_cells_smart(image):
    """Находит квадратные клетки на бланке и группирует их в строки."""
    img = image.convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)
    
    # Делаем линии более четкими для поиска
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ищем контуры
    contours, _ = cv2.findContours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    # Фильтруем контуры: нам нужны только квадраты (клетки)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        
        # Клетка должна быть квадратной и достаточно большой
        if 0.7 < aspect_ratio < 1.3 and area > 300: 
            cells.append((x, y, w, h))
            
    if not cells:
        return [], []

    # Сортируем по Y (сверху вниз)
    cells.sort(key=lambda b: b[1])
    
    # Группируем в строки
    rows = []
    current_row = [cells[0]]
    
    for cell in cells[1:]:
        # Если центр клетки по Y близок к центру предыдущей, это одна строка
        prev_center_y = current_row[-1][1] + current_row[-1][3] / 2
        curr_center_y = cell[1] + cell[3] / 2
        
        if abs(curr_center_y - prev_center_y) < (cell[3] * 0.6): # Порог расстояния
            current_row.append(cell)
        else:
            # Сортируем клетки в строке по X (слева направо)
            current_row.sort(key=lambda b: b[0])
            rows.append(current_row)
            current_row = [cell]
            
    if current_row:
        current_row.sort(key=lambda b: b[0])
        rows.append(current_row)
        
    return rows, cells

# ===== 4. ИНТЕРФЕЙС =====
st.set_page_config(page_title="Сканер ОГЭ", page_icon="📝")
st.title("📝 Прототип сканера бланков ОГЭ (Умный поиск)")

st.write("Загрузите фото бланка. Программа сама найдет клетки и правильно сгруппирует их по строкам.")
st.info("⚠️ Фотографируйте бланк ровно, чтобы клетки были квадратными. Выделяйте только область с ответами.")

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
    st.write("Шаг 1: Выделите рамкой область с ответами (желательно только сами клетки).")
    
    cropped_image = st_cropper(
        image, 
        realtime_update=True, 
        box_color='#FF0000', 
        aspect_ratio=None
    )
    
    if st.button("🚀 Начать сканирование"):
        with st.spinner('Ищем клетки и распознаем...'):
            # Используем новую умную функцию
            rows, all_cells = find_cells_smart(cropped_image)
            
            if not rows:
                st.error("Клетки не найдены. Попробуйте выделить область ровнее или сфотографировать бланк без теней.")
            else:
                st.success(f"Найдено строк: {len(rows)}")
                
                results = []
                for row in rows:
                    row_str = ""
                    # Берем только первые 17 клеток в строке (если найдет больше)
                    for i in range(min(17, len(row))):
                        x, y, w, h = row[i]
                        cell_img = cropped_image.crop((x, y, x+w, y+h))
                        digit = process_single_digit(cell_img)
                        row_str += digit
                    results.append(row_str)
                
                # Вывод
                st.write("### Результаты по заданиям:")
                for i, res in enumerate(results):
                    clean_res = res.replace("_", "")
                    st.write(f"**Строка {i+1}:** {clean_res}")
                    
                # Отладка
                with st.expander("Показать, как программа нашла клетки"):
                    draw_img = cropped_image.copy()
                    draw = ImageDraw.Draw(draw_img)
                    for (x, y, w, h) in all_cells:
                        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
                    st.image(draw_img, caption="Найденные клетки", use_container_width=True)
