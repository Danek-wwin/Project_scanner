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

# ===== 3. АВТОМАТИЧЕСКИЙ ПОИСК СТРОК =====
def find_rows_automatically(image, expected_rows):
    """Находит горизонтальные строки на основе контуров клеток."""
    img = image.convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)
    
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Склеиваем пунктир, чтобы клетки стали целыми квадратами
    kernel = np.ones((3, 3), np.uint8)
    arr_closed = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(arr_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cells_y = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        # Ищем только квадратные объекты (клетки)
        if 0.7 < aspect_ratio < 1.3 and area > 300: 
            cells_y.append(y + h / 2) # Запоминаем центр клетки по Y
            
    if not cells_y:
        return None

    # Сортируем все найденные Y-координаты
    cells_y.sort()
    
    # Группируем близкие координаты (это одна строка)
    rows_y = []
    current_row = [cells_y[0]]
    
    for y in cells_y[1:]:
        if abs(y - current_row[-1]) < 30: # Если координаты близко — это одна строка
            current_row.append(y)
        else:
            rows_y.append(np.mean(current_row)) # Берем среднюю высоту строки
            current_row = [y]
    rows_y.append(np.mean(current_row))
    
    return rows_y

# ===== 4. ИНТЕРФЕЙС =====
st.set_page_config(page_title="Сканер ОГЭ", page_icon="📝")
st.title("📝 Прототип сканера бланков ОГЭ (Авто-поиск строк)")

with st.sidebar:
    st.header("Настройки сканера")
    num_rows = st.number_input("Сколько всего строк сканировать?", min_value=1, max_value=20, value=12)
    num_cells = st.number_input("Сколько клеток в строке?", min_value=1, max_value=25, value=17)

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
    st.warning("⚠️ Выделите рамкой **ВЕСЬ блок с ответами** (с 1 по 12 задание).")
    
    cropped_image = st_cropper(
        image, 
        realtime_update=True, 
        box_color='#FF0000', 
        aspect_ratio=None
    )
    
    if st.button("🚀 Начать сканирование"):
        with st.spinner('Ищем строки и распознаем...'):
            
            # 1. Программа сама находит, где находятся строки на фото
            found_row_centers = find_rows_automatically(cropped_image, num_rows)
            
            if not found_row_centers:
                st.error("Не удалось найти клетки для авто-разметки. Попробуйте сфотографировать ровнее.")
            else:
                width, height = cropped_image.size
                cell_width = width / num_cells
                
                results = []
                
                # 2. Идем по найденным центрам строк и вырезаем их
                # Если программа нашла больше строк, чем нужно, берем первые num_rows
                actual_rows = min(len(found_row_centers), num_rows)
                
                for r in range(actual_rows):
                    row_str = ""
                    center_y = found_row_centers[r]
                    
                    # Высота строки — это примерно ширина клетки (так как она квадратная)
                    row_height = cell_width 
                    y_start = int(center_y - row_height / 2)
                    y_end = int(center_y + row_height / 2)
                    
                    # Защита от выхода за границы картинки
                    y_start = max(0, y_start)
                    y_end = min(height, y_end)
                    
                    for c in range(num_cells):
                        x_start = int(c * cell_width)
                        x_end = int((c + 1) * cell_width)
                        
                        cell = cropped_image.crop((x_start, y_start, x_end, y_end))
                        digit = process_single_digit(cell)
                        row_str += digit
                    
                    results.append(row_str)
                
                st.success(f"Распознано строк: {len(results)}")
                st.write("### Результаты по заданиям:")
                
                for i, res in enumerate(results):
                    clean_res = res.replace("_", "")
                    if clean_res:
                        st.write(f"**Задание {i+1}:** {clean_res}")
                    else:
                        st.write(f"**Задание {i+1}:** *нет ответа*")
                        
                # Визуализация (рисуем линии там, где программа нашла строки)
                with st.expander("Показать, как программа нашла строки"):
                    draw_img = cropped_image.copy()
                    draw = ImageDraw.Draw(draw_img)
                    
                    for center_y in found_row_centers[:num_rows]:
                        y = int(center_y)
                        draw.line([(0, y), (width, y)], fill="red", width=3)
                        
                    st.image(draw_img, caption="Красные линии — центры найденных строк", use_container_width=True)
