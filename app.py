import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2

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

# ===== 3. АВТОПОИСК БЛОКА ОТВЕТОВ =====
def find_answer_grid(image):
    """Находит координаты (x, y, w, h) самой большой группы клеток на фото."""
    img = image.convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)
    
    # Бинаризация
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ищем контуры
    contours, _ = cv2.findContours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Собираем все квадратные объекты (клетки)
    squares = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        if 0.8 < aspect_ratio < 1.2 and area > 100: # Фильтр квадратов
            squares.append((x, y, w, h))
            
    if not squares:
        return None
        
    # Находим самый большой кластер клеток (по площади)
    # Просто берем минимальные и максимальные координаты всех найденных квадратов
    x_coords = [s[0] for s in squares]
    y_coords = [s[1] for s in squares]
    x_ends = [s[0] + s[2] for s in squares]
    y_ends = [s[1] + s[3] for s in squares]
    
    min_x, max_x = min(x_coords), max(x_ends)
    min_y, max_y = min(y_coords), max(y_ends)
    
    # Немного отступаем, чтобы захватить края
    pad = 10
    return (max(0, min_x - pad), max(0, min_y - pad), 
            min(image.width, max_x + pad) - max(0, min_x - pad), 
            min(image.height, max_y + pad) - max(0, min_y - pad))

# ===== 4. ИНТЕРФЕЙС =====
st.set_page_config(page_title="Авто-Сканер ОГЭ", page_icon="📝")
st.title("📝 Сканер бланков ОГЭ (Авто-поиск)")

with st.sidebar:
    st.header("Настройки")
    num_rows = st.number_input("Сколько строк сканировать?", min_value=1, max_value=20, value=10)
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
    if st.button("🚀 Начать сканирование"):
        with st.spinner('Ищем область с ответами и распознаем...'):
            
            # 1. Автоматически находим область с ответами
            grid_box = find_answer_grid(image)
            
            if not grid_box:
                st.error("Не удалось найти клетки на фото. Сфотографируйте бланк ближе и при хорошем свете.")
            else:
                x, y, w, h = grid_box
                
                # 2. Вырезаем найденную область
                cropped_image = image.crop((x, y, x+w, y+h))
                
                # 3. Нарезаем на строки и клетки
                row_height = h / num_rows
                cell_width = w / num_cells
                
                results = []
                
                for r in range(num_rows):
                    row_str = ""
                    y_start = int(r * row_height)
                    y_end = int((r + 1) * row_height)
                    
                    for c in range(num_cells):
                        x_start = int(c * cell_width)
                        x_end = int((c + 1) * cell_width)
                        
                        cell = cropped_image.crop((x_start, y_start, x_end, y_end))
                        digit = process_single_digit(cell)
                        row_str += digit
                        
                    results.append(row_str)
                
                # Вывод
                st.success("Сканирование завершено!")
                st.write("### Результаты:")
                
                for i, res in enumerate(results):
                    clean_res = res.replace("_", "")
                    if clean_res:
                        st.write(f"**Строка {i+1}:** {clean_res}")
                
                # Визуализация
                with st.expander("Показать найденную область и сетку"):
                    draw_img = image.copy()
                    draw = ImageDraw.Draw(draw_img)
                    
                    # Рисуем общую рамку бланка
                    draw.rectangle([x, y, x+w, y+h], outline="green", width=3)
                    
                    # Рисуем сетку внутри
                    for r in range(num_rows + 1):
                        line_y = y + int(r * row_height)
                        draw.line([(x, line_y), (x+w, line_y)], fill="red", width=1)
                        
                    for c in range(num_cells + 1):
                        line_x = x + int(c * cell_width)
                        draw.line([(line_x, y), (line_x, y+h)], fill="blue", width=1)
                        
                    st.image(draw_img, caption="Зеленая рамка — область, которую нашла программа", use_container_width=True)
