import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
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

# ===== 2. ОБРАБОТКА ЦИФРЫ ВНУТРИ КЛЕТКИ =====
def predict_digit(cell_image):
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
        return str(output.argmax(dim=1).item())

# ===== 3. УМНЫЙ ПОИСК КЛЕТОК НА БЛАНКЕ =====
def scan_oge_blank(image):
    img = image.convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)
    
    # Бинаризация для выделения линий
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ищем все контуры
    contours, _ = cv2.findContours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    # Примерные размеры клетки на бланке (зависит от разрешения фото)
    # Мы будем фильтровать по площади и соотношению сторон
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        
        # Клетка должна быть квадратной (или близкой к этому) и не слишком мелкой
        if 0.8 < aspect_ratio < 1.2 and area > 100: 
            cells.append((x, y, w, h))
            
    if not cells:
        return [], ""

    # Сортируем клетки: сначала по Y (сверху вниз), потом по X (слева направо)
    # Чтобы не путать строки, используем округление по Y
    cells.sort(key=lambda b: (round(b[1] / 10), b[0]))
    
    # Группируем в строки
    rows = []
    current_row = [cells[0]]
    
    for cell in cells[1:]:
        # Если клетка находится примерно на той же высоте, что и предыдущая
        if abs(cell[1] - current_row[-1][1]) < 20:
            current_row.append(cell)
        else:
            rows.append(current_row)
            current_row = [cell]
    rows.append(current_row)
    
    # Теперь распознаем цифры
    # ВАЖНО: Нам нужны только первые 17 клеток в каждой строке (номера 1-17)
    results = []
    for r_idx, row in enumerate(rows):
        if r_idx >= 12: # Берем только первые 12 строк (задания 1-12)
            break
            
        row_str = ""
        for i in range(min(17, len(row))): # Берем макс 17 клеток в строке
            x, y, w, h = row[i]
            cell_img = image.crop((x, y, x+w, y+h))
            digit = predict_digit(cell_img)
            row_str += digit
        results.append(row_str)
        
    return results, rows

# ===== 4. ИНТЕРФЕЙС =====
st.set_page_config(page_title="Сканер ОГЭ", page_icon="📝")
st.title("📝 Сканер бланков ОГЭ")

st.write("Загрузите фото бланка, чтобы программа автоматически нашла клетки и распознала ответы.")
st.info("⚠️ Важно: Для корректной работы сделайте ровное фото бланка при хорошем освещении (без теней).")

option = st.radio("Способ ввода:", ["📷 Сделать фото", "📁 Загрузить файл"], horizontal=True)
image = None

if option == "📷 Сделать фото":
    picture = st.camera_input("Сделайте снимок")
    if picture:
        image = Image.open(picture)
else:
    uploaded_file = st.file_uploader("Выберите фото бланка...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image is not None:
    st.write("Шаг 1: Выделите рамкой **только блок с ответами** (с 1 по 12 задание). Не захватывайте шапку бланка.")
    
    # Обрезаем область
    cropped_image = st_cropper(
        image, 
        realtime_update=True, 
        box_color='#FF0000', 
        aspect_ratio=None
    )
    
    st.image(cropped_image, caption='Область для сканирования', use_container_width=True)
    
    if st.button("🚀 Начать сканирование"):
        with st.spinner('Ищем клетки и распознаем...'):
            results, cells = scan_oge_blank(cropped_image)
            
            if not results:
                st.error("Не удалось найти клетки. Попробуйте выделить область ровнее.")
            else:
                st.success("Сканирование завершено!")
                st.write("### Результаты по заданиям:")
                for i, res in enumerate(results):
                    st.write(f"**Задание {i+1}:** {res}")
                    
                # Визуализация найденных клеток (для отладки)
                with st.expander("Показать, какие клетки нашла программа"):
                    draw_img = cropped_image.copy()
                    draw = ImageDraw.Draw(draw_img)
                    for (x, y, w, h) in cells:
                        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
                    st.image(draw_img, caption="Найденные клетки", use_container_width=True)
