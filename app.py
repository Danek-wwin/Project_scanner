import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2
import easyocr  # <-- Импортируем OCR для поиска текста

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

# ===== 2. ЗАГРУЗКА OCR (Только один раз) =====
@st.cache_resource
def load_ocr():
    # Загружаем модель для русского и английского языка (без GPU)
    return easyocr.Reader(['ru', 'en'], gpu=False)

# ===== 3. ОБРАБОТКА ОДНОЙ ЦИФРЫ =====
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

# ===== 4. ИНТЕРФЕЙС =====
st.set_page_config(page_title="Умный Сканер ОГЭ", page_icon="📝")
st.title("📝 Сканер бланков ОГЭ (Поиск по тексту)")

with st.sidebar:
    st.header("Настройки")
    num_rows = st.number_input("Сколько строк сканировать?", min_value=1, max_value=20, value=12)
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
        with st.spinner('Ищем текст на бланке и распознаем ответы...'):
            
            reader = load_ocr()
            
            # Конвертируем картинку в формат для OCR
            img_np = np.array(image)
            
            # Ищем весь текст на картинке
            results_ocr = reader.readtext(img_np)
            
            # Ищем нашу фразу-якорь
            anchor_y = None
            for (bbox, text, prob) in results_ocr:
                # Ищем по ключевым словам, чтобы не зависеть от идеального распознавания
                text_lower = text.lower()
                if "задание" in text_lower and "выполняется" in text_lower and "бланке" in text_lower:
                    # bbox — это координаты углов найденного текста
                    # bbox[0][1] — это верхняя Y-координата текста
                    anchor_y = int(bbox[0][1])
                    st.success(f"Найден ориентир: «{text}»")
                    break
            
            if anchor_y is None:
                st.error("Не удалось найти текст «Задание выполняется...». Попробуйте сфотографировать бланк ровнее или при лучшем свете.")
            else:
                width, height = image.size
                
                # Начинаем сканировать чуть ниже найденного текста (добавляем отступ)
                y_start = anchor_y + 40 
                
                # Ограничиваем область по горизонтали (левая половина бланка)
                x_start = 0
                x_end = int(width * 0.5)
                
                # Вычисляем высоту строки
                # Предполагаем, что общая высота блока с ответами (до замены) ~ 60% экрана
                # Но мы просто отступим вниз на нужное количество строк
                # Высоту одной строки возьмем примерно 3.5% от высоты фото (зависит от фото)
                row_height = int(height * 0.035) 
                
                # Для надежности вычислим ширину клетки
                cell_width = (x_end - x_start) / num_cells
                
                results = []
                draw_img = image.copy()
                draw = ImageDraw.Draw(draw_img)
                
                for r in range(num_rows):
                    y1 = y_start + r * row_height
                    y2 = y1 + row_height
                    
                    # Если вышли за пределы фото — стоп
                    if y2 > height:
                        break
                        
                    # Рисуем линию для отладки
                    draw.line([(x_start, y1), (x_end, y1)], fill="red", width=2)
                    
                    row_str = ""
                    for c in range(num_cells):
                        x1 = int(x_start + c * cell_width)
                        x2 = int(x1 + cell_width)
                        
                        cell = image.crop((x1, y1, x2, y2))
                        digit = process_single_digit(cell)
                        row_str += digit
                        
                    results.append(row_str)
                
                # Вывод
                st.success("Сканирование завершено!")
                st.write("### Результаты:")
                
                for i, res in enumerate(results):
                    clean_res = res.replace("_", "")
                    if clean_res:
                        st.write(f"**Задание {i+1}:** {clean_res}")
                    else:
                        st.write(f"**Задание {i+1}:** *нет ответа*")
                
                with st.expander("Показать, как программа нашла строки"):
                    st.image(draw_img, caption="Красные линии — где программа искала ответы", use_container_width=True)
