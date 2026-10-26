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

# ===== 2. ФУНКЦИЯ ОБРАБОТКИ ОДНОЙ ЦИФРЫ =====
def process_single_digit(cell_image):
    img = cell_image.convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)
    
    # Бинаризация (Оцу)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Убираем мелкий мусор
    kernel = np.ones((2,2), np.uint8)
    arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    if w < 5 or h < 5:
        return None
        
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
    
    return tensor

# ===== 3. ИНТЕРФЕЙС =====
st.set_page_config(page_title="Сканер бланков ОГЭ", page_icon="📝")
st.title("📝 Прототип сканера бланков ОГЭ (Мультистрок)")

with st.sidebar:
    st.header("Настройки сканера")
    num_rows = st.number_input("Сколько строк сканировать?", min_value=1, max_value=20, value=10)
    num_cells = st.number_input("Сколько клеток в строке?", min_value=1, max_value=10, value=5)

st.write("---")
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
    st.info(f"Выделите рамкой **всю область ответов** (все {num_rows} строк).")
    
    # Обрезка всего блока ответов
    cropped_image = st_cropper(
        image, 
        realtime_update=True, 
        box_color='#FF0000', 
        aspect_ratio=None
    )
    
    st.image(cropped_image, caption='Выделенный блок ответов', use_container_width=True)
    
    if st.button("🚀 Запустить сканирование"):
        with st.spinner('Сканируем бланк...'):
            width, height = cropped_image.size
            
            # Вычисляем высоту одной строки
            row_height = height / num_rows
            # Ширина одной клетки
            cell_width = width / num_cells
            
            results = []
            
            # Проходим по каждой строке (сверху вниз)
            for r in range(num_rows):
                row_str = ""
                y_start = int(r * row_height)
                y_end = int((r + 1) * row_height)
                
                # Проходим по каждой клетке в строке (слева направо)
                for c in range(num_cells):
                    x_start = int(c * cell_width)
                    x_end = int((c + 1) * cell_width)
                    
                    cell = cropped_image.crop((x_start, y_start, x_end, y_end))
                    tensor = process_single_digit(cell)
                    
                    if tensor is not None:
                        with torch.no_grad():
                            t = tensor.unsqueeze(0)
                            output = model(t)
                            pred = output.argmax(dim=1).item()
                            row_str += str(pred)
                    else:
                        row_str += "_" # Если клетка пустая
                
                results.append(row_str)
            
            # Вывод результатов
            st.success("### Результаты сканирования:")
            for i, res in enumerate(results):
                st.write(f"**Задание {i+1}:** {res}")
