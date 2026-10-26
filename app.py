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

# ===== 3. АВТО-ПОИСК БЛАНКА (С УМНЫМИ ОТСТУПАМИ) =====
def find_blank_and_answers(image, num_rows):
    """Находит бланк и область ответов, делая отступы в зависимости от размера строки."""
    img = image.convert('L')
    arr = np.array(img)
    
    # 1. Ищем сам бланк
    _, thresh = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    c = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(c)
    
    if bw < image.width * 0.3 or bh < image.height * 0.3:
        return None

    # 2. Ищем горизонтальные линии
    search_area = arr[by + int(bh*0.15) : by + int(bh*0.85), bx:bx+bw]
    _, thresh_lines = cv2.threshold(search_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(bw*0.4), 1))
    detected_lines = cv2.morphologyEx(thresh_lines, cv2.MORPH_OPEN, kernel, iterations=2)
    line_contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Базовые границы (если линии не найдены)
    table_top = by + int(bh * 0.40)
    table_bottom = by + int(bh * 0.85)

    if line_contours:
        lines = [(cv2.boundingRect(lc)[1] + by + int(bh*0.15), cv2.boundingRect(lc)[0]) for lc in line_contours]
        lines.sort(key=lambda x: x[0])
        
        for l_y, l_x in lines:
            if by + bh*0.30 < l_y < by + bh*0.55:
                table_top = l_y
                break
        for l_y, l_x in reversed(lines):
            if by + bh*0.75 < l_y < by + bh*0.92:
                table_bottom = l_y
                break

    # 3. ВЫЧИСЛЯЕМ ОТСТУПЫ
    # Высота всей таблицы
    table_height = table_bottom - table_top
    # Высота одной строки
    row_height = table_height / num_rows
    
    # Отступаем сверху ровно на 1 строку (чтобы пропустить заголовок "Результаты...")
    final_top = table_top + int(row_height)
    
    # Отступаем снизу тоже на 0.5 строки (чтобы не захватить "Замену ответов")
    final_bottom = table_bottom - int(row_height * 0.5)

    # 4. Левая и правая границы
    answer_x_start = bx + int(bw * 0.06) # Отступаем слева, чтобы пропустить номера заданий
    answer_x_end = bx + int(bw * 0.48)   # Отступаем справа, чтобы не захватить вторую колонку
    
    return (bx, by, bw, bh), (answer_x_start, final_top, answer_x_end, final_bottom)

# ===== 4. ИНТЕРФЕЙС =====
st.set_page_config(page_title="Авто-Сканер ОГЭ", page_icon="📝")
st.title("📝 Сканер бланков ОГЭ (Умные отступы)")

with st.sidebar:
    st.header("Настройки")
    num_rows = st.number_input("Сколько строк сканировать?", min_value=1, max_value=20, value=11)
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
        with st.spinner('Ищем бланк и распознаем ответы...'):
            
            # Передаем num_rows в функцию, чтобы она знала высоту строки
            result = find_blank_and_answers(image, num_rows)
            
            if not result:
                st.error("Не удалось найти бланк на фото. Сфотографируйте его на контрастном фоне.")
            else:
                blank_box, answer_box = result
                bx, by, bw, bh = blank_box
                ax1, ay1, ax2, ay2 = answer_box
                
                # Вырезаем область с ответами
                cropped_image = image.crop((ax1, ay1, ax2, ay2))
                
                w = ax2 - ax1
                h = ay2 - ay1
                
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
                        st.write(f"**Задание {i+1}:** {clean_res}")
                    else:
                        st.write(f"**Задание {i+1}:** *нет ответа*")
                
                # Визуализация
                with st.expander("Показать, как программа нашла бланк"):
                    draw_img = image.copy()
                    draw = ImageDraw.Draw(draw_img)
                    
                    # Синяя рамка — весь бланк
                    draw.rectangle([bx, by, bx+bw, by+bh], outline="blue", width=3)
                    
                    # Зеленая рамка — область с ответами (с нашими отступами)
                    draw.rectangle([ax1, ay1, ax2, ay2], outline="green", width=3)
                    
                    st.image(draw_img, caption="Синий — лист, Зеленый — область ответов с отступами", use_container_width=True)
