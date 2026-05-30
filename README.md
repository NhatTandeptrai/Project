# README — Hướng dẫn chạy Phần 1 và Phần 2

## 1. Cần tải/cài trước

Trước khi chạy đồ án, cần cài các phần mềm sau:

### 1.1. Python

Cài Python phiên bản **3.10 trở lên**.

Kiểm tra Python đã cài chưa bằng lệnh:

```bash
python --version
```

Nếu máy báo ra phiên bản Python, ví dụ:

```text
Python 3.10.x
```

hoặc cao hơn là được.

---

### 1.2. Visual Studio Code

Cài **Visual Studio Code** để mở project và chạy notebook.

Sau khi cài VS Code, mở mục **Extensions** và cài thêm:

```text
Python
Jupyter
```

Hai extension này dùng để chọn kernel và chạy file `.ipynb`.

---

## 2. Cấu trúc thư mục cần có

Nên để project theo cấu trúc sau:

```text
TUDTK-P2/
│
├── README.md
├── requirements.txt
│
├── part1/
│   ├── linalg_utils.py
│   ├── ols_implementation.py
│   ├── ridge_lasso.py
│   ├── residual_analysis.py
│   ├── cross_validation.py
│   └── part1_notebook.ipynb
│
└── part 2/
    ├── data/
    │   └── Melbourne_housing_FULL.csv
    ├── data_pipeline.py
    ├── model_comparison.py
    └── part2_notebook.ipynb
```

Lưu ý quan trọng:

- File dữ liệu phải nằm trong thư mục:

```text
part 2/data/Melbourne_housing_FULL.csv
```

- Nếu thư mục của bạn tên là `part2` thay vì `part 2`, cần sửa lại đường dẫn tương ứng trong notebook.

---

## 3. Mở project trong VS Code

Mở VS Code, chọn:

```text
File → Open Folder
```

Sau đó chọn thư mục gốc của project:

```text
TUDTK-P2
```

Không nên mở riêng từng file `.ipynb`, vì như vậy VS Code có thể không nhận đúng đường dẫn giữa `part1` và `part 2`.

---

## 4. Tạo môi trường ảo `.venv`

Mở Terminal trong VS Code:

```text
Terminal → New Terminal
```

Đảm bảo Terminal đang đứng ở thư mục gốc `TUDTK-P2`.

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
.\.venv\Scripts\activate
```

Nếu kích hoạt thành công, Terminal sẽ hiện dạng:

```text
(.venv)
```

---

## 5. Cài thư viện cần thiết

Sau khi đã kích hoạt `.venv`, cài các thư viện cần thiết:

```bash
pip install pandas matplotlib seaborn scikit-learn jupyter ipykernel numpy statsmodels
```

Nếu có file `requirements.txt`, có thể dùng:

```bash
pip install -r requirements.txt
```

---

## 6. Đăng ký kernel cho notebook

Chạy lệnh sau trong Terminal đang kích hoạt `.venv`:

```bash
python -m ipykernel install --user --name tudtk_project --display-name "Python (TUDTK Project)"
```

Sau khi chạy xong, VS Code sẽ có kernel tên:

```text
Python (TUDTK Project)
```

Nếu chưa thấy kernel này, bấm nút refresh trong phần chọn kernel hoặc reload lại VS Code.

---

## 7. Chạy Phần 1

### 7.1. Mở notebook Phần 1

Mở file:

```text
part1/part1_notebook.ipynb
```

Ở góc phải phía trên notebook, bấm **Select Kernel** và chọn:

```text
Python (TUDTK Project)
```

Nếu không thấy, chọn kernel trỏ tới:

```text
.venv\Scripts\python.exe
```

---

### 7.2. Chạy notebook

Sau khi chọn kernel, bấm:

```text
Run All
```

hoặc chạy từng cell từ trên xuống.

Nên chạy từ đầu đến cuối, không nên chạy nhảy cell để tránh thiếu biến.

---

### 7.3. Nếu Phần 1 lỗi import

Nếu gặp lỗi:

```text
ModuleNotFoundError: No module named 'linalg_utils'
```

thêm cell sau vào đầu notebook Phần 1:

```python
import os
import sys

PART1_DIR = os.getcwd()

if PART1_DIR not in sys.path:
    sys.path.insert(0, PART1_DIR)

print("Current working directory:", os.getcwd())
```

Sau đó chạy lại notebook từ đầu.

---

## 8. Chạy Phần 2

### 8.1. Mở notebook Phần 2

Mở file:

```text
part 2/part2_notebook.ipynb
```

Chọn kernel:

```text
Python (TUDTK Project)
```

---

### 8.2. Kiểm tra file dữ liệu

Trong thư mục `part 2`, phải có file:

```text
data/Melbourne_housing_FULL.csv
```

Nếu notebook có dòng:

```python
df = pd.read_csv("data/Melbourne_housing_FULL.csv")
```

thì file dữ liệu phải đặt đúng vị trí trên.

---

### 8.3. Kiểm tra working directory

Ở đầu notebook Phần 2, nên chạy cell sau:

```python
import os

print(os.getcwd())
```

Kết quả đúng nên kết thúc bằng:

```text
part 2
```

Nếu kết quả chưa đúng, thêm cell này ở đầu notebook:

```python
import os

os.chdir(r"C:\duong_dan_cua_ban\TUDTK-P2\part 2")
print(os.getcwd())
```

Ví dụ:

```python
os.chdir(r"C:\1. Các môn học\6. TUDTK\1. Đồ án\TUDTK-P2\part 2")
print(os.getcwd())
```

---

### 8.4. Thêm đường dẫn import Phần 1

Trong notebook Phần 2, trước khi import `model_comparison`, thêm cell sau:

```python
import os
import sys

PART2_DIR = os.getcwd()
BASE_DIR = os.path.abspath("..")
PART1_DIR = os.path.join(BASE_DIR, "part1")

for path in [PART2_DIR, BASE_DIR, PART1_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

print("PART2_DIR:", PART2_DIR)
print("BASE_DIR:", BASE_DIR)
print("PART1_DIR:", PART1_DIR)
```

Sau đó import các hàm cần dùng:

```python
from model_comparison import run_ols_basic, run_ols_selected_by_vif, run_ridge_cv

from linalg_utils import transpose, matmul, matvec, mat_add, mat_scale, eye, inv
from residual_analysis import cooks_distance, norm_ppf
```

---

### 8.5. Chạy notebook Phần 2

Sau khi kiểm tra đường dẫn và chọn đúng kernel, bấm:

```text
Restart Kernel → Run All
```

Nên chạy toàn bộ notebook từ đầu đến cuối.

---

## 9. Lỗi thường gặp khi chạy Phần 2

### 9.1. Lỗi không tìm thấy file dữ liệu

Lỗi thường gặp:

```text
FileNotFoundError: data/Melbourne_housing_FULL.csv
```

Cách sửa:

- Kiểm tra file có đúng tên `Melbourne_housing_FULL.csv` không.
- Kiểm tra file có nằm trong `part 2/data/` không.
- Kiểm tra `os.getcwd()` có đang ở thư mục `part 2` không.

---

### 9.2. Lỗi không tìm thấy module Phần 1

Lỗi:

```text
ModuleNotFoundError: No module named 'linalg_utils'
```

Cách sửa:

Chạy lại cell thêm đường dẫn ở mục 8.4.

---

### 9.3. Lỗi không có thư viện

Ví dụ:

```text
ModuleNotFoundError: No module named 'pandas'
```

Cách sửa:

Kiểm tra đã chọn đúng kernel `.venv`, sau đó chạy lại:

```bash
pip install pandas matplotlib seaborn scikit-learn jupyter ipykernel numpy statsmodels
```

---

## 10. Thứ tự chạy khuyến nghị

Nên chạy theo thứ tự sau:

```text
1. Mở VS Code bằng thư mục TUDTK-P2
2. Tạo và kích hoạt .venv
3. Cài các thư viện cần thiết
4. Đăng ký kernel Python (TUDTK Project)
5. Chạy part1_notebook.ipynb
6. Chạy part2_notebook.ipynb
7. Kiểm tra bảng kết quả, biểu đồ và xuất báo cáo
```

---

## 11. Ghi chú

- Không nên chạy notebook bằng cách nhảy cell tùy ý.
- Nếu đổi vị trí thư mục hoặc đổi tên `part 2`, cần sửa lại đường dẫn trong notebook.
- Nếu thay đổi file code Phần 1, nên chạy lại toàn bộ Phần 2 để kết quả mô hình được cập nhật.
- Nếu bảng kết quả mô hình thay đổi, cần cập nhật lại số liệu trong báo cáo.
