
# HỆ THỐNG NHẬN DẠNG KHUÔN MẶT SỬ DỤNG PCA (EIGENFACES)

## 1. Giới thiệu

Dự án xây dựng hệ thống nhận dạng khuôn mặt sử dụng phương pháp Principal Component Analysis (PCA), cụ thể là thuật toán Eigenfaces.

Chương trình hỗ trợ thu thập dữ liệu ảnh, huấn luyện mô hình, đánh giá kết quả và nhận dạng khuôn mặt thông qua giao diện người dùng.

## 2. Cấu trúc mã nguồn

| STT | Tên file | Chức năng |
|---|---|---|
| 1 | [buoc1_thu_thap_anh.py](buoc1_thu_thap_anh.py) | Thu thập dữ liệu ảnh khuôn mặt |
| 2 | [buoc2_huan_luyen_danh_gia.py](buoc2_huan_luyen_danh_gia.py) | Huấn luyện và đánh giá mô hình |
| 3 | [buoc3_nhan_dang.py](buoc3_nhan_dang.py) | Thực hiện nhận dạng khuôn mặt |
| 4 | [chay.py](chay.py) | Khởi chạy và điều phối chương trình |
| 5 | [danh_gia.py](danh_gia.py) | Thực hiện đánh giá mô hình |
| 6 | [duong_dan.py](duong_dan.py) | Quản lý đường dẫn dữ liệu và tài nguyên |
| 7 | [gui.py](gui.py) | Xây dựng giao diện đồ họa |
| 8 | [kiem_chung.py](kiem_chung.py) | Kiểm chứng các tính chất toán học |
| 9 | [nhap_thu_cong.py](nhap_thu_cong.py) | Hỗ trợ nhập vùng khuôn mặt thủ công |
| 10 | [pca_core.py](pca_core.py) | Cài đặt thuật toán PCA và Eigenfaces |
| 11 | [requirements.txt](requirements.txt) | Danh sách thư viện cần thiết |

## 3. Công nghệ sử dụng

- Python
- NumPy
- OpenCV
- Matplotlib
- Tkinter

## 4. Cài đặt môi trường

### Bước 1: Tải mã nguồn

```bash
git clone https://github.com/tranquangphuc7a19-lang/BTL-DSTT-de-tai-5.git
```

Hoặc tải trực tiếp repository dưới dạng ZIP.

### Bước 2: Di chuyển vào thư mục dự án

```bash
cd BTL-DSTT-de-tai-5
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## 5. Khởi chạy chương trình

Có thể khởi chạy chương trình bằng lệnh:

```bash
python chay.py
```

Hoặc chạy trực tiếp giao diện:

```bash
python gui.py
```

## 6. Repository mã nguồn

Toàn bộ mã nguồn của dự án được lưu trữ tại:

[BTL-DSTT-de-tai-5 - GitHub](https://github.com/tranquangphuc7a19-lang/BTL-DSTT-de-tai-5)

