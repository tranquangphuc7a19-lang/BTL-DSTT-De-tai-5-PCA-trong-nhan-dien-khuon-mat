# Hệ thống nhận dạng khuôn mặt sử dụng PCA (Eigenfaces)

## 1. Giới thiệu

Dự án xây dựng hệ thống nhận dạng khuôn mặt
sử dụng phương pháp Principal Component Analysis
(PCA), cụ thể là phương pháp Eigenfaces.

Chương trình hỗ trợ thu thập dữ liệu ảnh,
tiền xử lý dữ liệu, huấn luyện mô hình,
đánh giá hiệu suất và nhận dạng khuôn mặt
thông qua giao diện đồ họa.

## 2. Công nghệ sử dụng

- Python
- NumPy
- OpenCV
- Matplotlib
- Tkinter

## 3. Các chức năng chính

- Thu thập và tiền xử lý ảnh khuôn mặt.
- Xây dựng mô hình PCA (Eigenfaces).
- Huấn luyện và đánh giá mô hình.
- Nhận dạng khuôn mặt.
- Kiểm chứng các tính chất toán học.
- Giao diện đồ họa hỗ trợ người dùng.

## 4. Cấu trúc dự án

- pca_core.py: Triển khai thuật toán PCA.
- danh_gia.py: Huấn luyện và đánh giá mô hình.
- gui.py: Giao diện đồ họa.
- kiem_chung.py: Kiểm chứng tính chất toán học.
- duong_dan.py: Quản lý đường dẫn.
- nhap_thu_cong.py: Hỗ trợ nhập vùng khuôn mặt.
- buoc1_thu_thap_anh.py: Thu thập dữ liệu.
- buoc2_huan_luyen_danh_gia.py: Huấn luyện và đánh giá.
- buoc3_nhan_dang.py: Thực hiện nhận dạng.
- chay.py: Khởi chạy chương trình.

## 5. Cài đặt

Cài đặt Python phù hợp với môi trường dự án.

Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

## 6. Khởi chạy chương trình

Trên Windows, có thể chạy:

```bash
python chay.py
```

Hoặc chạy trực tiếp giao diện:

```bash
python gui.py
```

## 7. Mã nguồn

Toàn bộ mã nguồn được lưu trữ tại repository này.
