HỆ THỐNG NHẬN DẠNG KHUÔN MẶT SỬ DỤNG PCA

1. Giới thiệu

Đề tài xây dựng chương trình nhận dạng khuôn mặt sử dụng thuật toán PCA (Principal Component Analysis), hay còn gọi là phương pháp Eigenfaces.

Chương trình hỗ trợ thu thập ảnh khuôn mặt, huấn luyện mô hình và nhận dạng khuôn mặt từ dữ liệu đầu vào.

2. Công nghệ sử dụng

Ngôn ngữ lập trình: Python

Thư viện: OpenCV, NumPy và các thư viện được liệt kê trong requirements.txt

Thuật toán: PCA (Eigenfaces)

3. Cấu trúc dự án

buoc1_thu_thap_anh.py: Thu thập ảnh khuôn mặt.

buoc2_huan_luyen_danh_gia.py: Huấn luyện và đánh giá mô hình.

buoc3_nhan_dang.py: Nhận dạng khuôn mặt.

chay.py: Chạy chương trình.

danh_gia.py: Chức năng đánh giá.

duong_dan.py: Quản lý đường dẫn.

gui.py: Giao diện chương trình.

kiem_chung.py: Chức năng kiểm chứng.

nhap_thu_cong.py: Nhập dữ liệu thủ công.

pca_core.py: Phần xử lý thuật toán PCA.

requirements.txt: Danh sách thư viện cần cài đặt.

4. Kết quả thực nghiệm

Các hình ảnh giao diện, kết quả nhận dạng, biểu đồ và dữ liệu thực nghiệm được lưu trong các thư mục tương ứng:

results/

figures/

data/

video/

5. Mục đích

Dự án được thực hiện phục vụ học tập và nghiên cứu thuật toán PCA trong bài toán nhận dạng khuôn mặt.
