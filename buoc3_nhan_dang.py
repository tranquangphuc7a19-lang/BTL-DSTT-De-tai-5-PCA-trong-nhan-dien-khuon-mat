"""
BƯỚC 3: ỨNG DỤNG NHẬN DẠNG KHUÔN MẶT (dòng lệnh; giao diện đồ họa có sẵn trong gui.py)
  Webcam (thời gian thực):   python buoc3_nhan_dang.py
  Ảnh tĩnh (dự phòng):       python buoc3_nhan_dang.py --anh anh1.jpg anh2.jpg
Phím:  q = thoát.   Cần có mo_hinh.npz (tạo bởi bước 2).
"""
import argparse

import cv2

import duong_dan as dd
from pca_core import MoHinh, anh_mat_tu_anh_ngoai, doc_anh, tao_bo_do_mat, tim_mat

XANH, DO = (0, 200, 0), (0, 0, 255)


def xu_ly_khung(khung, bo_do, mo_hinh, min_size=80):
    """Xác định vùng mặt (Haar, hoặc khung cố định nếu bo_do=None) -> cắt -> nhận dạng -> vẽ lên khung. Trả về (tên|None, d_min) hoặc None."""
    xam = cv2.cvtColor(khung, cv2.COLOR_BGR2GRAY)
    mat = tim_mat(bo_do, xam, min_size)
    if mat is None:
        cv2.putText(khung, "Khong thay khuon mat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DO, 2)
        return None
    x, y, w, h = mat
    ten, d = mo_hinh.nhan_dang(xam[y:y + h, x:x + w])
    mau = XANH if ten is not None else DO
    nhan = f"{ten}  d={d:.2f}" if ten is not None else f"NGUOI LA  d={d:.2f}"
    cv2.rectangle(khung, (x, y), (x + w, y + h), mau, 2)
    cv2.putText(khung, nhan, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mau, 2)
    return ten, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mo_hinh", default=dd.MO_HINH)
    ap.add_argument("--anh", nargs="*")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--khong_haar", action="store_true", help="không dùng Haar: webcam dùng khung vuông cố định, ảnh tĩnh xem là ảnh mặt")
    a = ap.parse_args()

    mo_hinh = MoHinh(a.mo_hinh)
    print(f"Mô hình: k = {mo_hinh.k}, ε = {mo_hinh.eps:.3f}, {len(mo_hinh.ten)} người: {mo_hinh.ten}")
    bo_do = None if a.khong_haar else tao_bo_do_mat()

    if a.anh:
        for duong_dan in a.anh:
            khung = doc_anh(duong_dan, cv2.IMREAD_COLOR)
            if khung is None:
                print("Không đọc được", duong_dan); continue
            if bo_do is None:      # ảnh tĩnh, không Haar: dùng đúng đường xử lý ảnh ngoài như lúc nhập dữ liệu
                kq = mo_hinh.nhan_dang(anh_mat_tu_anh_ngoai(cv2.cvtColor(khung, cv2.COLOR_BGR2GRAY)))
            else:
                kq = xu_ly_khung(khung, bo_do, mo_hinh, min_size=40)
            print(duong_dan, "->", kq if kq else "không thấy mặt")
            cv2.imshow("Ket qua (nhan phim bat ky de tiep tuc)", khung); cv2.waitKey(0)
        cv2.destroyAllWindows(); return

    cam = cv2.VideoCapture(a.camera)
    if not cam.isOpened():
        raise SystemExit("Không mở được webcam. Hãy dùng chế độ --anh để dự phòng.")
    while True:
        ok, khung = cam.read()
        if not ok:
            break
        xu_ly_khung(khung, bo_do, mo_hinh)
        cv2.imshow("Nhan dang khuon mat bang PCA (q = thoat)", khung)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
