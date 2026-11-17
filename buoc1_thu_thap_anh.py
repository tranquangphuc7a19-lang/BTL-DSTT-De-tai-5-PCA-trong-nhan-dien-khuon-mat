"""
BƯỚC 1: THU THẬP ẢNH KHUÔN MẶT BẰNG WEBCAM (dòng lệnh; giao diện đồ họa có sẵn trong gui.py)
Cách chạy:   python buoc1_thu_thap_anh.py --ten Hieu --so_anh 20
Phím:  SPACE = chụp,  q = thoát
Mỗi lần chạy là MỘT PHIÊN chụp (mã phiên lưu trong tên file). Hãy chụp mỗi người ở ít nhất 2 phiên
(khác thời điểm / ánh sáng) để bước đánh giá chia train/test theo phiên, tránh ảnh gần trùng nhau.
Kết quả: data/<ten>/<ten>_s<phien>_<stt>.png  (ảnh xám 64x64, chỉ gồm vùng khuôn mặt).
"""
import argparse
import datetime
import os

import cv2

import duong_dan as dd
from pca_core import cat_mat, chuan_hoa_ten, ghi_anh, tao_bo_do_mat, tim_mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ten", required=True, help="tên người")
    ap.add_argument("--so_anh", type=int, default=20)
    ap.add_argument("--thu_muc", default=dd.DATA)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--khong_haar", action="store_true", help="không dùng Haar: chụp vùng trong khung vuông cố định giữa màn hình")
    a = ap.parse_args()

    ten = chuan_hoa_ten(a.ten)
    if not ten:
        raise SystemExit("Tên không hợp lệ.")
    thu_muc_nguoi = os.path.join(a.thu_muc, ten)
    os.makedirs(thu_muc_nguoi, exist_ok=True)
    phien = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    bo_do = None if a.khong_haar else tao_bo_do_mat()

    cam = cv2.VideoCapture(a.camera)
    if not cam.isOpened():
        raise SystemExit("Không mở được webcam. Hãy đóng các ứng dụng khác đang dùng camera.")
    dem = 0
    while dem < a.so_anh:
        ok, khung = cam.read()
        if not ok:
            break
        xam = cv2.cvtColor(khung, cv2.COLOR_BGR2GRAY)
        mat = tim_mat(bo_do, xam)
        hien_thi = khung.copy()
        if mat is not None:
            x, y, w, h = mat
            cv2.rectangle(hien_thi, (x, y), (x + w, y + h), (0, 255, 0) if bo_do is not None else (0, 255, 255), 2)
        cv2.putText(hien_thi, f"{ten}: {dem}/{a.so_anh}  SPACE=chup  q=thoat", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Thu thap anh", hien_thi)
        phim = cv2.waitKey(1) & 0xFF
        if phim == ord("q"):
            break
        if phim == 32 and mat is not None:
            duong_dan = os.path.join(thu_muc_nguoi, f"{ten}_s{phien}_{dem:03d}.png")
            ghi_anh(duong_dan, cat_mat(xam, mat))
            dem += 1
            print("Đã lưu", duong_dan)
    cam.release(); cv2.destroyAllWindows()
    print(f"Xong. Đã lưu {dem} ảnh (phiên {phien}) vào {thu_muc_nguoi}")


if __name__ == "__main__":
    main()
