"""
CHƯƠNG TRÌNH CHÍNH (MENU) – chỉ cần chạy file này, chọn số rồi làm theo hướng dẫn.
Tự cài thư viện còn thiếu, tự chạy đúng thư mục, tự hỏi tên khi chụp ảnh.
"""
import os
import subprocess
import sys
import unicodedata

THU_MUC = os.path.dirname(os.path.abspath(__file__))
os.chdir(THU_MUC)                       # luôn chạy đúng thư mục chứa chương trình


def cv2_dung_duoc():
    """OpenCV 5.x không còn CascadeClassifier (bộ dò mặt) => phải dùng bản 4.13.x."""
    try:
        import cv2
        return hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data")
    except Exception:
        return False


def dam_bao_thu_vien():
    can = [("numpy", "numpy>=2.3"), ("cv2", "opencv-python>=4.13,<5"),
           ("matplotlib", "matplotlib")]
    thieu, go_cv2 = [], False
    for mod, goi in can:
        if mod == "cv2":
            if not cv2_dung_duoc():
                thieu.append(goi); go_cv2 = True
            continue
        try:
            __import__(mod)
        except Exception:
            thieu.append(goi)
    if thieu:
        print("Đang cài/sửa thư viện:", ", ".join(thieu), "(cần Internet, chờ vài phút)...")
        subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        if go_cv2:   # gỡ bản OpenCV không phù hợp trước khi cài bản đúng
            subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python",
                             "opencv-python-headless", "opencv-contrib-python"])
        ma = subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", *thieu])
        if ma != 0:
            print("\nCài thư viện thất bại. Hãy chụp màn hình lỗi phía trên.")
            print(f"Python đang dùng: {sys.version.split()[0]}. Cần Python 3.11 trở lên; nếu vẫn lỗi hãy cài Python 3.12 (py install 3.12) rồi chạy lại.")
            input("Nhấn Enter để thoát.")
            sys.exit(1)
        if not cv2_dung_duoc():
            input("OpenCV vẫn chưa đúng phiên bản. Hãy chạy lại chương trình một lần nữa. Nhấn Enter để thoát.")
            sys.exit(1)


def bo_dau(s):
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return "".join(c for c in s if c.isalnum())


def trang_thai():
    nguoi = []
    if os.path.isdir("data"):
        for t in sorted(os.listdir("data")):
            d = os.path.join("data", t)
            if os.path.isdir(d):
                nguoi.append((t, len([f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))])))
    print("\n" + "=" * 60)
    print(" TRẠNG THÁI HIỆN TẠI")
    if nguoi:
        print(f"  Dữ liệu: {len(nguoi)} người, {sum(n for _, n in nguoi)} ảnh")
        print("  " + ", ".join(f"{t}({n})" for t, n in nguoi))
    else:
        print("  Dữ liệu: CHƯA CÓ ảnh nào  -> hãy chọn 1 để chụp ảnh")
    print("  Mô hình: " + ("ĐÃ có (mo_hinh.npz)" if os.path.exists("mo_hinh.npz")
                           else "CHƯA có  -> chọn 2 sau khi chụp đủ ảnh"))
    print("=" * 60)


def chay(*lenh):
    subprocess.call([sys.executable, *lenh])


def tam_dung():
    input("\nNhấn Enter để quay lại menu...")


def main():
    dam_bao_thu_vien()
    if "--gui" in sys.argv:             # mở thẳng giao diện đồ họa
        chay("gui.py"); return
    while True:
        trang_thai()
        print("""
  1. Chụp ảnh khuôn mặt bằng webcam (mỗi người làm 1 lần)
  2. Huấn luyện + đánh giá (tạo mô hình, đồ thị, ngưỡng)
  3. Kiểm chứng lý thuyết (6 phép kiểm tra)
  4. DEMO nhận dạng bằng webcam
  5. Nhận dạng từ file ảnh (phương án dự phòng)
  6. MỞ GIAO DIỆN ĐỒ HỌA (khuyên dùng)
  0. Thoát
""")
        c = input("Chọn số: ").strip()
        if c == "1":
            ten = bo_dau(input("Nhập tên (ví dụ: Hieu): ").strip())
            if not ten:
                print("Tên không hợp lệ."); tam_dung(); continue
            so = input("Số ảnh cần chụp [20]: ").strip() or "20"
            print("\nHướng dẫn: nhấn SPACE để chụp, q để thoát. Đổi góc đầu, biểu cảm, ánh sáng giữa các lần chụp.")
            chay("buoc1_thu_thap_anh.py", "--ten", ten, "--so_anh", so if so.isdigit() else "20")
            tam_dung()
        elif c == "2":
            chay("buoc2_huan_luyen_danh_gia.py"); tam_dung()
        elif c == "3":
            chay("kiem_chung.py"); tam_dung()
        elif c == "4":
            if not os.path.exists("mo_hinh.npz"):
                print("Chưa có mô hình. Hãy chọn 2 trước.")
            else:
                print("Webcam đang mở. Nhấn q trong cửa sổ camera để thoát.")
                chay("buoc3_nhan_dang.py")
            tam_dung()
        elif c == "5":
            if not os.path.exists("mo_hinh.npz"):
                print("Chưa có mô hình. Hãy chọn 2 trước."); tam_dung(); continue
            f = input("Kéo thả file ảnh vào đây rồi nhấn Enter: ").strip().strip('"')
            if f:
                chay("buoc3_nhan_dang.py", "--anh", f)
            tam_dung()
        elif c == "6":
            chay("gui.py")
        elif c == "0":
            break


if __name__ == "__main__":
    main()
