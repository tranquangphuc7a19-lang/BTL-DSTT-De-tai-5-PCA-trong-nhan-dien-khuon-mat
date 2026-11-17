"""
ĐƯỜNG DẪN DÙNG CHUNG
Khi chạy bằng Python:  thư mục gốc = thư mục chứa mã nguồn.
Khi đã đóng gói .exe:  thư mục gốc = thư mục chứa file .exe  (data/, mo_hinh.npz, ket_qua/ nằm cạnh file exe,
                        KHÔNG nằm trong thư mục tạm của PyInstaller để dữ liệu không bị mất khi tắt chương trình).
"""
import os
import sys


def thu_muc_goc():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


GOC = thu_muc_goc()
DATA = os.path.join(GOC, "data")
MO_HINH = os.path.join(GOC, "mo_hinh.npz")
KET_QUA = os.path.join(GOC, "ket_qua")
