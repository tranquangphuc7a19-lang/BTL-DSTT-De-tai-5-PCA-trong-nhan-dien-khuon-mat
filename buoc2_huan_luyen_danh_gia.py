"""
BƯỚC 2: HUẤN LUYỆN + ĐÁNH GIÁ + CHỌN NGƯỠNG eps   (logic nằm trong danh_gia.py)
Cách chạy:   python buoc2_huan_luyen_danh_gia.py [--theta 0.95]
Kết quả:     thư mục ket_qua/ (các hình dùng cho báo cáo)  và  file mo_hinh.npz
"""
import argparse

import duong_dan as dd
from danh_gia import chay_danh_gia
from pca_core import THETA_MAC_DINH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=dd.DATA)
    ap.add_argument("--ti_le_train", type=float, default=0.7)
    ap.add_argument("--lan_lap", type=int, default=20)
    ap.add_argument("--k_toi_da", type=int, default=60)
    ap.add_argument("--theta", type=float, default=THETA_MAC_DINH, help="ngưỡng phương sai tích lũy (4.21)")
    ap.add_argument("--ra", default=dd.KET_QUA)
    ap.add_argument("--mo_hinh", default=dd.MO_HINH)
    a = ap.parse_args()
    try:
        chay_danh_gia(a.data, a.ra, a.mo_hinh, a.ti_le_train, a.lan_lap, a.k_toi_da, a.theta)
    except RuntimeError as e:
        raise SystemExit(f"Lỗi: {e}")


if __name__ == "__main__":
    main()
