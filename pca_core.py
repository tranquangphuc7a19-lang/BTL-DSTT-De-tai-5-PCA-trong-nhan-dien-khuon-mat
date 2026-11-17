"""
LÕI THUẬT TOÁN PCA / EIGENFACES  (bám sát Mục 4, Mục 5.2 và 5.3 của báo cáo)
=============================================================================
Ký hiệu giống báo cáo:
    D = H*W      số chiều của một ảnh (64*64 = 4096)
    N            số ảnh huấn luyện
    X  (D x N)   mỗi CỘT là một ảnh đã duỗi thành vectơ, giá trị trong [0,1]
    mu (D x 1)   khuôn mặt trung bình
    A  (D x N)   A = X - mu           (các cột là phi_i = x_i - mu)
    L  (N x N)   L = (1/N) A^T A      (ma trận rút gọn)
    lam, V       trị riêng / vectơ riêng của L
    U  (D x r)   Eigenfaces: u_i = A v_i / ||A v_i||   (r = số trị riêng dương <= N-1)
    Uk (D x k)   k Eigenfaces ứng với k trị riêng lớn nhất;  k chọn theo eta_k >= theta (4.21)
    w = Uk^T (x - mu)   vectơ đặc trưng k chiều
PCA được TỰ CÀI ĐẶT bằng NumPy, không dùng sklearn.PCA.
OpenCV chỉ dùng để: đọc/ghi ảnh, đổi kích thước, cân bằng sáng, dò và cắt vùng mặt (Haar Cascade).
"""
import os
import re
import sys
import unicodedata

import cv2
import numpy as np

KICH_THUOC = 64          # H = W = 64  =>  D = 4096
PHAN_MO_RONG = (".png", ".jpg", ".jpeg", ".bmp", ".pgm")
CAN_BANG_SANG = True     # Bước 1: cân bằng histogram để giảm ảnh hưởng của ánh sáng
THETA_MAC_DINH = 0.95    # ngưỡng tỉ lệ phương sai tích lũy (4.21)
TEN_HAAR = "haarcascade_frontalface_default.xml"


# ----------------------------------------------------------------------------
# Tiện ích đọc/ghi ảnh an toàn với đường dẫn có dấu tiếng Việt (Windows)
# ----------------------------------------------------------------------------
def doc_anh(duong_dan, che_do=cv2.IMREAD_GRAYSCALE):
    try:
        du_lieu = np.fromfile(duong_dan, dtype=np.uint8)
        return cv2.imdecode(du_lieu, che_do) if du_lieu.size else None
    except OSError:
        return None


def ghi_anh(duong_dan, anh):
    duoi = os.path.splitext(duong_dan)[1] or ".png"
    ok, bo_dem = cv2.imencode(duoi, anh)
    if not ok:
        raise IOError(f"Không ghi được ảnh {duong_dan}")
    bo_dem.tofile(duong_dan)


def chuan_hoa_ten(s):
    """Bỏ dấu tiếng Việt, chỉ giữ chữ và số (tên thư mục / nhãn hiển thị trên ảnh)."""
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return "".join(c for c in s if c.isalnum())


# ----------------------------------------------------------------------------
# Dò mặt (chỉ để TÌM và CẮT vùng mặt, không tham gia nhận dạng)
# ----------------------------------------------------------------------------
def _tim_xml_haar():
    ung_vien = []
    try:
        ung_vien.append(os.path.join(cv2.data.haarcascades, TEN_HAAR))
    except Exception:
        pass
    if hasattr(sys, "_MEIPASS"):                                   # khi chạy từ .exe đóng gói
        ung_vien += [os.path.join(sys._MEIPASS, "cv2", "data", TEN_HAAR), os.path.join(sys._MEIPASS, TEN_HAAR)]
    from duong_dan import GOC
    ung_vien.append(os.path.join(GOC, TEN_HAAR))                   # có thể chép file xml cạnh chương trình
    for p in ung_vien:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Không tìm thấy {TEN_HAAR}. Hãy chép file này vào cạnh chương trình.")


def tao_bo_do_mat():
    """Trả về cv2.CascadeClassifier đã nạp xong (có phương án dự phòng khi đường dẫn có dấu)."""
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("OpenCV bản này không có CascadeClassifier. Cài: pip install \"opencv-python>=4.13,<5\"")
    duong_dan = _tim_xml_haar()
    bo_do = cv2.CascadeClassifier(duong_dan)
    if bo_do.empty():                                              # đường dẫn Unicode -> nạp từ bộ nhớ
        with open(duong_dan, "rb") as f:
            noi_dung = f.read().decode("utf-8")
        fs = cv2.FileStorage(noi_dung, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        bo_do = cv2.CascadeClassifier()
        bo_do.read(fs.getFirstTopLevelNode())
    if bo_do.empty():
        raise RuntimeError("Không nạp được bộ dò khuôn mặt Haar Cascade.")
    return bo_do


def tim_mat_lon_nhat(bo_do, xam, min_size=80):
    cac_mat = bo_do.detectMultiScale(xam, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    return tuple(max(cac_mat, key=lambda r: r[2] * r[3])) if len(cac_mat) > 0 else None


TI_LE_KHUNG_CO_DINH = 0.6   # cạnh khung vuông cố định = 60% cạnh ngắn của khung hình


def khung_co_dinh(rong, cao, ti_le=TI_LE_KHUNG_CO_DINH):
    """Ô vuông cố định ở giữa khung hình (x, y, w, h): người dùng tự đưa mặt vào ô này (thay cho dò mặt Haar)."""
    c = int(min(rong, cao) * ti_le)
    return ((rong - c) // 2, (cao - c) // 2, c, c)


def tim_mat(bo_do, xam, min_size=80):
    """Vùng mặt trên khung webcam: bo_do=None -> khung cố định ở giữa (không Haar); ngược lại dò Haar (có thể trả None)."""
    if bo_do is None:
        return khung_co_dinh(xam.shape[1], xam.shape[0])
    return tim_mat_lon_nhat(bo_do, xam, min_size)


def cat_mat(xam, hcn, kich_thuoc=KICH_THUOC):
    """Cắt vùng mặt hcn=(x,y,w,h) và đưa về kich_thuoc x kich_thuoc (ảnh lưu trong data/)."""
    x, y, w, h = hcn
    return cv2.resize(xam[y:y + h, x:x + w], (kich_thuoc, kich_thuoc))


TI_LE_LECH_TOI_DA = 1.25   # ảnh ngoài có cạnh dài/cạnh ngắn vượt mức này thì cắt vuông ở giữa (tránh méo mặt)
CANH_DO_MAT_TOI_THIEU = 160  # chỉ khi bật dò Haar: ảnh có cạnh ngắn <= mức này được xem là đã cắt sẵn mặt


def anh_mat_tu_anh_ngoai(xam, bo_do=None, min_size=40):
    """
    ẢNH GỬI TỪ BÊN NGOÀI (ảnh điện thoại, bộ dữ liệu...) -> ảnh khuôn mặt xám KICH_THUOC x KICH_THUOC.
    Dùng chung cho NHẬP DỮ LIỆU và NHẬN DẠNG để hai bên xử lý ảnh giống hệt nhau.
      * bo_do = None (mặc định, KHÔNG dùng Haar): xem ảnh là ảnh mặt. Nếu ảnh lệch tỉ lệ quá TI_LE_LECH_TOI_DA
        thì cắt vuông ở giữa; sau đó đưa về KICH_THUOC x KICH_THUOC.
      * bo_do = bộ dò Haar: ảnh lớn thì dò mặt lớn nhất rồi cắt; không thấy mặt -> trả về None (bỏ qua ảnh).
        Ảnh nhỏ (cạnh ngắn <= CANH_DO_MAT_TOI_THIEU) được xem là đã cắt sẵn mặt.
    Cân bằng histogram và chuẩn hoá [0,1] vẫn nằm ở tien_xu_ly (lúc huấn luyện / nhận dạng).
    """
    h, w = xam.shape[:2]
    if bo_do is not None and min(h, w) > CANH_DO_MAT_TOI_THIEU:
        mat = tim_mat_lon_nhat(bo_do, xam, min_size)
        return None if mat is None else cat_mat(xam, mat)
    if max(h, w) / min(h, w) > TI_LE_LECH_TOI_DA:
        c = min(h, w)
        return cat_mat(xam, ((w - c) // 2, (h - c) // 2, c, c))
    return cat_mat(xam, (0, 0, w, h))


def nhap_danh_sach_anh(ds_file, ten, phien, thu_muc_data, bo_do=None):
    """Nhập một danh sách file ảnh ngoài vào data/<ten>/<ten>_s<phien>_<stt>.png. Trả về (số ảnh đã lưu, số ảnh bỏ qua)."""
    thu_muc_dich = os.path.join(thu_muc_data, ten)
    os.makedirs(thu_muc_dich, exist_ok=True)
    luu = bo = 0
    for f in ds_file:
        xam = doc_anh(f)
        mat = anh_mat_tu_anh_ngoai(xam, bo_do) if xam is not None else None
        if mat is None:                                   # file hỏng, hoặc bật Haar mà không thấy mặt
            bo += 1
            continue
        stt = 0
        while os.path.exists(os.path.join(thu_muc_dich, f"{ten}_s{phien}_{stt:03d}.png")):
            stt += 1
        ghi_anh(os.path.join(thu_muc_dich, f"{ten}_s{phien}_{stt:03d}.png"), mat)
        luu += 1
    return luu, bo


# ----------------------------------------------------------------------------
# Bước 1 (Mục 5.2): TIỀN XỬ LÝ ẢNH
# ----------------------------------------------------------------------------
def tien_xu_ly(anh_xam, can_bang=None):
    """Ảnh xám (mảng 2 chiều) -> vectơ D chiều, giá trị trong [0,1]."""
    if can_bang is None:
        can_bang = CAN_BANG_SANG
    anh = cv2.resize(anh_xam, (KICH_THUOC, KICH_THUOC))          # cùng kích thước H x W
    if anh.dtype != np.uint8:
        anh = np.clip(anh, 0, 255).astype(np.uint8)
    if can_bang:
        anh = cv2.equalizeHist(anh)                                # giảm ảnh hưởng ánh sáng
    return anh.astype(np.float64).reshape(-1) / 255.0             # duỗi thành vectơ + chuẩn hoá [0,1]


MAU_PHIEN = re.compile(r"_s([A-Za-z0-9]+)_\d+\.[A-Za-z]+$")


def nap_du_lieu(thu_muc="data"):
    """
    Đọc thư mục  data/<ten_nguoi>/<ten>_s<phien>_<stt>.png
    Trả về:  X (D x N), nhan (N,) chỉ số người, ten (danh sách tên), phien (N,) mã phiên chụp
    (ảnh cũ không có mã phiên được xem như cùng một phiên "0").
    """
    if not os.path.isdir(thu_muc):
        raise RuntimeError(f"Chưa có thư mục dữ liệu '{thu_muc}'. Hãy thu thập ảnh trước.")
    cot, nhan, ten, phien = [], [], [], []
    for ten_nguoi in sorted(os.listdir(thu_muc)):
        duong_dan_nguoi = os.path.join(thu_muc, ten_nguoi)
        if not os.path.isdir(duong_dan_nguoi):
            continue
        tep = sorted(f for f in os.listdir(duong_dan_nguoi) if f.lower().endswith(PHAN_MO_RONG))
        so_anh_hop_le = 0
        for f in tep:
            anh = doc_anh(os.path.join(duong_dan_nguoi, f))
            if anh is None:
                continue
            if so_anh_hop_le == 0:
                ten.append(ten_nguoi)
            m = MAU_PHIEN.search(f)
            cot.append(tien_xu_ly(anh))
            nhan.append(len(ten) - 1)
            phien.append(m.group(1) if m else "0")
            so_anh_hop_le += 1
    if not cot:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong '{thu_muc}'. Hãy thu thập ảnh trước.")
    X = np.array(cot).T            # mỗi ảnh là một CỘT  ->  D x N
    return X, np.array(nhan), ten, np.array(phien)


# ----------------------------------------------------------------------------
# Lớp PCA / Eigenfaces
# ----------------------------------------------------------------------------
class Eigenfaces:
    def fit(self, X):
        """Giai đoạn 1 (Mục 5.4): xây dựng không gian Eigenfaces từ X (D x N)."""
        D, N = X.shape
        self.mu = X.mean(axis=1, keepdims=True)          # Bước 2: mu = (x1+...+xN)/N
        A = X - self.mu                                  #         phi_i = x_i - mu
        L = (A.T @ A) / N                                # Bước 3: L = (1/N) A^T A   (N x N, nhỏ)
        lam, V = np.linalg.eigh(L)                       # Bước 4: giải  L v = lam v  (L đối xứng)
        thu_tu = np.argsort(lam)[::-1]                   #         sắp xếp trị riêng GIẢM DẦN
        lam, V = lam[thu_tu], V[:, thu_tu]
        giu = lam > 1e-10 * lam[0]                       #         chỉ giữ trị riêng DƯƠNG (rank <= N-1)
        lam, V = lam[giu], V[:, giu]
        U = A @ V                                        #         u_i = A v_i / ||A v_i||
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        self.lam, self.U = lam, U                        # lưu đủ r thành phần, chọn k sau
        return self

    # --- chọn k theo (4.21) ------------------------------------------------------
    def ti_le_phuong_sai_tich_luy(self):
        """eta_k = (lam_1+...+lam_k) / (lam_1+...+lam_r)   (công thức 4.21)."""
        return np.cumsum(self.lam) / self.lam.sum()

    def chon_k_theo_phuong_sai(self, theta=THETA_MAC_DINH):
        """k nhỏ nhất sao cho eta_k >= theta."""
        k = int(np.searchsorted(self.ti_le_phuong_sai_tich_luy(), theta) + 1)
        return min(k, len(self.lam))

    # --- chiếu / tái tạo ------------------------------------------------------
    def chieu(self, X, k):
        """w = Uk^T (x - mu). X có thể là 1 vectơ (D,) hoặc ma trận (D x M). Trả về (k x M)."""
        if X.ndim == 1:
            X = X[:, None]
        return self.U[:, :k].T @ (X - self.mu)

    def tai_tao(self, w, k):
        """x ≈ mu + Uk w   (công thức 4.22)."""
        if w.ndim == 1:
            w = w[:, None]
        return self.mu + self.U[:, :k] @ w


# ----------------------------------------------------------------------------
# Nhận dạng bằng khoảng cách Euclid (Mục 5.3)
# ----------------------------------------------------------------------------
def ma_tran_khoang_cach(W_test, W_train):
    """Khoảng cách L2 (4.10) giữa mọi cặp cột:  ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (tiết kiệm bộ nhớ)."""
    a2 = (W_test ** 2).sum(axis=0)[:, None]
    b2 = (W_train ** 2).sum(axis=0)[None, :]
    d2 = a2 + b2 - 2.0 * (W_test.T @ W_train)
    return np.sqrt(np.maximum(d2, 0.0))


def tim_gan_nhat(W_test, W_train, khoi=2000):
    """
    W_test (k x M), W_train (k x N).
    Với mỗi ảnh test trả về: chỉ số ảnh train gần nhất và d_min = min_i ||w_test - w_i||_2
    Bộ nhớ O(khoi*N), không còn O(k*M*N).
    """
    M = W_test.shape[1]
    chi_so = np.empty(M, dtype=int)
    d_min = np.empty(M)
    for i in range(0, M, khoi):
        d = ma_tran_khoang_cach(W_test[:, i:i + khoi], W_train)
        j = d.argmin(axis=1)
        chi_so[i:i + khoi] = j
        d_min[i:i + khoi] = d[np.arange(d.shape[0]), j]
    return chi_so, d_min


def luu_mo_hinh(duong_dan, pca, k, W_train, nhan_train, ten, eps, theta=THETA_MAC_DINH):
    np.savez(duong_dan, mu=pca.mu, Uk=pca.U[:, :k], lam=pca.lam, k=k,
             W=W_train, nhan=nhan_train, ten=np.array(ten), eps=eps,
             theta=theta, can_bang=int(CAN_BANG_SANG))


class MoHinh:
    """Mô hình đã huấn luyện, dùng cho giai đoạn nhận dạng (webcam / ảnh tĩnh)."""

    def __init__(self, duong_dan="mo_hinh.npz"):
        d = np.load(duong_dan, allow_pickle=False)
        self.mu, self.Uk, self.W = d["mu"], d["Uk"], d["W"]
        self.nhan, self.ten = d["nhan"], [str(t) for t in d["ten"]]
        self.k, self.eps = int(d["k"]), float(d["eps"])
        self.theta = float(d["theta"]) if "theta" in d.files else None
        # mô hình cũ (không có trường này) được huấn luyện khi chưa cân bằng sáng
        self.can_bang = bool(int(d["can_bang"])) if "can_bang" in d.files else False

    def nhan_dang(self, anh_xam_mat):
        """Ảnh xám đã cắt vùng mặt -> (tên hoặc None nếu là người lạ, d_min)."""
        x = tien_xu_ly(anh_xam_mat, self.can_bang)[:, None]  # Giai đoạn 2, bước 2
        w = self.Uk.T @ (x - self.mu)                        # bước 3: w_test = Uk^T (x_test - mu)
        chi_so, d_min = tim_gan_nhat(w, self.W)              # bước 4-5: khoảng cách nhỏ nhất
        d_min = float(d_min[0])
        if d_min <= self.eps:                                # bước 6: so với ngưỡng eps
            return self.ten[int(self.nhan[chi_so[0]])], d_min
        return None, d_min                                   # bước 7: người lạ
