"""
GIAO DIỆN ĐỒ HỌA – NHẬN DẠNG KHUÔN MẶT BẰNG PCA (EIGENFACES)
Chạy:  python gui.py      (hoặc mở NhanDangPCA.exe sau khi đóng gói – xem HUONG_DAN_EXE.md)

Các tab:  1) Thu thập ảnh   2) Huấn luyện & đánh giá   3) Nhận dạng   4) Kiểm chứng lý thuyết   5) Hướng dẫn
Toàn bộ thuật toán nằm trong pca_core.py / danh_gia.py; file này chỉ là lớp giao diện.
"""
import base64
import datetime
import os
import queue
import shutil
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

import danh_gia
import duong_dan as dd
import kiem_chung
from nhap_thu_cong import KhoanhMat
from buoc3_nhan_dang import xu_ly_khung
from pca_core import (MAU_PHIEN, PHAN_MO_RONG, THETA_MAC_DINH, MoHinh, anh_mat_tu_anh_ngoai, cat_mat, chuan_hoa_ten,
                      doc_anh, ghi_anh, nhap_danh_sach_anh, tao_bo_do_mat, tim_mat)

XANH, DO, VANG = (0, 200, 0), (0, 0, 255), (0, 220, 255)


def sang_photo(anh, rong, cao):
    """Ảnh OpenCV (BGR hoặc xám) -> tk.PhotoImage vừa khung rong x cao (không cần Pillow)."""
    h, w = anh.shape[:2]
    ti_le = min(rong / w, cao / h, 1.0)
    if ti_le < 1.0:
        anh = cv2.resize(anh, (max(1, int(w * ti_le)), max(1, int(h * ti_le))), interpolation=cv2.INTER_AREA)
    ok, bo_dem = cv2.imencode(".png", anh, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return tk.PhotoImage(data=base64.b64encode(bo_dem.tobytes()))


def thong_ke_du_lieu():
    """[(tên, số ảnh, số phiên)] của thư mục data/."""
    kq = []
    if os.path.isdir(dd.DATA):
        for t in sorted(os.listdir(dd.DATA)):
            d = os.path.join(dd.DATA, t)
            if os.path.isdir(d):
                tep = [f for f in os.listdir(d) if f.lower().endswith(PHAN_MO_RONG)]
                phien = {(MAU_PHIEN.search(f).group(1) if MAU_PHIEN.search(f) else "0") for f in tep}
                kq.append((t, len(tep), len(phien)))
    return kq


def nhap_thu_muc_anh(thu_muc_anh, ten, phien, bo_do=None):
    """Nạp ảnh có sẵn (bộ dữ liệu chuẩn...) vào data/<ten>/ bằng đường xử lý ảnh ngoài chung. Trả về (lưu, bỏ qua)."""
    ds = [os.path.join(thu_muc_anh, f) for f in sorted(os.listdir(thu_muc_anh)) if f.lower().endswith(PHAN_MO_RONG)]
    return nhap_danh_sach_anh(ds, ten, phien, dd.DATA, bo_do)


class UngDung(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nhận dạng khuôn mặt bằng PCA (Eigenfaces)")
        self.geometry("1200x780")
        self.minsize(1000, 660)
        self.cam = self.bo_do = self.mo_hinh = self.mat_cuoi = self._id_vong = None
        self.che_do, self.phien, self.so_chup_phien, self.lan_tu_dong = None, None, 0, 0.0
        self.q = queue.Queue()
        self.var_do_mat = tk.BooleanVar(value=False)     # False = KHÔNG dùng Haar cho ảnh ngoài (mặc định)
        self.anh_ref = {}

        # ---- thanh trên
        thanh = ttk.Frame(self, padding=(10, 6)); thanh.pack(fill="x")
        ttk.Label(thanh, text="Camera số:").pack(side="left")
        self.var_cam = tk.IntVar(value=0)
        ttk.Spinbox(thanh, from_=0, to=5, width=4, textvariable=self.var_cam).pack(side="left", padx=(4, 16))
        self.var_status = tk.StringVar(value="Sẵn sàng.")
        ttk.Label(thanh, textvariable=self.var_status, foreground="#555").pack(side="left")

        self.nb = ttk.Notebook(self); self.nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._tao_tab_huong_dan(); self._tao_tab_thu_thap(); self._tao_tab_huan_luyen()
        self._tao_tab_nhan_dang(); self._tao_tab_kiem_chung()
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._tat_camera())

        self.bind("<space>", self._phim_cach)
        self.protocol("WM_DELETE_WINDOW", self._thoat)
        self._lam_moi_bang(); self._cap_nhat_thong_tin_mo_hinh()
        self.after(100, self._doc_hang_doi)

    # ================================================================= tiện ích chung
    def report_callback_exception(self, loai, gia_tri, tb):
        noi_dung = "".join(traceback.format_exception(loai, gia_tri, tb))
        try:
            with open(os.path.join(dd.GOC, "loi.log"), "a", encoding="utf-8") as f:
                f.write(f"--- {datetime.datetime.now()} ---\n{noi_dung}\n")
        except OSError:
            pass
        messagebox.showerror("Lỗi chương trình", f"{gia_tri}\n\n(Chi tiết đã ghi vào loi.log)")

    def _bo_do_khung(self):
        """Bộ dò cho webcam theo công tắc Haar (đọc lại MỖI KHUNG nên đổi được khi camera đang chạy). None = khung cố định."""
        if not self.var_do_mat.get():
            return None
        if self.bo_do is None:
            try:
                self.bo_do = tao_bo_do_mat()
            except Exception:
                self.var_do_mat.set(False)
                self.var_tt.set("Không tạo được bộ dò Haar → tự chuyển sang khung vuông cố định.")
                return None
        return self.bo_do

    def _lay_bo_do(self):
        if self.bo_do is None:
            try:
                self.bo_do = tao_bo_do_mat()
            except Exception as e:
                messagebox.showerror("Không tạo được bộ dò khuôn mặt", str(e))
        return self.bo_do

    def _doc_hang_doi(self):
        try:
            while True:
                loai, nd = self.q.get_nowait()
                if loai == "log":
                    self._them_log(self.txt_log, nd)
                elif loai == "xong":
                    self._xong_huan_luyen(nd)
                elif loai == "loi":
                    self.btn_train.config(state="normal"); self.var_status.set("Huấn luyện thất bại.")
                    self._them_log(self.txt_log, f"✗ LỖI: {nd}"); messagebox.showerror("Lỗi", nd)
                elif loai == "kc":
                    self._them_log(self.txt_kc, nd)
                elif loai == "kc_xong":
                    self.btn_kc.config(state="normal"); self.var_status.set("Kiểm chứng xong.")
                elif loai == "kc_loi":
                    self.btn_kc.config(state="normal"); self._them_log(self.txt_kc, f"✗ LỖI: {nd}")
        except queue.Empty:
            pass
        self.after(100, self._doc_hang_doi)

    @staticmethod
    def _them_log(txt, s):
        txt.configure(state="normal"); txt.insert("end", s + "\n"); txt.see("end"); txt.configure(state="disabled")

    @staticmethod
    def _xoa_log(txt):
        txt.configure(state="normal"); txt.delete("1.0", "end"); txt.configure(state="disabled")

    @staticmethod
    def _khung_chu(cha, **kw):
        f = ttk.Frame(cha); txt = tk.Text(f, state="disabled", wrap="word", **kw)
        sb = ttk.Scrollbar(f, command=txt.yview); txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        return f, txt

    # ================================================================= camera dùng chung
    def _mo_camera(self, che_do):
        self._tat_camera()
        idx = self.var_cam.get()
        cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
        if not cam.isOpened():
            cam.release()
            messagebox.showerror("Không mở được camera",
                                 "Hãy đóng các ứng dụng khác đang dùng camera, hoặc đổi 'Camera số' ở thanh trên.")
            return False
        self.cam, self.che_do, self.mat_cuoi = cam, che_do, None
        self._vong_lap()
        return True

    def _tat_camera(self):
        if self._id_vong is not None:
            self.after_cancel(self._id_vong); self._id_vong = None
        self.che_do = None
        if self.cam is not None:
            self.cam.release(); self.cam = None
            for lbl in (self.lbl_tt, self.lbl_nd):
                lbl.configure(image="", text="(camera đang tắt)")
            self.anh_ref.clear()

    def _vong_lap(self):
        if self.cam is None:
            return
        ok, khung = self.cam.read()
        if ok:
            if self.che_do == "thu_thap":
                khung, lbl = self._khung_thu_thap(khung), self.lbl_tt
            else:
                khung, lbl = self._khung_nhan_dang(khung), self.lbl_nd
            anh = sang_photo(khung, 720, 540)
            self.anh_ref["video"] = anh
            lbl.configure(image=anh, text="")
        self._id_vong = self.after(30, self._vong_lap)

    # ================================================================= TAB 1: thu thập
    def _tao_tab_thu_thap(self):
        tab = ttk.Frame(self.nb, padding=8); self.nb.add(tab, text="2. Thu thập ảnh")
        tab.columnconfigure(0, weight=1); tab.rowconfigure(0, weight=1)
        self.lbl_tt = ttk.Label(tab, text="(camera đang tắt)", anchor="center", relief="sunken")
        self.lbl_tt.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        pr = ttk.Frame(tab); pr.grid(row=0, column=1, sticky="ns")
        ttk.Label(pr, text="Tên người (không cần gõ dấu):").pack(anchor="w")
        self.var_ten = tk.StringVar(); ttk.Entry(pr, textvariable=self.var_ten, width=28).pack(anchor="w", pady=(0, 6))
        h = ttk.Frame(pr); h.pack(anchor="w", pady=2)
        ttk.Label(h, text="Số ảnh mỗi phiên:").pack(side="left")
        self.var_so_anh = tk.StringVar(value="20"); ttk.Spinbox(h, from_=5, to=200, width=5, textvariable=self.var_so_anh).pack(side="left", padx=6)
        self.var_auto = tk.BooleanVar(value=False)
        ttk.Checkbutton(pr, text="Tự động chụp (0,6 giây/ảnh)", variable=self.var_auto).pack(anchor="w", pady=2)
        ttk.Button(pr, text="▶ Bật camera / bắt đầu phiên mới", command=self._bat_thu_thap).pack(fill="x", pady=(8, 2))
        ttk.Button(pr, text="📷 Chụp (phím SPACE)", command=self.chup).pack(fill="x", pady=2)
        ttk.Button(pr, text="■ Tắt camera", command=self._tat_camera).pack(fill="x", pady=2)
        self.var_tt = tk.StringVar(value="Gợi ý: mỗi người chụp ≥ 2 phiên (đổi ánh sáng/thời điểm); trong phiên hãy đổi góc đầu, biểu cảm.")
        ttk.Label(pr, textvariable=self.var_tt, wraplength=300, foreground="#0a5").pack(anchor="w", pady=6)
        ttk.Separator(pr).pack(fill="x", pady=4)
        ttk.Label(pr, text="Nhập ảnh gửi từ bên ngoài (máy tự xử lý):").pack(anchor="w")
        ttk.Button(pr, text="📥 Chọn các file ảnh → tên ở trên", command=self._nhap_anh_ngoai).pack(fill="x", pady=2)
        ttk.Checkbutton(pr, text="Dùng Haar dò mặt (tắt = khung cố định / ảnh đã là mặt)", variable=self.var_do_mat).pack(anchor="w")
        ttk.Button(pr, text="✋ Tự khoanh mặt bằng chuột (khi cần chính xác)", command=self._nhap_thu_cong).pack(fill="x", pady=2)
        ttk.Separator(pr).pack(fill="x", pady=4)
        ttk.Label(pr, text="Nạp ảnh có sẵn (bộ dữ liệu chuẩn):").pack(anchor="w")
        ttk.Button(pr, text="Thư mục ảnh của 1 người → tên ở trên", command=self._nap_mot_nguoi).pack(fill="x", pady=2)
        ttk.Button(pr, text="Bộ dữ liệu (mỗi thư mục con = 1 người)", command=self._nap_bo_du_lieu).pack(fill="x", pady=2)
        ttk.Separator(pr).pack(fill="x", pady=4)
        ttk.Label(pr, text="Dữ liệu hiện có:").pack(anchor="w")
        self.bang = ttk.Treeview(pr, columns=("ten", "anh", "phien"), show="headings", height=8)
        for c, t, w in (("ten", "Tên", 120), ("anh", "Số ảnh", 60), ("phien", "Số phiên", 70)):
            self.bang.heading(c, text=t); self.bang.column(c, width=w, anchor="center")
        self.bang.pack(fill="x")
        ttk.Button(pr, text="Xóa người đã chọn", command=self._xoa_nguoi).pack(fill="x", pady=(4, 0))

    def _bat_thu_thap(self):
        if not chuan_hoa_ten(self.var_ten.get()):
            messagebox.showwarning("Thiếu tên", "Hãy nhập tên người cần thu thập trước."); return
        self.phien = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")     # đặt TRƯỚC khi bật camera
        self.so_chup_phien = 0
        if self._mo_camera("thu_thap"):
            self.var_tt.set(f"Phiên mới {self.phien}. Nhấn SPACE hoặc nút Chụp.")

    def _khung_thu_thap(self, khung):
        xam = cv2.cvtColor(khung, cv2.COLOR_BGR2GRAY)
        bo_do = self._bo_do_khung()
        mat = tim_mat(bo_do, xam)
        self.mat_cuoi = (xam, mat) if mat is not None else None
        if mat is not None:
            x, y, w, h = mat
            cv2.rectangle(khung, (x, y), (x + w, y + h), XANH if bo_do is not None else VANG, 2)
            if bo_do is None:
                cv2.putText(khung, "Dua mat vao khung", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, VANG, 2)
        else:
            cv2.putText(khung, "Khong thay khuon mat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DO, 2)
        if self.var_auto.get() and mat is not None and time.time() - self.lan_tu_dong >= 0.6:
            self.lan_tu_dong = time.time(); self.chup()
        return khung

    def _phim_cach(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox, ttk.Combobox)):
            return
        if self.che_do == "thu_thap":
            self.chup()

    def chup(self):
        if self.che_do != "thu_thap" or self.mat_cuoi is None:
            return
        ten = chuan_hoa_ten(self.var_ten.get())
        if not ten:
            self.var_auto.set(False); messagebox.showwarning("Thiếu tên", "Hãy nhập tên người."); return
        try:
            muc_tieu = int(self.var_so_anh.get())
        except ValueError:
            muc_tieu = 20
        if self.so_chup_phien >= muc_tieu:
            self.var_auto.set(False)
            self.var_tt.set("Đủ ảnh cho phiên này. Đổi ánh sáng/thời điểm rồi bấm 'Bật camera / bắt đầu phiên mới'.")
            return
        xam, mat = self.mat_cuoi
        thu_muc = os.path.join(dd.DATA, ten); os.makedirs(thu_muc, exist_ok=True)
        stt = 0
        while os.path.exists(os.path.join(thu_muc, f"{ten}_s{self.phien}_{stt:03d}.png")):
            stt += 1
        ghi_anh(os.path.join(thu_muc, f"{ten}_s{self.phien}_{stt:03d}.png"), cat_mat(xam, mat))
        self.so_chup_phien += 1
        self.var_tt.set(f"{ten}: đã chụp {self.so_chup_phien}/{muc_tieu} ảnh trong phiên này.")
        self._lam_moi_bang()

    def _lam_moi_bang(self):
        for i in self.bang.get_children():
            self.bang.delete(i)
        for t, n, p in thong_ke_du_lieu():
            self.bang.insert("", "end", values=(t, n, p))

    def _xoa_nguoi(self):
        chon = self.bang.selection()
        if not chon:
            return
        ten = self.bang.item(chon[0])["values"][0]
        if messagebox.askyesno("Xác nhận", f"Xóa toàn bộ ảnh của '{ten}'? Không thể hoàn tác."):
            shutil.rmtree(os.path.join(dd.DATA, str(ten)), ignore_errors=True); self._lam_moi_bang()

    def _nhap_thu_cong(self):
        ten = chuan_hoa_ten(self.var_ten.get())
        if not ten:
            messagebox.showwarning("Thiếu tên", "Hãy nhập tên người ở trên trước."); return
        files = filedialog.askopenfilenames(title=f"Chọn các ảnh của '{ten}'",
                                            filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp *.pgm")])
        if not files:
            return
        self._tat_camera()
        phien = "tay" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")   # mỗi lần nhập = 1 phiên riêng

        def xong(n):
            self.var_tt.set(f"Nhập thủ công: đã lưu {n}/{len(files)} ảnh cho '{ten}' (phiên {phien}).")
            self._lam_moi_bang()
        KhoanhMat(self, ten, phien, files, xong)

    def _bo_do_neu_can(self):
        """None nếu không dùng Haar; ngược lại trả về bộ dò (hoặc False nếu không tạo được)."""
        if not self.var_do_mat.get():
            return None
        return self._lay_bo_do() or False

    def _nhap_anh_ngoai(self):
        ten = chuan_hoa_ten(self.var_ten.get())
        if not ten:
            messagebox.showwarning("Thiếu tên", "Hãy nhập tên người ở trên trước."); return
        files = filedialog.askopenfilenames(title=f"Chọn các ảnh của '{ten}'",
                                            filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp *.pgm")])
        if not files:
            return
        bo_do = self._bo_do_neu_can()
        if bo_do is False:
            return
        self._tat_camera()
        phien = "ngoai" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")      # mỗi lần nhập = 1 phiên riêng
        luu, bo = nhap_danh_sach_anh(files, ten, phien, dd.DATA, bo_do)
        self.var_tt.set(f"Đã nhập {luu}/{len(files)} ảnh cho '{ten}' (phiên {phien})." + (f" Bỏ qua {bo} ảnh (hỏng hoặc không thấy mặt)." if bo else ""))
        self._lam_moi_bang()

    def _nap_mot_nguoi(self):
        ten = chuan_hoa_ten(self.var_ten.get())
        if not ten:
            messagebox.showwarning("Thiếu tên", "Hãy nhập tên người ở trên trước."); return
        thu_muc = filedialog.askdirectory(title="Chọn thư mục chứa ảnh của người này")
        bo_do = self._bo_do_neu_can() if thu_muc else None
        if thu_muc and bo_do is not False:
            n, bo = nhap_thu_muc_anh(thu_muc, ten, "nhap" + datetime.datetime.now().strftime("%H%M%S"), bo_do)
            self.var_tt.set(f"Đã nạp {n} ảnh cho '{ten}'." + (f" Bỏ qua {bo} ảnh." if bo else "")); self._lam_moi_bang()

    def _nap_bo_du_lieu(self):
        goc = filedialog.askdirectory(title="Chọn thư mục gốc (mỗi thư mục con là ảnh của 1 người)")
        bo_do = self._bo_do_neu_can() if goc else None
        if not goc or bo_do is False:
            return
        tong = nguoi = bo_tong = 0
        phien = "nhap" + datetime.datetime.now().strftime("%H%M%S")
        for con in sorted(os.listdir(goc)):
            d = os.path.join(goc, con)
            ten = chuan_hoa_ten(con)
            if os.path.isdir(d) and ten:
                n, bo = nhap_thu_muc_anh(d, ten, phien, bo_do)
                tong += n; bo_tong += bo; nguoi += 1 if n else 0
                self.var_status.set(f"Đang nạp... {con}"); self.update_idletasks()
        self.var_tt.set(f"Đã nạp {tong} ảnh của {nguoi} người." + (f" Bỏ qua {bo_tong} ảnh." if bo_tong else "")); self.var_status.set("Sẵn sàng."); self._lam_moi_bang()

    # ================================================================= TAB 2: huấn luyện
    def _tao_tab_huan_luyen(self):
        tab = ttk.Frame(self.nb, padding=8); self.nb.add(tab, text="3. Huấn luyện & đánh giá")
        tab.columnconfigure(0, weight=1); tab.rowconfigure(2, weight=1)
        p = ttk.LabelFrame(tab, text="Tham số", padding=8); p.grid(row=0, column=0, sticky="ew")
        self.var_theta = tk.StringVar(value=str(THETA_MAC_DINH)); self.var_tl = tk.StringVar(value="0.7")
        self.var_ll = tk.StringVar(value="20"); self.var_km = tk.StringVar(value="60")
        for i, (nhan, v, tip) in enumerate((("θ (phương sai giữ lại, 4.21)", self.var_theta, "0.90–0.99"),
                                            ("Tỉ lệ train", self.var_tl, "0.5–0.8"),
                                            ("Số lần lặp chia", self.var_ll, "≥ 5"),
                                            ("k tối đa trên đồ thị", self.var_km, ""))):
            ttk.Label(p, text=nhan).grid(row=0, column=2 * i, padx=(0, 4), sticky="e")
            ttk.Entry(p, textvariable=v, width=6).grid(row=0, column=2 * i + 1, padx=(0, 14))
        self.btn_train = ttk.Button(p, text="▶ Huấn luyện + đánh giá", command=self._chay_huan_luyen)
        self.btn_train.grid(row=0, column=8, padx=8)
        self.var_tom_tat = tk.StringVar(value="Chưa có kết quả.")
        ttk.Label(tab, textvariable=self.var_tom_tat, foreground="#0a5", wraplength=1100).grid(row=1, column=0, sticky="w", pady=6)
        pw = ttk.PanedWindow(tab, orient="vertical"); pw.grid(row=2, column=0, sticky="nsew")
        kh, self.txt_log = self._khung_chu(pw, height=8, font=("Consolas", 10)); pw.add(kh, weight=1)
        kq = ttk.Frame(pw); pw.add(kq, weight=3)
        top = ttk.Frame(kq); top.pack(fill="x")
        ttk.Label(top, text="Xem hình kết quả:").pack(side="left")
        self.var_hinh = tk.StringVar()
        self.cbo_hinh = ttk.Combobox(top, textvariable=self.var_hinh, state="readonly", width=45)
        self.cbo_hinh.pack(side="left", padx=6); self.cbo_hinh.bind("<<ComboboxSelected>>", self._hien_hinh)
        ttk.Button(top, text="Mở thư mục kết quả", command=self._mo_thu_muc_ket_qua).pack(side="left")
        self.lbl_hinh = ttk.Label(kq, anchor="center", relief="sunken"); self.lbl_hinh.pack(fill="both", expand=True, pady=4)
        self._nap_ds_hinh()

    def _chay_huan_luyen(self):
        try:
            theta, tl, ll, km = (float(self.var_theta.get()), float(self.var_tl.get()),
                                 int(self.var_ll.get()), int(self.var_km.get()))
            assert 0.5 <= theta < 1 and 0.3 <= tl <= 0.9 and ll >= 1 and km >= 1
        except (ValueError, AssertionError):
            messagebox.showerror("Tham số không hợp lệ", "θ trong [0.5, 1), tỉ lệ train trong [0.3, 0.9], số lần lặp ≥ 1, k tối đa ≥ 1."); return
        self._tat_camera(); self._xoa_log(self.txt_log); self.btn_train.config(state="disabled")
        self.var_status.set("Đang huấn luyện... (vài chục giây)")

        def lam():
            try:
                kq = danh_gia.chay_danh_gia(dd.DATA, dd.KET_QUA, dd.MO_HINH, tl, ll, km, theta,
                                            log=lambda s: self.q.put(("log", s)))
                self.q.put(("xong", kq))
            except Exception as e:
                self.q.put(("loi", str(e)))
        threading.Thread(target=lam, daemon=True).start()

    def _xong_huan_luyen(self, kq):
        self.btn_train.config(state="normal"); self.var_status.set("Huấn luyện xong, đã lưu mô hình.")
        self.var_tom_tat.set(f"{kq['n_anh']} ảnh / {kq['n_nguoi']} người | k = {kq['k']} (θ = {kq['theta']*100:.0f}%) | "
                             f"độ chính xác {kq['acc']*100:.1f}% ± {kq['acc_sd']*100:.1f}% ({kq['che_do']}) | ε = {kq['eps']:.3f}")
        self.mo_hinh = None; self._cap_nhat_thong_tin_mo_hinh(); self._nap_ds_hinh()

    def _nap_ds_hinh(self):
        ds = sorted(f for f in os.listdir(dd.KET_QUA) if f.lower().endswith(".png")) if os.path.isdir(dd.KET_QUA) else []
        self.cbo_hinh["values"] = ds
        if ds:
            self.var_hinh.set(ds[min(3, len(ds) - 1)]); self.after(200, self._hien_hinh)

    def _hien_hinh(self, _=None):
        anh = doc_anh(os.path.join(dd.KET_QUA, self.var_hinh.get()), cv2.IMREAD_COLOR)
        if anh is not None:
            self.update_idletasks()
            p = sang_photo(anh, max(300, self.lbl_hinh.winfo_width() - 8), max(200, self.lbl_hinh.winfo_height() - 8))
            self.anh_ref["hinh"] = p; self.lbl_hinh.configure(image=p)

    def _mo_thu_muc_ket_qua(self):
        os.makedirs(dd.KET_QUA, exist_ok=True)
        try:
            os.startfile(dd.KET_QUA)                                  # Windows
        except AttributeError:
            messagebox.showinfo("Thư mục kết quả", dd.KET_QUA)

    # ================================================================= TAB 3: nhận dạng
    def _tao_tab_nhan_dang(self):
        tab = ttk.Frame(self.nb, padding=8); self.nb.add(tab, text="4. Nhận dạng")
        tab.columnconfigure(0, weight=1); tab.rowconfigure(0, weight=1)
        self.lbl_nd = ttk.Label(tab, text="(camera đang tắt)", anchor="center", relief="sunken")
        self.lbl_nd.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        pr = ttk.Frame(tab); pr.grid(row=0, column=1, sticky="ns")
        ttk.Button(pr, text="▶ Nhận dạng bằng webcam", command=self._bat_nhan_dang).pack(fill="x", pady=2)
        ttk.Button(pr, text="■ Tắt camera", command=self._tat_camera).pack(fill="x", pady=2)
        ttk.Button(pr, text="🖼 Nhận dạng từ file ảnh…", command=self._nhan_dang_anh).pack(fill="x", pady=(2, 0))
        ttk.Checkbutton(pr, text="Dùng Haar dò mặt (tắt = khung cố định / ảnh đã là mặt)", variable=self.var_do_mat).pack(anchor="w", pady=(0, 10))
        self.var_kq = tk.StringVar(value="—")
        self.lbl_kq = tk.Label(pr, textvariable=self.var_kq, font=("Segoe UI", 18, "bold"), wraplength=300, justify="left")
        self.lbl_kq.pack(anchor="w", pady=4)
        self.var_d = tk.StringVar(value="")
        ttk.Label(pr, textvariable=self.var_d, wraplength=300).pack(anchor="w")
        ttk.Separator(pr).pack(fill="x", pady=10)
        self.var_info = tk.StringVar()
        ttk.Label(pr, textvariable=self.var_info, wraplength=300, foreground="#555").pack(anchor="w")
        ttk.Button(pr, text="Tải lại mô hình", command=lambda: self._tai_mo_hinh(True)).pack(fill="x", pady=6)

    def _tai_mo_hinh(self, ep_buoc=False):
        if not os.path.exists(dd.MO_HINH):
            messagebox.showwarning("Chưa có mô hình", "Hãy sang tab '3. Huấn luyện & đánh giá' và chạy huấn luyện trước."); return False
        if self.mo_hinh is None or ep_buoc:
            try:
                self.mo_hinh = MoHinh(dd.MO_HINH)
            except Exception as e:
                messagebox.showerror("Không nạp được mô hình", str(e)); return False
            self._cap_nhat_thong_tin_mo_hinh()
        return True

    def _cap_nhat_thong_tin_mo_hinh(self):
        if os.path.exists(dd.MO_HINH):
            try:
                m = MoHinh(dd.MO_HINH)
                self.var_info.set(f"Mô hình: k = {m.k}, ε = {m.eps:.3f}, {len(m.ten)} người ({', '.join(m.ten)})")
                return
            except Exception:
                pass
        self.var_info.set("Chưa có mô hình.")

    def _bat_nhan_dang(self):
        if self._tai_mo_hinh(True):
            self._mo_camera("nhan_dang")

    def _khung_nhan_dang(self, khung):
        self._hien_ket_qua(xu_ly_khung(khung, self._bo_do_khung(), self.mo_hinh))
        return khung

    def _hien_ket_qua(self, kq):
        if kq is None:
            self.var_kq.set("Không thấy khuôn mặt"); self.var_d.set(""); self.lbl_kq.configure(fg="#888"); return
        ten, d = kq
        if ten is not None:
            self.var_kq.set(ten); self.lbl_kq.configure(fg="#0a8a0a")
        else:
            self.var_kq.set("NGƯỜI LẠ"); self.lbl_kq.configure(fg="#d01010")
        self.var_d.set(f"d_min = {d:.3f}   (ngưỡng ε = {self.mo_hinh.eps:.3f})")

    def _nhan_dang_anh(self):
        if not self._tai_mo_hinh(True):
            return
        bo_do = self._bo_do_neu_can()
        if bo_do is False:
            return
        f = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp *.pgm")])
        if not f:
            return
        self._tat_camera()
        khung = doc_anh(f, cv2.IMREAD_COLOR)
        if khung is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh."); return
        if bo_do is not None:
            self._hien_ket_qua(xu_ly_khung(khung, bo_do, self.mo_hinh, min_size=40))   # dò Haar, vẽ khung lên ảnh
        else:                                                    # mặc định: ảnh ngoài xử lý giống lúc nhập dữ liệu
            mat = anh_mat_tu_anh_ngoai(cv2.cvtColor(khung, cv2.COLOR_BGR2GRAY))
            self._hien_ket_qua(self.mo_hinh.nhan_dang(mat))
        p = sang_photo(khung, 720, 540); self.anh_ref["anh"] = p; self.lbl_nd.configure(image=p, text="")

    # ================================================================= TAB 4: kiểm chứng
    def _tao_tab_kiem_chung(self):
        tab = ttk.Frame(self.nb, padding=8); self.nb.add(tab, text="5. Kiểm chứng lý thuyết")
        self.btn_kc = ttk.Button(tab, text="▶ Chạy 6 phép kiểm chứng (Mục 4)", command=self._chay_kiem_chung)
        self.btn_kc.pack(anchor="w")
        kh, self.txt_kc = self._khung_chu(tab, font=("Consolas", 10)); kh.pack(fill="both", expand=True, pady=6)

    def _chay_kiem_chung(self):
        self._xoa_log(self.txt_kc); self.btn_kc.config(state="disabled"); self.var_status.set("Đang kiểm chứng...")

        def lam():
            try:
                kiem_chung.chay_kiem_chung(dd.DATA, log=lambda s: self.q.put(("kc", s)))
                self.q.put(("kc_xong", None))
            except Exception as e:
                self.q.put(("kc_loi", str(e)))
        threading.Thread(target=lam, daemon=True).start()

    # ================================================================= TAB 5: hướng dẫn
    def _tao_tab_huong_dan(self):
        tab = ttk.Frame(self.nb, padding=12); self.nb.add(tab, text="1. Hướng dẫn")
        kh, txt = self._khung_chu(tab, font=("Segoe UI", 11)); kh.pack(fill="both", expand=True)
        self._them_log(txt, (
            "QUY TRÌNH SỬ DỤNG\n"
            "1) Tab 2: nhập tên → 'Bật camera' → chụp ~20 ảnh (đổi góc đầu, biểu cảm). Lặp lại ở một phiên khác\n"
            "   (đổi ánh sáng/thời điểm) để có ≥ 2 phiên mỗi người. Chỉ cần 1 người vẫn huấn luyện và nhận dạng được; từ 2–3 người trở lên sẽ đánh giá và hiệu chỉnh phát hiện người lạ tốt hơn.\n"
            "   Có thể nạp ảnh từ bộ dữ liệu chuẩn bằng các nút 'Nạp ảnh có sẵn'.\n"
            "2) Tab 3: bấm 'Huấn luyện + đánh giá' → tạo mô hình, các hình cho báo cáo và ngưỡng ε.\n"
            "3) Tab 4: nhận dạng bằng webcam hoặc file ảnh. Tên xanh = người đã biết; đỏ = người lạ (d_min > ε).\n"
            "4) Tab 5: kiểm chứng bằng số các định lý ở Mục 4 của báo cáo.\n\n"
            "DỮ LIỆU ĐƯỢC LƯU Ở ĐÂU: thư mục data/, mo_hinh.npz, ket_qua/ nằm cạnh chương trình.\n"
            "Nếu chương trình báo lỗi, xem file loi.log cạnh chương trình."))

    # ================================================================= thoát
    def _thoat(self):
        self._tat_camera(); self.destroy()


def main():
    try:                                   # chữ nét hơn trên màn hình độ phân giải cao (Windows)
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    os.makedirs(dd.DATA, exist_ok=True)
    UngDung().mainloop()


if __name__ == "__main__":
    main()
