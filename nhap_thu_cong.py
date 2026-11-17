"""
NHẬP ẢNH THỦ CÔNG – tự khoanh vùng khuôn mặt bằng chuột (KHÔNG dùng Haar Cascade)
====================================================================================
Người dùng chọn một hay nhiều file ảnh, với từng ảnh tự kéo một khung vuông quanh khuôn mặt.
Vùng được khoanh sẽ qua đúng đường xử lý như ảnh chụp webcam:
    cat_mat(xám, (x, y, w, h))  ->  ảnh xám 64x64  ->  data/<ten>/<ten>_s<phien>_<stt>.png
(cân bằng histogram và chuẩn hóa [0,1] vẫn nằm ở tien_xu_ly khi huấn luyện/nhận dạng.)

Khung luôn là HÌNH VUÔNG vì ảnh cuối cùng là 64x64; khung chữ nhật sẽ làm méo mặt khi resize.
Phím tắt:  Enter = lưu và sang ảnh tiếp   |   Esc hoặc Space = bỏ qua ảnh này   |   chuột phải = xóa khung
"""
import base64
import os
import tkinter as tk
from tkinter import ttk

import cv2

import duong_dan as dd
from pca_core import KICH_THUOC, cat_mat, doc_anh, ghi_anh

KHUNG_RONG, KHUNG_CAO = 820, 600       # kích thước vùng hiển thị ảnh (ảnh lớn hơn sẽ thu nhỏ để hiển thị)
CANH_TOI_THIEU = 24                    # cạnh khung tối thiểu (điểm ảnh gốc) để không lưu vùng quá nhỏ


def khung_vuong(x0, y0, x1, y1, w, h):
    """
    Từ điểm bấm (x0,y0) và điểm kéo tới (x1,y1) (đều theo tọa độ ẢNH GỐC) trả về khung vuông (x, y, canh)
    nằm trọn trong ảnh w x h. Góc bấm chuột là góc cố định, cạnh = max(|dx|,|dy|) rồi bị cắt cho vừa ảnh.
    """
    x0 = min(max(x0, 0), w - 1); y0 = min(max(y0, 0), h - 1)
    dx, dy = x1 - x0, y1 - y0
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    canh = max(abs(dx), abs(dy))
    canh = min(canh, (w - 1 - x0) if sx > 0 else x0, (h - 1 - y0) if sy > 0 else y0)
    canh = int(max(canh, 0))
    x = x0 if sx > 0 else x0 - canh
    y = y0 if sy > 0 else y0 - canh
    return int(x), int(y), canh


def _sang_photo(anh):
    ok, bo_dem = cv2.imencode(".png", anh, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return tk.PhotoImage(data=base64.b64encode(bo_dem.tobytes()))


class KhoanhMat(tk.Toplevel):
    """Cửa sổ khoanh mặt thủ công. Gọi khi_xong(so_anh_da_luu) lúc đóng."""

    def __init__(self, cha, ten, phien, danh_sach_file, khi_xong=None):
        super().__init__(cha)
        self.title(f"Nhập ảnh thủ công – {ten}")
        self.transient(cha)
        self.ten, self.phien, self.ds, self.khi_xong = ten, phien, list(danh_sach_file), khi_xong
        self.vi_tri = -1
        self.xam = None
        self.ti_le = 1.0
        self.khung = None                 # (x, y, canh) theo tọa độ ảnh gốc
        self.diem_bam = None
        self.da_luu = 0
        self.da_bo = 0
        self._photo = self._xem = None

        self.canvas = tk.Canvas(self, width=KHUNG_RONG, height=KHUNG_CAO, bg="#222", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, rowspan=2, padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self._bam)
        self.canvas.bind("<B1-Motion>", self._keo)
        self.canvas.bind("<ButtonRelease-1>", self._tha)
        self.canvas.bind("<ButtonPress-3>", lambda e: self._xoa_khung())

        pr = ttk.Frame(self, padding=(0, 8, 8, 8)); pr.grid(row=0, column=1, sticky="n")
        self.var_tt = tk.StringVar()
        ttk.Label(pr, textvariable=self.var_tt, wraplength=240, justify="left").pack(anchor="w")
        ttk.Label(pr, text="\nCách làm: nhấn giữ chuột tại một góc quanh mặt rồi kéo chéo. "
                           "Khung luôn là hình vuông. Chuột phải để vẽ lại.",
                  wraplength=240, foreground="#555").pack(anchor="w")
        ttk.Label(pr, text="\nẢnh sẽ được lưu (xám 64×64):").pack(anchor="w")
        self.lbl_xem = ttk.Label(pr, text="(chưa khoanh)", relief="sunken", anchor="center", width=22)
        self.lbl_xem.pack(anchor="w", pady=4)
        self.btn_luu = ttk.Button(pr, text="💾 Lưu & ảnh tiếp (Enter)", command=self._luu, state="disabled")
        self.btn_luu.pack(fill="x", pady=(10, 2))
        ttk.Button(pr, text="⏭ Bỏ qua ảnh này (Esc)", command=self._bo_qua).pack(fill="x", pady=2)
        ttk.Button(pr, text="✔ Xong / đóng", command=self._dong).pack(fill="x", pady=(14, 2))

        self.bind("<Return>", lambda e: self._luu())
        self.bind("<Escape>", lambda e: self._bo_qua())
        self.bind("<space>", lambda e: self._bo_qua())
        self.protocol("WM_DELETE_WINDOW", self._dong)
        self.after(50, self._anh_tiep)
        self.grab_set()
        self.focus_set()

    # ------------------------------------------------------------------ chuyển ảnh
    def _anh_tiep(self):
        while True:
            self.vi_tri += 1
            if self.vi_tri >= len(self.ds):
                self._dong(); return
            xam = doc_anh(self.ds[self.vi_tri])
            if xam is not None:
                break
            self.da_bo += 1                                   # file hỏng / không đọc được
        self.xam = xam
        h, w = xam.shape
        self.ti_le = min(KHUNG_RONG / w, KHUNG_CAO / h, 1.0)
        hien = xam if self.ti_le >= 1.0 else cv2.resize(
            xam, (max(1, int(w * self.ti_le)), max(1, int(h * self.ti_le))), interpolation=cv2.INTER_AREA)
        self._photo = _sang_photo(hien)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self._dw, self._dh = hien.shape[1], hien.shape[0]
        self._xoa_khung()
        self._cap_nhat_tt()

    def _cap_nhat_tt(self):
        self.var_tt.set(f"Ảnh {self.vi_tri + 1}/{len(self.ds)}: {os.path.basename(self.ds[self.vi_tri])}\n"
                        f"Đã lưu: {self.da_luu}   Bỏ qua: {self.da_bo}   (phiên '{self.phien}')")

    # ------------------------------------------------------------------ vẽ khung bằng chuột
    def _tu_man_hinh(self, ex, ey):
        ex = min(max(ex, 0), self._dw - 1); ey = min(max(ey, 0), self._dh - 1)
        return ex / self.ti_le, ey / self.ti_le

    def _bam(self, e):
        self.diem_bam = self._tu_man_hinh(e.x, e.y)

    def _keo(self, e):
        if self.diem_bam is None or self.xam is None:
            return
        h, w = self.xam.shape
        x1, y1 = self._tu_man_hinh(e.x, e.y)
        self.khung = khung_vuong(*self.diem_bam, x1, y1, w, h)
        self._ve_khung()

    def _tha(self, e):
        self.diem_bam = None
        self._cap_nhat_xem()

    def _ve_khung(self):
        self.canvas.delete("khung")
        if self.khung is None:
            return
        x, y, c = self.khung
        hop = "#00e060" if c >= CANH_TOI_THIEU else "#e03030"
        self.canvas.create_rectangle(x * self.ti_le, y * self.ti_le, (x + c) * self.ti_le, (y + c) * self.ti_le,
                                     outline=hop, width=2, tags="khung")

    def _xoa_khung(self):
        self.khung = None; self.diem_bam = None
        self.canvas.delete("khung")
        self._cap_nhat_xem()

    def _hop_le(self):
        return self.khung is not None and self.khung[2] >= CANH_TOI_THIEU

    def _cat(self):
        x, y, c = self.khung
        return cat_mat(self.xam, (x, y, c, c))               # đúng hàm dùng cho webcam

    def _cap_nhat_xem(self):
        if not self._hop_le():
            self.lbl_xem.configure(image="", text="(chưa khoanh hoặc khung quá nhỏ)")
            self.btn_luu.configure(state="disabled"); self._xem = None; return
        nho = self._cat()
        self._xem = _sang_photo(cv2.resize(nho, (KICH_THUOC * 2, KICH_THUOC * 2), interpolation=cv2.INTER_NEAREST))
        self.lbl_xem.configure(image=self._xem, text="")
        self.btn_luu.configure(state="normal")

    # ------------------------------------------------------------------ lưu / bỏ qua / đóng
    def _luu(self):
        if self.xam is None or not self._hop_le():
            return
        thu_muc = os.path.join(dd.DATA, self.ten); os.makedirs(thu_muc, exist_ok=True)
        stt = 0
        while os.path.exists(os.path.join(thu_muc, f"{self.ten}_s{self.phien}_{stt:03d}.png")):
            stt += 1
        ghi_anh(os.path.join(thu_muc, f"{self.ten}_s{self.phien}_{stt:03d}.png"), self._cat())
        self.da_luu += 1
        self._anh_tiep()

    def _bo_qua(self):
        if self.xam is None:
            return
        self.da_bo += 1
        self._anh_tiep()

    def _dong(self):
        try:
            self.grab_release()
        except tk.TclError:
            pass
        cb, n = self.khi_xong, self.da_luu
        self.destroy()
        if cb:
            cb(n)
