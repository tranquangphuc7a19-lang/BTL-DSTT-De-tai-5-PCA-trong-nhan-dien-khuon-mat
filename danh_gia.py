"""
HUẤN LUYỆN + ĐÁNH GIÁ (dùng chung cho dòng lệnh buoc2_... và giao diện gui.py)

Quy trình (trung thực, không "học thuộc" dữ liệu test):
  1. Xây Eigenfaces trên toàn bộ dữ liệu để vẽ hình và chọn k theo (4.21):  k nhỏ nhất có eta_k >= theta.
  2. Chia train/test:  ưu tiên chia theo PHIÊN CHỤP (cả phiên vào train hoặc cả phiên vào test), vì ảnh chụp
     liên tiếp trong cùng một phiên gần như trùng nhau. Người chỉ có 1 phiên thì chia ngẫu nhiên từng ảnh.
  3. Eigenfaces CHỈ xây từ tập train; nhận dạng tập test bằng khoảng cách Euclid (1-NN). Lặp lại nhiều lần,
     lấy trung bình. Độ chính xác theo k chỉ để THAM KHẢO; k dùng thật là k_theta.
  4. Chọn eps: ảnh người ĐÃ BIẾT cho d_min nhỏ, ảnh người CHƯA BIẾT cho d_min lớn (người lạ được mô phỏng bằng
     cách loại lần lượt từng người khỏi tập train). eps tối đa hoá độ chính xác cân bằng (biết/lạ).
  5. Mô hình cuối cùng huấn luyện trên TOÀN BỘ ảnh với k = k_theta và lưu lại.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pca_core import (Eigenfaces, KICH_THUOC, THETA_MAC_DINH, luu_mo_hinh, nap_du_lieu, tim_gan_nhat)


def chia_train_test(nhan, phien, ti_le_train, rng):
    """Trả về (chỉ số train, chỉ số test, số người được chia theo phiên)."""
    train, test, so_theo_phien = [], [], 0
    for p in np.unique(nhan):
        idx = np.where(nhan == p)[0]
        cac_phien = np.unique(phien[idx])
        if len(cac_phien) >= 2:                              # chia theo phiên chụp
            ps = rng.permutation(cac_phien)
            n_tr = min(max(1, int(round(ti_le_train * len(ps)))), len(ps) - 1)
            vao_train = np.isin(phien[idx], ps[:n_tr])
            train += list(idx[vao_train]); test += list(idx[~vao_train])
            so_theo_phien += 1
        else:                                                # chỉ 1 phiên: chia ngẫu nhiên từng ảnh
            idx = rng.permutation(idx)
            if len(idx) == 1:                                # 1 ảnh duy nhất: chỉ đưa vào train
                train += list(idx); continue
            n_tr = min(max(1, int(round(ti_le_train * len(idx)))), len(idx) - 1)
            train += list(idx[:n_tr]); test += list(idx[n_tr:])
    return np.array(train, dtype=int), np.array(test, dtype=int), so_theo_phien


def _anh(v):
    return v.reshape(KICH_THUOC, KICH_THUOC)


def chay_danh_gia(data, ra, duong_mo_hinh, ti_le_train=0.7, lan_lap=20, k_toi_da=60,
                  theta=THETA_MAC_DINH, log=print):
    os.makedirs(ra, exist_ok=True)
    rng = np.random.default_rng(0)                           # cố định seed => kết quả lặp lại được
    X, nhan, ten, phien = nap_du_lieu(data)
    D, N = X.shape
    n_nguoi = len(ten)
    log(f"Đã nạp {N} ảnh của {n_nguoi} người: {', '.join(ten)}   (D = {D})")
    if n_nguoi < 1:
        raise RuntimeError("Cần ít nhất 1 người và có ảnh hợp lệ để huấn luyện nhận dạng.")

    # ---------------------------------------------------------------- (A) Eigenfaces toàn bộ dữ liệu, chọn k theo (4.21)
    pca_all = Eigenfaces().fit(X)
    r = len(pca_all.lam)
    eta = pca_all.ti_le_phuong_sai_tich_luy()
    k_theta = pca_all.chon_k_theo_phuong_sai(theta)
    log(f"Số trị riêng dương r = {r}.  Giữ {theta*100:.0f}% phương sai (4.21) cần k = {k_theta}")

    fig, ax = plt.subplots(3, 6, figsize=(12, 6.5)); ax = ax.ravel()
    ax[0].imshow(_anh(pca_all.mu[:, 0]), cmap="gray"); ax[0].set_title("Mặt trung bình μ")
    ax[1].axis("off")
    for i in range(min(16, r)):
        ax[2 + i].imshow(_anh(pca_all.U[:, i]), cmap="gray")
        ax[2 + i].set_title(f"Eigenface {i+1}\nλ={pca_all.lam[i]:.3f}", fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(ra, "hinh1_mat_trung_binh_eigenfaces.png"), dpi=150); plt.close(fig)

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, r + 1), eta, "o-", ms=3)
    plt.axhline(theta, color="r", ls="--", label=f"θ = {theta*100:.0f}%")
    plt.axvline(k_theta, color="r", ls=":", label=f"k = {k_theta}")
    plt.xlabel("Số thành phần chính k"); plt.ylabel("Tỉ lệ phương sai tích lũy ηₖ")
    plt.title("Tỉ lệ phương sai tích lũy theo k"); plt.grid(alpha=.4); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(ra, "hinh2_phuong_sai_tich_luy.png"), dpi=150); plt.close()

    ks_demo = sorted({k for k in (1, 5, 10, 20, 40, k_theta, r) if 1 <= k <= r})
    x0 = X[:, [0]]
    fig, ax = plt.subplots(1, len(ks_demo) + 1, figsize=(2.2 * (len(ks_demo) + 1), 2.8))
    ax[0].imshow(_anh(x0[:, 0]), cmap="gray"); ax[0].set_title("Ảnh gốc")
    for a, k in zip(ax[1:], ks_demo):
        xr = pca_all.tai_tao(pca_all.chieu(x0, k), k)[:, 0]
        a.imshow(_anh(np.clip(xr, 0, 1)), cmap="gray")
        a.set_title(f"k = {k}\nsai số {np.linalg.norm(x0[:,0]-xr):.2f}", fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(ra, "hinh3_tai_tao_theo_k.png"), dpi=150); plt.close(fig)

    # ---------------------------------------------------------------- (B) Độ chính xác, lặp nhiều lần chia
    k_max = min(max(k_toi_da, k_theta), r)
    ds_k = np.arange(1, k_max + 1)
    chinh_xac = np.full((lan_lap, len(ds_k)), np.nan)
    acc_k_theta = []
    d_biet = []                                              # d_min của ảnh người đã biết, tại k_theta
    cm = np.zeros((n_nguoi, n_nguoi), int)                   # ma trận nhầm lẫn cộng dồn qua mọi lần chia
    so_phien_tb = 0
    for lan in range(lan_lap):
        tr, te, so_phien_tb = chia_train_test(nhan, phien, ti_le_train, rng)
        if len(te) == 0:
            raise RuntimeError("Không còn ảnh nào để kiểm tra. Mỗi người cần ít nhất 2 ảnh.")
        pca = Eigenfaces().fit(X[:, tr])
        kk = min(k_max, len(pca.lam))
        for j, k in enumerate(ds_k):
            if k > kk:
                continue
            idx, d = tim_gan_nhat(pca.chieu(X[:, te], k), pca.chieu(X[:, tr], k))
            chinh_xac[lan, j] = np.mean(nhan[tr][idx] == nhan[te])
            if k == min(k_theta, kk):
                acc_k_theta.append(chinh_xac[lan, j]); d_biet += d.tolist()
                for that, doan in zip(nhan[te], nhan[tr][idx]):
                    cm[that, doan] += 1
        if (lan + 1) % 5 == 0:
            log(f"  ... đã chạy {lan+1}/{lan_lap} lần chia")
    tb, sd = np.nanmean(chinh_xac, axis=0), np.nanstd(chinh_xac, axis=0)
    j_best = int(np.nanargmax(tb)); k_best = int(ds_k[j_best])
    acc_theta, sd_theta = float(np.mean(acc_k_theta)), float(np.std(acc_k_theta))
    che_do = (f"chia theo PHIÊN CHỤP cho {so_phien_tb}/{n_nguoi} người" if so_phien_tb else "chia NGẪU NHIÊN từng ảnh")
    log(f"Cách chia train/test: {che_do}.")
    if so_phien_tb < n_nguoi:
        log("  ⚠ Người chỉ có 1 phiên chụp bị chia ngẫu nhiên từng ảnh → ảnh gần trùng nhau có thể làm độ chính xác cao hơn thực tế. "
            "Hãy chụp mỗi người ở ≥ 2 phiên (đổi thời điểm/ánh sáng).")
    log(f"Độ chính xác tại k = {k_theta} (theo θ): {acc_theta*100:.1f}% ± {sd_theta*100:.1f}%   (trung bình {lan_lap} lần chia)")
    log(f"Tham khảo: độ chính xác cao nhất {tb[j_best]*100:.1f}% tại k = {k_best}")

    plt.figure(figsize=(8, 4.8))
    plt.plot(ds_k, tb, "o-", color="tab:red", ms=4, label="Trung bình")
    plt.fill_between(ds_k, tb - sd, np.minimum(tb + sd, 1), color="tab:red", alpha=.2, label="± độ lệch chuẩn")
    if k_theta <= k_max:
        plt.plot(k_theta, tb[k_theta - 1], "b*", ms=16, label=f"k theo θ = {k_theta}")
    plt.axvline(k_best, color="gray", ls=":", label=f"k tốt nhất (tham khảo) = {k_best}")
    plt.xlabel("Số thành phần chính k"); plt.ylabel("Độ chính xác (tập kiểm tra)")
    plt.title("Độ chính xác nhận dạng theo số thành phần chính k")
    plt.ylim(0, 1.02); plt.grid(alpha=.4); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(ra, "hinh4_do_chinh_xac_theo_k.png"), dpi=150); plt.close()

    # Ma trận nhầm lẫn: đặt kích thước tối thiểu đủ lớn để biểu đồ 2x2 không bị quá nhỏ.
    # Khi có nhiều người, tăng kích thước theo số lớp và tự giảm cỡ chữ nhãn trục.
    kich_thuoc_hinh = max(7.0, 4.5 + 0.75 * n_nguoi)
    fig, ax = plt.subplots(figsize=(kich_thuoc_hinh, kich_thuoc_hinh * 0.85), dpi=160)
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    nguong_mau = cm.max() / 2 if cm.size else 0
    co_chu_o = max(9, min(16, 18 - n_nguoi // 2))
    for i in range(n_nguoi):
        for j in range(n_nguoi):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=co_chu_o, fontweight="bold",
                color="white" if cm[i, j] > nguong_mau else "black"
            )

    ax.set_xticks(range(n_nguoi))
    ax.set_yticks(range(n_nguoi))
    ax.set_xticklabels(ten, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(ten, fontsize=11)
    ax.set_xlabel("Dự đoán", fontsize=13, labelpad=10)
    ax.set_ylabel("Thực tế", fontsize=13, labelpad=10)
    ax.set_title(
        f"Ma trận nhầm lẫn (k = {k_theta}, cộng dồn {lan_lap} lần)",
        fontsize=14, pad=14
    )
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(
        os.path.join(ra, "hinh5_ma_tran_nham_lan.png"),
        dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    # ---------------------------------------------------------------- (C) Chọn ngưỡng eps
    d_biet = np.array(d_biet)
    diem_eps = None
    if n_nguoi >= 3:
        d_la = []
        for p in range(n_nguoi):                             # loại người p khỏi train => p là "người lạ"
            con_lai = np.where(nhan != p)[0]
            tr_p, _, _ = chia_train_test(nhan[con_lai], phien[con_lai], ti_le_train, rng)
            tr_p = con_lai[tr_p]
            pca_p = Eigenfaces().fit(X[:, tr_p])
            kp = min(k_theta, len(pca_p.lam))
            _, d = tim_gan_nhat(pca_p.chieu(X[:, nhan == p], kp), pca_p.chieu(X[:, tr_p], kp))
            d_la += d.tolist()
        d_la = np.array(d_la)
        ung_vien = np.sort(np.concatenate([d_biet, d_la]))
        diem = [((d_biet <= e).mean() + (d_la > e).mean()) / 2 for e in ung_vien]   # độ chính xác cân bằng
        eps = float(ung_vien[int(np.argmax(diem))]); diem_eps = float(max(diem))
        log(f"Ngưỡng ε = {eps:.3f}  (độ chính xác cân bằng biết/lạ = {diem_eps*100:.1f}%)")
        plt.figure(figsize=(8, 4.5))
        bins = np.linspace(0, max(d_biet.max(), d_la.max()), 40)
        plt.hist(d_biet, bins, alpha=.6, label="Người đã biết", color="tab:green")
        plt.hist(d_la, bins, alpha=.6, label="Người lạ (mô phỏng)", color="tab:red")
        plt.axvline(eps, color="k", ls="--", label=f"ε = {eps:.2f}")
        plt.xlabel("Khoảng cách nhỏ nhất d_min"); plt.ylabel("Số ảnh")
        plt.title("Phân bố d_min và ngưỡng ε"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(ra, "hinh6_nguong_eps.png"), dpi=150); plt.close()
    else:
        eps = float(np.percentile(d_biet, 95))
        if n_nguoi == 1:
            log(f"Chế độ 1 người: ε = {eps:.3f} (phân vị 95% khoảng cách của người đã biết). Không đủ dữ liệu để hiệu chỉnh khả năng phát hiện người lạ; cần thử thêm người lạ thực tế.")
        else:
            log(f"Chỉ có {n_nguoi} người nên không mô phỏng được người lạ. Tạm đặt ε = {eps:.3f} (phân vị 95% của người đã biết).")

    # ---------------------------------------------------------------- (D) Mô hình cuối cùng trên toàn bộ dữ liệu
    W = pca_all.chieu(X, k_theta)
    luu_mo_hinh(duong_mo_hinh, pca_all, k_theta, W, nhan, ten, eps, theta)
    log(f"Đã lưu mô hình: {duong_mo_hinh}  (k = {k_theta}, ε = {eps:.3f}).  Hình lưu trong: {ra}")
    return dict(n_anh=N, n_nguoi=n_nguoi, r=r, theta=theta, k=k_theta, acc=acc_theta, acc_sd=sd_theta,
                k_best=k_best, acc_best=float(tb[j_best]), eps=eps, diem_eps=diem_eps, che_do=che_do)
