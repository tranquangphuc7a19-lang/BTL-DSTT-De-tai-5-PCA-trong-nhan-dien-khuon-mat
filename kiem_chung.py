"""
KIỂM CHỨNG TÍNH ĐÚNG ĐẮN CỦA CHƯƠNG TRÌNH  (đúng yêu cầu "cách kiểm chứng")
Cách chạy:   python kiem_chung.py
Mỗi mục kiểm tra một khẳng định trong lý thuyết (Mục 4) bằng số liệu thật.
"""
import argparse

import numpy as np

import duong_dan as dd
from pca_core import Eigenfaces, nap_du_lieu


def chay_kiem_chung(data, log=print):
    X, nhan, ten, phien = nap_du_lieu(data)
    D, N = X.shape
    pca = Eigenfaces().fit(X)
    A = X - pca.mu
    r = len(pca.lam)
    log(f"D = {D}, N = {N}, r = {r} trị riêng dương\n")

    # 1) Eigenfaces trực chuẩn:  U^T U = I   (4.7)
    sai_so = np.abs(pca.U.T @ pca.U - np.eye(r)).max()
    log(f"[1] (4.7)  Trực chuẩn  max|UᵀU - I| = {sai_so:.2e}   (≈ 0 là đúng)")

    # 2) Định lý 4.2.3: u_i là vectơ riêng của C = (1/N) A A^T ứng với lam_i :  C u = lam u   (4.17)
    C = (A @ A.T) / N
    kiem = [np.linalg.norm(C @ pca.U[:, i] - pca.lam[i] * pca.U[:, i]) for i in range(min(5, r))]
    log(f"[2] (4.17) ||C·u - λ·u|| của 5 Eigenface đầu = {np.max(kiem):.2e}   (≈ 0 là đúng)")

    # 3) C đối xứng, nửa xác định dương (4.13); trị riêng của L trùng trị riêng lớn nhất của C
    doi_xung = np.abs(C - C.T).max()
    lam_C = np.sort(np.linalg.eigvalsh(C))[::-1]
    log(f"[3] (4.13) max|C - Cᵀ| = {doi_xung:.2e};  trị riêng nhỏ nhất của C = {lam_C[-1]:.2e}  (≥ 0 trong sai số làm tròn)")
    log(f"    (4.16) Trị riêng từ L so với từ C trực tiếp: sai lệch lớn nhất = {np.abs(lam_C[:r] - pca.lam).max():.2e}")

    # 4) Vết:  tr(C) = tổng trị riêng   (4.15)
    log(f"[4] (4.15) tr(C) = {np.trace(C):.6f},  Σλ = {pca.lam.sum():.6f}")

    # 5) Phương sai của hình chiếu lên u_j chính là lam_j   (4.19)-(4.20)
    pv = [np.mean((pca.U[:, j] @ A) ** 2) for j in range(min(5, r))]
    log("[5] (4.19) Phương sai hình chiếu (1/N)Σ(uⱼᵀφᵢ)²  so với λⱼ:")
    for j in range(min(5, r)):
        log(f"     j = {j+1}:  {pv[j]:.6f}   vs   {pca.lam[j]:.6f}")

    # 6) Sai số tái tạo trung bình = tổng các trị riêng bị bỏ
    log("[6] (4.22) Sai số tái tạo bình phương trung bình  (1/N)Σ||φᵢ - UₖUₖᵀφᵢ||²  so với  Σ_{j>k} λⱼ:")
    for k in sorted({k for k in (1, 5, 10, min(20, r - 1)) if k >= 1}):
        Uk = pca.U[:, :k]
        loi = np.mean(np.sum((A - Uk @ (Uk.T @ A)) ** 2, axis=0))
        log(f"     k = {k:3d}:  {loi:.6f}   vs   {pca.lam[k:].sum():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=dd.DATA)
    a = ap.parse_args()
    try:
        chay_kiem_chung(a.data)
    except RuntimeError as e:
        raise SystemExit(f"Lỗi: {e}")


if __name__ == "__main__":
    main()
