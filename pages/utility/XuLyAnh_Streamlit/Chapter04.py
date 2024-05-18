import numpy as np
import cv2
L = 256

def Spectrum(imgin):
    if len(imgin.shape) == 2:  # Ảnh đen trắng
        M, N = imgin.shape
    elif len(imgin.shape) == 3:  # Ảnh màu
        M, N, _ = imgin.shape
    else:
        raise ValueError("Unsupported image format")
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)

    # Bước 1 và 2:
    # Tạo ảnh mới có kích thước PxQ
    # và thêm số 0 và phần mở rộng
    fp = np.zeros((P, Q), np.float32)
    if len(imgin.shape) == 2:  # Ảnh đen trắng
        fp[:M, :N] = imgin
    elif len(imgin.shape) == 3:  # Ảnh màu
        fp[:M, :N] = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    fp = fp / (L - 1)

    # Bước 3:
    # Nhân (-1)^(x+y) để dời vào tâm ảnh
    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]

    # Bước 4:
    # Tính DFT
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Tính spectrum
    S = np.sqrt(F[:, :, 0] ** 2 + F[:, :, 1] ** 2)
    S = np.clip(S, 0, L - 1)
    S = S.astype(np.uint8)
    return S


def FrequencyFilter(imgin):
    if len(imgin.shape) == 2:  # Ảnh đen trắng
        M, N = imgin.shape
    elif len(imgin.shape) == 3:  # Ảnh màu
        M, N, _ = imgin.shape
    else:
        raise ValueError("Unsupported image format")

    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)

    # Bước 1 và 2:
    # Tạo ảnh mới có kích thước PxQ
    # và thêm số 0 vào phần mở rộng
    fp = np.zeros((P, Q), np.float32)
    if len(imgin.shape) == 2:  # Ảnh đen trắng
        fp[:M, :N] = imgin
    elif len(imgin.shape) == 3:  # Ảnh màu
        fp[:M, :N] = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    # Bước 3:
    # Nhân (-1)^(x+y) để dời vào tâm ảnh
    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]
    # Bước 4:
    # Tính DFT
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Bước 5:
    # Tạo bộ lọc H thực High Pass Butterworth
    H = np.zeros((P, Q), np.float32)
    D0 = 60
    n = 2
    for u in range(0, P):
        for v in range(0, Q):
            Duv = np.sqrt((u - P // 2) ** 2 + (v - Q // 2) ** 2)
            if Duv > 0:
                H[u, v] = 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
    # Bước 6:
    # G = F*H nhân từng cặp
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u, v, 0] = F[u, v, 0] * H[u, v]
            G[u, v, 1] = F[u, v, 1] * H[u, v]

    # Bước 7:
    # IDFT
    g = cv2.idft(G, flags=cv2.DFT_SCALE)
    # Lấy phần thực
    gp = g[:, :, 0]
    # Nhân với (-1)^(x+y)
    for x in range(0, P):
        for y in range(0, Q):
            if (x + y) % 2 == 1:
                gp[x, y] = -gp[x, y]
    # Bước 8:
    # Lấy kích thước ảnh ban đầu
    imgout = gp[0:M, 0:N]
    imgout = np.clip(imgout, 0, 255)
    imgout = imgout.astype(np.uint8)
    return imgout

import numpy as np

def CreateNotchRejectFilter(M, N):
    u1, v1 = 44, 58
    u2, v2 = 40, 119
    u3, v3 = 86, 59
    u4, v4 = 82, 119

    D0 = 10
    n = 2
    H = np.ones((M, N), np.float32)
    for u in range(0, M):
        for v in range(0, N):
            h = 1.0
            # Bộ lọc u1, v1
            Duv = np.sqrt((u - u1)**2 + (v - v1)**2)
            if Duv > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
            else:
                h = h * 0.0

            Duv = np.sqrt((u - (M - u1))**2 + (v - (N - v1))**2)
            if Duv > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
            else:
                h = h * 0.0

            # Bộ lọc u2, v2
            Duv = np.sqrt((u - u2)**2 + (v - v2)**2)
            if Duv > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
            else:
                h = h * 0.0

            Duv = np.sqrt((u - (M - u2))**2 + (v - (N - v2))**2)
            if Duv > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
            else:
                h = h * 0.0

            # Bộ lọc u3, v3
            Duv = np.sqrt((u - u3)**2 + (v - v3)**2)
            if Duv > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
            else:
                h = h * 0.0

            Duv = np.sqrt((u - (M - u3))**2 + (v - (N - v3))**2)
            if Duv > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
            else:
                h = h * 0.0

            # Bộ lọc u4, v4
            Duv = np.sqrt((u - u4)**2 + (v - v4)**2)
            if Duv > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
            else:
                h = h * 0.0

            Duv = np.sqrt((u - (M - u4))**2 + (v - (N - v4))**2)
            if Duv > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
            else:
                h = h * 0.0

            H[u, v] = h
    return H

def DrawNotchRejectFilter():
    P, Q = 256, 256  # Hoặc thay đổi kích thước theo yêu cầu của bạn
    L = 256

    H = CreateNotchRejectFilter(P, Q)
    H = H * (L - 1)
    H = H.astype(np.uint8)
    return H
def RemoveMoire(imgin):
    if len(imgin.shape) == 3:  # Ảnh màu, chuyển đổi sang ảnh đen trắng
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    else:  # Ảnh đen trắng, sử dụng trực tiếp
        imgin_gray = imgin

    M, N = imgin_gray.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)

    # Bước 1 và 2:
    # Tạo ảnh mới có kích thước PxQ
    # và thêm số 0 vào phần mở rộng
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = imgin_gray

    # Bước 3:
    # Nhân (-1)^(x+y) để dời vào tâm ảnh
    for x in range(0, P):
        for y in range(0, Q):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]

    # Bước 4:
    # Tính DFT    
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Bước 5:
    # Tạo bộ lọc NotchReject 
    H = CreateNotchRejectFilter(P, Q)
    H = cv2.resize(H, (Q, P))  # Chuyển kích thước của H về (Q, P) thay vì (N, M)
    H = np.fft.fftshift(H)  # Dời tâm về giữa

    # Bước 6:
    # Tăng kích thước của F sao cho phù hợp với H
    F_padded = np.zeros_like(H, dtype=np.complex64)
    F_padded[:P, :Q] = F[:, :, 0]  # Đảm bảo kích thước phù hợp với H
    F_padded = np.fft.fftshift(F_padded)

    # G = F*H nhân từng cặp
    G = F_padded * H

    # Bước 7:
    # IDFT
    G = G.astype(np.float32)  # Chuyển đổi kiểu dữ liệu của G sang np.float32
    g = cv2.idft(G, flags=cv2.DFT_SCALE)

    # Lấy phần thực
    gp = g[:M, :N]  # Sử dụng chỉ hai chỉ số để truy cập mảng

    # Nhân với (-1)^(x+y)
    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                gp[x, y] = -gp[x, y]

    # Bước 8:
    # Lấy kích thước ảnh ban đầu
    imgout = gp.astype(np.uint8)
    return imgout
