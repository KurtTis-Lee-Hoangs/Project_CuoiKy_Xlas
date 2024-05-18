import numpy as np
import cv2

L = 256

def Negative(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        M, N = imgin.shape
        imgout = np.zeros((M, N), np.uint8)
        for x in range(0, M):
            for y in range(0, N):
                r = imgin[x, y]
                s = L - 1 - r
                imgout[x, y] = s
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return Negative(imgin_gray)


def Logarit(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        M, N = imgin.shape
        imgout = np.zeros((M,N), np.uint8)
        c = (L-1)/np.log(L)
        for x in range(0, M):
            for y in range(0, N):
                r = imgin[x,y]
                if r == 0:
                    r = 1
                s = c*np.log(1+r)
                imgout[x,y] = np.uint8(s)
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return Logarit(imgin_gray)
    
def Power(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        M, N = imgin.shape
        imgout = np.zeros((M,N), np.uint8)
        gamma = 5.0
        c = np.power(L-1,1-gamma)
        for x in range(0, M):
            for y in range(0, N):
                r = imgin[x,y]
                s = c*np.power(r,gamma)
                imgout[x,y] = np.uint8(s)
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return Power(imgin_gray)

def PiecewiseLinear(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        M, N = imgin.shape
        imgout = np.zeros((M,N), np.uint8)
        rmin, rmax, vi_tri_rmin, vi_tri_rmax = cv2.minMaxLoc(imgin)
        r1 = rmin
        s1 = 0
        r2 = rmax
        s2 = L-1
        for x in range(0, M):
            for y in range(0, N):
                r = imgin[x,y]
                if r < r1:
                    s = s1/r1*r
                elif r < r2:
                    s = (s2-s1)/(r2-r1)*(r-r1) + s1
                else:
                    s = (L-1-s2)/(L-1-r2)*(r-r2) + s2
                imgout[x,y] = np.uint8(s)
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return PiecewiseLinear(imgin_gray)

def Histogram(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        M, N = imgin.shape
        imgout = np.zeros((M,L), np.uint8) + 255
        h = np.zeros(L, np.int32)
        for x in range(0, M):
            for y in range(0, N):
                r = imgin[x,y]
                h[r] = h[r]+1
        p = h/(M*N)
        scale = 2000
        for r in range(0, L):
            cv2.line(imgout,(r,M-1),(r,M-1-int(scale*p[r])), (0,0,0))
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return Histogram(imgin_gray)

def HistEqual(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        M, N = imgin.shape
        imgout = np.zeros((M, N), np.uint8)
        h = np.zeros(256, np.int32)
        for x in range(0, M):
            for y in range(0, N):
                r = imgin[x, y]
                h[r] = h[r] + 1
        p = h / (M * N)

        s = np.zeros(256, np.float64)
        for k in range(0, 256):
            for j in range(0, k + 1):
                s[k] = s[k] + p[j]

        for x in range(0, M):
            for y in range(0, N):
                r = imgin[x, y]
                imgout[x, y] = np.uint8(255 * s[r])
        return imgout
    else:
        # Trường hợp ảnh màu, chuyển đổi sang ảnh đen trắng
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        if imgin_gray.dtype != np.uint8:
            imgin_gray = imgin_gray.astype(np.uint8)
        
        # Chuyển ảnh sang định dạng grayscale trước khi áp dụng equalizeHist
        imgout = cv2.equalizeHist(imgin_gray)
        return imgout


def HistEqualColor(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        B = imgin
        G = imgin
        R = imgin
        B = cv2.equalizeHist(B)
        G = cv2.equalizeHist(G)
        R = cv2.equalizeHist(R)

        imgout = np.dstack([B, G, R])
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        imgout = cv2.equalizeHist(imgin_gray)
        return cv2.cvtColor(imgout, cv2.COLOR_GRAY2BGR)

def LocalHist(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        M, N = imgin.shape
        imgout = np.zeros((M,N), np.uint8)
        m = 3
        n = 3
        w = np.zeros((m,n), np.uint8)
        a = m // 2
        b = n // 2
        for x in range(a, M-a):
            for y in range(b, N-b):
                for s in range(-a, a+1):
                    for t in range(-b, b+1):
                        w[s+a,t+b] = imgin[x+s,y+t]
                w = cv2.equalizeHist(w)
                imgout[x,y] = w[a,b]
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return LocalHist(imgin_gray)

def HistStat(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        M, N = imgin.shape
        imgout = np.zeros((M,N), np.uint8)
        m = 3
        n = 3
        w = np.zeros((m,n), np.uint8)
        a = m // 2
        b = n // 2
        mG, sigmaG = cv2.meanStdDev(imgin)
        C = 22.8
        k0 = 0.0
        k1 = 0.1
        k2 = 0.0
        k3 = 0.1
        for x in range(a, M-a):
            for y in range(b, N-b):
                for s in range(-a, a+1):
                    for t in range(-b, b+1):
                        w[s+a,t+b] = imgin[x+s,y+t]
                msxy, sigmasxy = cv2.meanStdDev(w)
                r = imgin[x,y]
                if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy <= k3*sigmaG):
                    imgout[x,y] = np.uint8(C*r)
                else:
                    imgout[x,y] = r
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return HistStat(imgin_gray)

def MyBoxFilter(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        M, N = imgin.shape
        imgout = np.zeros((M,N), np.uint8)
        m = 11
        n = 11
        w = np.ones((m,n))
        w = w/(m*n)

        a = m // 2
        b = n // 2
        for x in range(a, M-a):
            for y in range(b, M-b):
                r = 0.0
                for s in range(-a, a+1):
                    for t in range(-b, b+1):
                        r = r + w[s+a,t+b]*imgin[x+s,y+t]
                imgout[x,y] = np.uint8(r)
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return MyBoxFilter(imgin_gray)

def BoxFilter(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        m = 21
        n = 21
        w = np.ones((m,n))
        w = w/(m*n)
        imgout = cv2.filter2D(imgin,cv2.CV_8UC1,w)
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return BoxFilter(imgin_gray)

def Threshold(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        temp = cv2.blur(imgin, (15,15))
        retval, imgout = cv2.threshold(temp,64,255,cv2.THRESH_BINARY)
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return Threshold(imgin_gray)

def MedianFilter(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        M, N = imgin.shape
        imgout = np.zeros((M,N), np.uint8)
        m = 5
        n = 5
        w = np.zeros((m,n), np.uint8)
        a = m // 2
        b = n // 2
        for x in range(0, M):
            for y in range(0, N):
                for s in range(-a, a+1):
                    for t in range(-b, b+1):
                        w[s+a,t+b] = imgin[(x+s)%M,(y+t)%N]
                w_1D = np.reshape(w, (m*n,))
                w_1D = np.sort(w_1D)
                imgout[x,y] = w_1D[m*n//2]
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return MedianFilter(imgin_gray)

def Sharpen(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        # Đạo hàm cấp 2 của ảnh
        w = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        temp = cv2.filter2D(imgin,cv2.CV_32FC1,w)

        # Hàm cv2.Laplacian chỉ tính đạo hàm cấp 2
        # cho bộ lọc có số -4 chính giữa
        imgout = imgin - temp
        imgout = np.clip(imgout, 0, L-1)
        imgout = imgout.astype(np.uint8)
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return Sharpen(imgin_gray)
 
def Gradient(imgin):
    if len(imgin.shape) == 2:  # Kiểm tra xem là ảnh đen trắng
        sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

        # Đạo hàm cấp 1 theo hướng x
        mygx = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_x)
        # Đạo hàm cấp 1 theo hướng y
        mygy = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_y)

        # Lưu ý: cv2.Sobel có hướng x nằm ngang
        # ngược lại với sách Digital Image Processing
        gx = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 1, dy = 0)
        gy = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 0, dy = 1)

        imgout = abs(gx) + abs(gy)
        imgout = np.clip(imgout, 0, L-1)
        imgout = imgout.astype(np.uint8)
        return imgout
    else:
        # Trường hợp ảnh màu, có thể chuyển đổi sang ảnh đen trắng hoặc chọn kênh màu cụ thể
        imgin_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        return Gradient(imgin_gray)

















