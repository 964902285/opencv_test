## 低通滤波
import sys
import cv2
import numpy as np

# ------截断频率------
radius = 50
MAX_RADIUS = 100
# ------低通滤波器的类型------
lpType = 0
MAX_LPTYPE = 2


def fft2Image(src):
    """
    快速傅里叶变换
    Args:
        src:

    Returns:

    """
    r, c = src.shape[:2]
    # 得到快速傅里叶变换的最优扩充
    rPadded = cv2.getOptimalDFTSize(r)
    cPadded = cv2.getOptimalDFTSize(c)
    # 边缘扩充，下边缘和右边缘的扩充值为0
    fft2 = np.zeros((rPadded, cPadded, 2), np.float32)
    fft2[:r, :c, 0] = src
    # 快速傅里叶变换
    cv2.dft(fft2, fft2, cv2.DFT_COMPLEX_OUTPUT)

    return fft2


def amplitudeSpectrum(fft2):
    # 求幅度
    real2 = np.power(fft2[:, :, 0], 2.0)
    Imag2 = np.power(fft2[:, :, 1], 2.0)
    amplitude = np.sqrt(real2 + Imag2)

    return amplitude


def graySpectrum(amplitude):
    # 对比度拉伸
    amplitude = np.log(amplitude + 1.0)
    # 归一化，傅里叶谱的灰底级显示
    spectrum = np.zeros(amplitude.shape, np.float32)
    cv2.normalize(amplitude, spectrum, 0, 1, cv2.NORM_MINMAX)

    return spectrum


def createLPFilter(shape, center, radius, lpType=2, n=2):
    # 滤波器的高和宽
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)
    # 构建低通滤波器
    lpFilter = np.zeros(shape, np.float32)
    if radius <= 0:
        return lpFilter
    if lpType == 0:  # 理想低通滤波器
        lpFilter = np.copy(d)
        lpFilter[lpFilter < pow(radius, 2.0)] = 1
        lpFilter[lpFilter >= pow(radius, 2.0)] = 0
    elif lpType == 1:  # 巴特沃斯低通滤波器
        lpFilter = 1.0 / (1.0 + np.power(np.sqrt(d) / radius, 2 * n))
    elif lpType == 2:  # 高斯低通滤波器
        lpFilter = np.exp(-d / (2.0 * pow(radius, 2.0)))

    return lpFilter


if __name__ == "__main__":
    # ------1 载入------
    image = cv2.imread('../imgs/src6.jpg', cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    cv2.imshow("image", image)
    # ------2 每一个元素乘以(-1)^(r+c)------
    fimage = np.zeros(image.shape, np.float32)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if (r + c) % 2:  # 取模
                fimage[r][c] = -1 * image[r][c]
            else:  # 0
                fimage[r][c] = image[r][c]

    # ------3,4 右侧下侧补0，快速傅里叶变换------
    fImagefft2 = fft2Image(fimage)
    # 傅里叶谱
    amplitude = amplitudeSpectrum(fImagefft2)
    # 傅里叶谱的灰度级显示
    spectrum = graySpectrum(amplitude)

    cv2.imshow("originalSpectrum", spectrum)
    # 找到傅里叶谱的最大值的位置
    minValue, maxValue, minLoc, maxLoc = cv2.minMaxLoc(amplitude)
    # 低通傅里叶谱的灰度级显示窗口
    cv2.namedWindow("lpFilterSpectrum", 1)


    def nothing(*args):
        pass


    # 调整低通滤波器的类型
    cv2.createTrackbar("lpType", "lpFilterSpectrum", lpType, MAX_LPTYPE, nothing)
    # 调整截断频率
    cv2.createTrackbar("radius", "lpFilterSpectrum", radius, MAX_RADIUS, nothing)
    # 低通滤波的结果
    res = np.zeros(spectrum.shape, np.float32)
    while True:
        # 得到当前截断频率，低通滤波器的类型
        radius = cv2.getTrackbarPos("radius", "lpFilterSpectrum")
        lpType = cv2.getTrackbarPos("lpType", "lpFilterSpectrum")
        # ------5 构建低通滤波器------
        lpFilter = createLPFilter(spectrum.shape, maxLoc, radius, lpType)
        # ------6 低通滤波器和快速傅里叶变换点乘------
        rows, cols = spectrum.shape[:2]
        fImagefft2_lpFilter = np.zeros(fImagefft2.shape, fImagefft2.dtype)
        for i in range(2):
            fImagefft2_lpFilter[:rows, :cols, i] = fImagefft2[:rows, :cols, i] * lpFilter
        # 低通傅里叶变换的傅里叶谱
        lp_amplitude = amplitudeSpectrum(fImagefft2_lpFilter)
        # 显示低通滤波后的傅里叶变换的灰度级
        lp_sepctrum = graySpectrum(lp_amplitude)
        cv2.imshow("lpFilterSpectrum", lp_sepctrum)
        # ------7,8 对低通傅里叶变换执行傅里叶逆变换，并取其实部------
        cv2.dft(fImagefft2_lpFilter, res, cv2.DFT_REAL_OUTPUT + cv2.DCT_INVERSE + cv2.DFT_SCALE)
        # ------9 乘以(-1)^(r+c)------
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2:  # 取模
                    res[r][c] *= -1
        # ------10 数据类型转换，截取左上角部分，其大小和输入图像的大小相同------
        for r in range(rows):
            for c in range(cols):
                if res[r][c] < 0:
                    res[r][c] = 0
                elif res[r][c] > 255:
                    res[r][c] = 255
        lpRes = res.astype(np.uint8)
        lpRes = lpRes[:image.shape[0], :image.shape[1]]
        cv2.imshow("LPFilter", lpRes)
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cv2.destroyAllWindows()
