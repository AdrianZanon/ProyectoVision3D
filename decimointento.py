import numpy as np
import cv2 as cv
import glob
import os
import random
import matplotlib.pyplot as plt


# ======================== Etapa 1: Calibración ========================
def calibrate(showPics=True):
    calibrationDir = r'C:\Users\javip\OneDrive\Escritorio\Nueva carpeta (2)\ProyectoVision3D\Camara\Fotos_Javier'
    imgPathList = glob.glob(os.path.join(calibrationDir, 'Foto_*.jpg'))

    if not imgPathList:
        print("No se encontraron imágenes. Verifica la ruta y los nombres de los archivos.")
        return None, None, None, None, None

    nCols = 8  # Columnas
    nRows = 6  # Filas
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

    worldPtsList = []
    imgPtsList = []

    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        if imgBGR is None:
            print(f"No se pudo leer la imagen: {curImgPath}")
            continue

        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nCols, nRows), None)

        if cornersFound:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefined)

            if showPics:
                cv.drawChessboardCorners(imgBGR, (nCols, nRows), cornersRefined, cornersFound)
                imgBGR_resized = cv.resize(imgBGR, None, fx=0.5, fy=0.5)
                cv.imshow('Chessboard', imgBGR_resized)
                cv.waitKey(500)

    cv.destroyAllWindows()

    if not worldPtsList:
        print("No se detectaron esquinas en ninguna imagen.")
        return None, None, None, None, None

    repError, K, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)

    print('Matriz intrínseca (K):\n', K)
    print("Error de reproyección (pixeles): {:.4f}".format(repError))

    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'calibration.npz')
    np.savez(paramPath, repError=repError, camMatrix=K,
             distCoeff=distCoeff, rvecs=rvecs, tvecs=tvecs)

    R, _ = cv.Rodrigues(rvecs[0])
    Rt = np.hstack((R, tvecs[0]))
    P = K @ Rt
    return P


# ================== Etapa 2: Descomposición de P ======================
def descomponer_proyeccion(P):
    M = P[:, :3]
    K, R = np.linalg.qr(np.linalg.inv(M))
    K = np.linalg.inv(K)
    R = np.linalg.inv(R)
    t = np.linalg.inv(K) @ P[:, 3]

    if K[2, 2] < 0:
        K *= -1
        R *= -1

    return K, R, t


# ============== Etapa 3: Matriz Fundamental y RANSAC ==================
def detectar_correspondencias(img1, img2):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    return np.array(pts1), np.array(pts2)


def normalizar_puntos(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    T = np.array([
        [np.sqrt(2) / std, 0, -mean[0] * np.sqrt(2) / std],
        [0, np.sqrt(2) / std, -mean[1] * np.sqrt(2) / std],
        [0, 0, 1]
    ])
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_hom.T).T
    return pts_norm, T


def estimar_F_8_puntos(pts1, pts2):
    pts1_norm, T1 = normalizar_puntos(pts1)
    pts2_norm, T2 = normalizar_puntos(pts2)

    A = []
    for p1, p2 in zip(pts1_norm, pts2_norm):
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt

    F_final = T2.T @ F_rank2 @ T1
    return F_final / F_final[2, 2]


def calcular_error_fundamental(F, pt1, pt2):
    pt1_h = np.append(pt1, 1)
    pt2_h = np.append(pt2, 1)
    return abs(pt2_h.T @ F @ pt1_h)


def ransac_fundamental(pts1, pts2, iteraciones=2000, umbral=0.01):
    max_inliers = []
    mejor_F = None

    for _ in range(iteraciones):
        idx = random.sample(range(len(pts1)), 8)
        F = estimar_F_8_puntos(pts1[idx], pts2[idx])

        inliers = []
        for i in range(len(pts1)):
            error = calcular_error_fundamental(F, pts1[i], pts2[i])
            if error < umbral:
                inliers.append(i)

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            mejor_F = F

    pts1_inliers = pts1[max_inliers]
    pts2_inliers = pts2[max_inliers]

    return mejor_F, pts1_inliers, pts2_inliers


# ================== Etapa 4: Matriz Esencial ==========================
def calcular_matriz_esencial(F, K1, K2):
    E = K2.T @ F @ K1
    return E


# ============= Etapa 6: Rectificación Estereoscópica ==================
def epipolo(F):
    U, S, Vt = np.linalg.svd(F)
    e = Vt[-1]
    return e / e[2]


def construir_H_hartley(e, img_shape, centro=None):
    h, w = img_shape[:2]
    if centro is None:
        centro = np.array([w / 2, h / 2])

    Ttrans = np.array([
        [1, 0, -centro[0]],
        [0, 1, -centro[1]],
        [0, 0, 1]
    ])

    e_ = Ttrans @ e
    ex, ey = e_[0], e_[1]

    r = np.sqrt(ex ** 2 + ey ** 2)
    cos_alpha = ex / r
    sin_alpha = ey / r

    Trot = np.array([
        [cos_alpha, sin_alpha, 0],
        [-sin_alpha, cos_alpha, 0],
        [0, 0, 1]
    ])

    e_rot = Trot @ e_
    if e_rot[0] < 0:
        cos_alpha = -cos_alpha
        sin_alpha = -sin_alpha
        Trot = np.array([
            [cos_alpha, sin_alpha, 0],
            [-sin_alpha, cos_alpha, 0],
            [0, 0, 1]
        ])
        e_rot = Trot @ e_

    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1 / e_rot[0], 0, 1]
    ])

    H = np.linalg.inv(Ttrans) @ G @ Trot @ Ttrans
    return H


def seleccionar_puntos_manual(imgL, imgR):
    puntos = []

    def click_event(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            puntos.append([x, y])
            cv.circle(imgL, (x, y), 5, (0, 0, 255), -1)
            cv.imshow('Izquierda', imgL)

    cv.imshow('Izquierda', imgL)
    cv.setMouseCallback('Izquierda', click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return np.array(puntos, dtype=np.float32)


def rectificar_manual(imgL, imgR, F):
    print("Selecciona puntos en la imagen izquierda. Presiona cualquier tecla cuando termines.")
    ptsL = seleccionar_puntos_manual(imgL.copy(), imgR.copy())

    print("Selecciona puntos correspondientes en la imagen derecha. Presiona cualquier tecla cuando termines.")
    ptsR = seleccionar_puntos_manual(imgR.copy(), imgL.copy())

    H1 = construir_H_hartley(epipolo(F), imgL.shape)
    H2 = construir_H_hartley(epipolo(F.T), imgR.shape)

    imgL_rect = cv.warpPerspective(imgL, H1, (imgL.shape[1], imgL.shape[0]))
    imgR_rect = cv.warpPerspective(imgR, H2, (imgR.shape[1], imgR.shape[0]))

    return imgL_rect, imgR_rect


# ======================= Uso Ejemplo ===========================
def main():
    P = calibrate(showPics=True)
    if P is None:
        return

    K, R, t = descomponer_proyeccion(P)

    # Cargar imágenes de prueba (dos imágenes estéreo)
    pathL = r'C:\Users\javip\OneDrive\Escritorio\Nueva carpeta (2)\ProyectoVision3D\Camara\Fotos_Javier\Foto_1.jpg'
    pathR = r'C:\Users\javip\OneDrive\Escritorio\Nueva carpeta (2)\ProyectoVision3D\Camara\Fotos_Javier\Foto_2.jpg'
    imgL = cv.imread(pathL)
    imgR = cv.imread(pathR)

    if imgL is None or imgR is None:
        print("No se pudieron cargar las imágenes estéreo.")
        return

    pts1, pts2 = detectar_correspondencias(imgL, imgR)
    F, inliers1, inliers2 = ransac_fundamental(pts1, pts2)

    print("Matriz Fundamental calculada:\n", F)

    imgL_rect, imgR_rect = rectificar_manual(imgL, imgR, F)

    # Mostrar imágenes rectificadas
    cv.imshow('Izquierda Rectificada', imgL_rect)
    cv.imshow('Derecha Rectificada', imgR_rect)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
