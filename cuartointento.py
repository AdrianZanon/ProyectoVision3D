# En este intento llego hasta la etapa 6 , pero tengo que moficar algunas cosas

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

    H = G @ Trot @ Ttrans
    return H


def rectificacion_estereoscopica_hartley(F, img1, img2, pts1, pts2, centro=None):
    e1 = epipolo(F.T)
    e2 = epipolo(F)

    H1 = construir_H_hartley(e1, img1.shape, centro)
    M = np.eye(3)

    pts1_hom = np.hstack([pts1, np.ones((pts1.shape[0], 1))]).T
    pts2_hom = np.hstack([pts2, np.ones((pts2.shape[0], 1))]).T

    y_tilde_L = H1 @ pts1_hom
    y_tilde_R_temp = H1 @ M @ pts2_hom

    y_tilde_L /= y_tilde_L[2, :]
    y_tilde_R_temp /= y_tilde_R_temp[2, :]

    Y_tilde_L = y_tilde_L
    u_tilde_R = y_tilde_R_temp[0, :]

    YTY = Y_tilde_L @ Y_tilde_L.T
    YTu = Y_tilde_L @ u_tilde_R.T
    try:
        a_bar = np.linalg.solve(YTY, YTu)
    except np.linalg.LinAlgError:
        a_bar = np.linalg.pinv(YTY) @ YTu

    a, b, c = a_bar
    A = np.array([
        [a, b, c],
        [0, 1, 0],
        [0, 0, 1]
    ])

    H2 = A @ H1 @ M
    return H1, H2


def asegurar_orientacion(H):
    if np.linalg.det(H) < 0:
        H[1, :] *= -1
    return H


def aplicar_rectificacion_manual(img1, img2, H1, H2):
    h, w = img1.shape[:2]

    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    corners1 = cv.perspectiveTransform(corners.reshape(-1, 1, 2), H1)
    corners2 = cv.perspectiveTransform(corners.reshape(-1, 1, 2), H2)

    all_corners = np.vstack([corners1, corners2])
    x_min, y_min = np.min(all_corners, axis=(0, 1))
    x_max, y_max = np.max(all_corners, axis=(0, 1))

    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    H1_adj = T @ H1
    H2_adj = T @ H2

    new_w = int(np.ceil(x_max - x_min))
    new_h = int(np.ceil(y_max - y_min))

    img1_rect = cv.warpPerspective(img1, H1_adj, (new_w, new_h))
    img2_rect = cv.warpPerspective(img2, H2_adj, (new_w, new_h))

    return img1_rect, img2_rect


# ================ Selección Manual de Puntos ===================
def seleccionar_puntos_manual(img_left, img_right, num_puntos=8):
    puntos = {'left': [], 'right': []}

    def click_event(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            img, lado = param
            cv.circle(img, (x, y), 5, (0, 255, 0), -1)
            puntos[lado].append((x, y))
            cv.imshow(lado.capitalize(), img)

    img_left_copy = img_left.copy()
    img_right_copy = img_right.copy()

    cv.namedWindow('Left', cv.WINDOW_NORMAL)
    cv.resizeWindow('Left', 800, 600)
    cv.setMouseCallback('Left', click_event, (img_left_copy, 'left'))

    cv.namedWindow('Right', cv.WINDOW_NORMAL)
    cv.resizeWindow('Right', 800, 600)
    cv.setMouseCallback('Right', click_event, (img_right_copy, 'right'))

    print(f"Click {num_puntos} puntos en cada imagen en orden correspondiente")
    while len(puntos['left']) < num_puntos or len(puntos['right']) < num_puntos:
        cv.imshow('Left', img_left_copy)
        cv.imshow('Right', img_right_copy)
        key = cv.waitKey(20)
        if key == 27:
            break

    cv.destroyAllWindows()
    return np.array(puntos['left']), np.array(puntos['right'])


# ===================== Visualización Epipolar =========================
def dibujar_lineas_epipolares(img1, img2, F, pts1, pts2, num_lineas=10, manual_points=None):
    img1_color = cv.cvtColor(img1, cv.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
    img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()

    indices = np.random.choice(len(pts1), size=min(num_lineas, len(pts1)), replace=False)

    if manual_points is not None:
        for mp1, mp2 in zip(manual_points[0], manual_points[1]):
            cv.circle(img1_color, tuple(map(int, mp1)), 8, (0, 255, 0), -1)
            cv.circle(img2_color, tuple(map(int, mp2)), 8, (0, 255, 0), -1)

    for i in indices:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        pt1 = pts1[i]
        pt2 = pts2[i]

        cv.circle(img1_color, tuple(map(int, pt1)), 5, color, -1)
        cv.circle(img2_color, tuple(map(int, pt2)), 5, color, -1)

        line2 = F @ np.array([pt1[0], pt1[1], 1])
        x0, x1 = 0, img2.shape[1]
        y0 = int(-(line2[2] + line2[0] * x0) / line2[1])
        y1 = int(-(line2[2] + line2[0] * x1) / line2[1])
        cv.line(img2_color, (x0, y0), (x1, y1), color, 1)

        line1 = F.T @ np.array([pt2[0], pt2[1], 1])
        y0 = int(-(line1[2] + line1[0] * x0) / line1[1])
        y1 = int(-(line1[2] + line1[0] * x1) / line1[1])
        cv.line(img1_color, (x0, y0), (x1, y1), color, 1)

    img1_resized = cv.resize(img1_color, (800, 600))
    img2_resized = cv.resize(img2_color, (800, 600))

    cv.imshow('Epipolar Lines - Left', img1_resized)
    cv.imshow('Epipolar Lines - Right', img2_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


# ===================== Visualización Epipolar Rectificada =========================
def dibujar_lineas_epipolares_rectificadas(img1, img2, F_rect, H1, H2, manual_points):
    pts_left = cv.perspectiveTransform(manual_points[0].reshape(-1, 1, 2).astype(np.float32), H1)
    pts_right = cv.perspectiveTransform(manual_points[1].reshape(-1, 1, 2).astype(np.float32), H2)

    for pt1, pt2 in zip(pts_left.squeeze(), pts_right.squeeze()):
        cv.circle(img1, tuple(map(int, pt1)), 8, (0, 255, 0), -1)
        cv.circle(img2, tuple(map(int, pt2)), 8, (0, 255, 0), -1)

        line = F_rect @ np.array([pt1[0], pt1[1], 1])
        x = np.array([0, img2.shape[1]])
        y = (-line[2] - line[0] * x) / line[1]
        cv.line(img2, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), (0, 0, 255), 2)

        line = F_rect.T @ np.array([pt2[0], pt2[1], 1])
        y = (-line[2] - line[0] * x) / line[1]
        cv.line(img1, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), (0, 0, 255), 2)

    img1 = cv.resize(img1, (800, 600))
    img2 = cv.resize(img2, (800, 600))
    cv.imshow('Rectified Left', img1)
    cv.imshow('Rectified Right', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()


# ============================ Main =====================================
if __name__ == '__main__':
    # Calibración
    P = calibrate(showPics=True)
    K, R, t = descomponer_proyeccion(P)
    baseline = np.linalg.norm(t)

    # Cargar imágenes
    img1 = cv.imread('foto3.jpg')
    img2 = cv.imread('foto4.jpg')
    if img1 is None or img2 is None:
        print("Error cargando imágenes")
        exit()

    # Etapa 3: Correspondencias
    pts1, pts2 = detectar_correspondencias(img1, img2)
    F, inliers1, inliers2 = ransac_fundamental(pts1, pts2)

    # Selección manual de puntos (8 puntos)
    print("\n=== Seleccione 8 puntos correspondientes en ambas imágenes ===")
    pts_manual_left, pts_manual_right = seleccionar_puntos_manual(img1, img2, num_puntos=8)

    # Dibujar líneas epipolares originales
    dibujar_lineas_epipolares(img1, img2, F,
                              np.vstack([inliers1, pts_manual_left]),
                              np.vstack([inliers2, pts_manual_right]),
                              num_lineas=10)

    # Rectificación
    centro = (img1.shape[1] // 2, img1.shape[0] // 2)
    H1, H2 = rectificacion_estereoscopica_hartley(F, img1, img2, inliers1, inliers2, centro)
    H1 = asegurar_orientacion(H1)
    H2 = asegurar_orientacion(H2)
    img1_rect, img2_rect = aplicar_rectificacion_manual(img1, img2, H1, H2)

    # Dibujar líneas epipolares rectificadas
    F_rect = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])  # F esperado para imágenes rectificadas
    dibujar_lineas_epipolares_rectificadas(img1_rect.copy(), img2_rect.copy(),
                                           F_rect, H1, H2,
                                           (pts_manual_left, pts_manual_right))