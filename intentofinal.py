import numpy as np
import cv2 as cv
import glob
import os
import random
import matplotlib.pyplot as plt


# --------------------------- Etapa 1: Calibración de cámara ---------------------------
def calibrate(showPics=True):
    # Ruta de las imágenes de calibración
    calibrationDir = r'C:\Users\javip\OneDrive\Escritorio\Nueva carpeta (2)\ProyectoVision3D\Camara\Fotos_Javier'
    imgPathList = glob.glob(os.path.join(calibrationDir, 'Foto_*.jpg'))

    if not imgPathList:
        print("No se encontraron imágenes. Verifica la ruta y los nombres de los archivos.")
        return None, None, None, None, None

    # Parámetros ajedrez
    nCols = 8  # Columnas
    nRows = 6  # Filas
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Coordenadas del mundo real
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

    # Calibración
    repError, K, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)

    print('Matriz intrínseca (K):\n', K)  # Matriz intrínseca
    print("Error de reproyección (pixeles): {:.4f}".format(repError))

    # Guardar resultados
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'calibration.npz')
    np.savez(paramPath, repError=repError, camMatrix=K,  # Guardado de las variables de la calibración
             distCoeff=distCoeff, rvecs=rvecs, tvecs=tvecs)

    R, _ = cv.Rodrigues(rvecs[0])  # Cálculo de la matriz de rotación con Rodrigues
    Rt = np.hstack((R, tvecs[0]))  # Matriz de rotación y traslación combinadas para pasar de 3x3 a 3x4
    P = K @ Rt  # Devolución de la matriz de P
    return P, K


def descomponer_proyeccion(P):
    M = P[:, :3]  # Extracción de la submatriz 3x3
    K, R = np.linalg.qr(np.linalg.inv(M))
    K = np.linalg.inv(K)  # Inversión lineal de K
    R = np.linalg.inv(R)
    t = np.linalg.inv(K) @ P[:, 3]

    if K[2, 2] < 0:
        K *= -1
        R *= -1

    return K, R, t


# --------------------------- Etapa 2: Detección de correspondencias ---------------------------
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


# --------------------------- Etapa 3: Estimación de la matriz fundamental ---------------------------
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


# --------------------------- Etapa 4: Visualización de líneas epipolares ---------------------------
def dibujar_lineas_epipolares(img1, img2, F, pts1, pts2, num_lineas=15):
    img1_color = img1.copy() if len(img1.shape) == 3 else cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2_color = img2.copy() if len(img2.shape) == 3 else cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    indices = random.sample(range(len(pts1)), min(num_lineas, len(pts1)))
    pts1_sample = pts1[indices]
    pts2_sample = pts2[indices]

    for pt1, pt2 in zip(pts1_sample, pts2_sample):
        l1 = F.T @ np.array([pt2[0], pt2[1], 1])
        a, b, c = l1
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, x1 = 0, img1.shape[1]
        y0 = int(-c / b) if b != 0 else 0
        y1 = int(-(a * x1 + c) / b) if b != 0 else 0
        cv.line(img1_color, (x0, y0), (x1, y1), color, 1)
        cv.circle(img1_color, tuple(np.int32(pt1)), 5, color, -1)

        l2 = F @ np.array([pt1[0], pt1[1], 1])
        a, b, c = l2
        x0, x1 = 0, img2.shape[1]
        y0 = int(-c / b) if b != 0 else 0
        y1 = int(-(a * x1 + c) / b) if b != 0 else 0
        cv.line(img2_color, (x0, y0), (x1, y1), color, 1)
        cv.circle(img2_color, tuple(np.int32(pt2)), 5, color, -1)

    img1_resized = cv.resize(img1_color, None, fx=0.5, fy=0.5)
    img2_resized = cv.resize(img2_color, None, fx=0.5, fy=0.5)

    cv.imshow('Epipolar Lines - Left', img1_resized)
    cv.imshow('Epipolar Lines - Right', img2_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


# --------------------------- Etapa 5: Cálculo de la matriz esencial ---------------------------
def calcular_matriz_esencial(F, K1, K2):
    E = K2.T @ F @ K1
    return E


# --------------------------- Etapa 6: Rectificación estereoscópica sin calibración ---------------------------
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


def rectificacion_basada_puntos(F, img1, img2, pts1, pts2):
    """
    Implementación del algoritmo 20.1 para rectificación estereoscópica sin calibración
    """
    # 1. Calcular epipolos
    e1 = epipolo(F.T)  # Epipolo en imagen 1
    e2 = epipolo(F)  # Epipolo en imagen 2

    # 2. Usar un punto alejado de los epipolos
    h, w = img1.shape[:2]
    p0 = np.array([w / 2, h / 2, 1])
    if np.linalg.norm(e1[:2] - p0[:2]) < min(w, h) / 4:
        p0 = np.array([w / 4, h / 4, 1])

    # 3. Construir homografía H1 para la primera imagen
    H1 = construir_H_hartley(e1, img1.shape, centro=p0[:2])

    # 4. Construir homografía H2 para la segunda imagen
    M = np.eye(3)  # Matriz de transformación

    # Convertir puntos a coordenadas homogéneas
    pts1_hom = np.hstack([pts1, np.ones((pts1.shape[0], 1))]).T
    pts2_hom = np.hstack([pts2, np.ones((pts2.shape[0], 1))]).T

    # Transformar puntos con H1
    y_tilde_L = H1 @ pts1_hom
    y_tilde_R_temp = H1 @ M @ pts2_hom

    # Normalizar coordenadas homogéneas
    y_tilde_L /= y_tilde_L[2, :]
    y_tilde_R_temp /= y_tilde_R_temp[2, :]

    Y_tilde_L = y_tilde_L
    u_tilde_R = y_tilde_R_temp[0, :]

    # Resolver el sistema lineal para encontrar la transformación A
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

    # Asegurar que las homografías mantengan la orientación
    H1 = asegurar_orientacion(H1)
    H2 = asegurar_orientacion(H2)

    # Aplicar rectificación
    img1_rect, img2_rect = aplicar_rectificacion_manual(img1, img2, H1, H2)

    return H1, H2, img1_rect, img2_rect


# --------------------------- Funciones para selección manual de puntos ---------------------------
def capturar_puntos_manuales(imagen, titulo_ventana, num_puntos):
    puntos = []
    imagen_mostrar = imagen.copy()

    def callback_raton(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            puntos.append((x, y))
            cv.circle(imagen_mostrar, (x, y), 8, (0, 255, 0), -1)
            cv.imshow(titulo_ventana, imagen_mostrar)
            print(f"Punto {len(puntos)}/{num_puntos}: ({x}, {y})")

    cv.namedWindow(titulo_ventana, cv.WINDOW_NORMAL)
    cv.resizeWindow(titulo_ventana, 800, 600)
    cv.setMouseCallback(titulo_ventana, callback_raton)

    print(f"Seleccione {num_puntos} puntos en la imagen. Cierre la ventana cuando termine.")
    while len(puntos) < num_puntos:
        cv.imshow(titulo_ventana, imagen_mostrar)
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

    cv.destroyWindow(titulo_ventana)
    return np.array(puntos)


def visualizar_rectificacion(img1, img2, img1_rect, img2_rect):
    """Visualiza las imágenes originales y rectificadas"""
    img1_color = cv.cvtColor(img1, cv.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
    img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cv.cvtColor(img1_color, cv.COLOR_BGR2RGB))
    plt.title('Imagen Izquierda Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv.cvtColor(img2_color, cv.COLOR_BGR2RGB))
    plt.title('Imagen Derecha Original')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv.cvtColor(img1_rect, cv.COLOR_BGR2RGB))
    plt.title('Imagen Izquierda Rectificada')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv.cvtColor(img2_rect, cv.COLOR_BGR2RGB))
    plt.title('Imagen Derecha Rectificada')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def dibujar_lineas_epipolares_manuales(img_izquierda, img_derecha, F, puntos_izq):
    # Preparar imágenes
    img_izq = img_izquierda.copy() if len(img_izquierda.shape) == 3 else cv.cvtColor(img_izquierda, cv.COLOR_GRAY2BGR)
    img_der = img_derecha.copy() if len(img_derecha.shape) == 3 else cv.cvtColor(img_derecha, cv.COLOR_GRAY2BGR)

    # Dibujar puntos en imagen izquierda
    for pt in puntos_izq:
        pt = tuple(map(int, pt))
        cv.circle(img_izq, pt, 8, (0, 255, 0), -1)

    # Dibujar líneas epipolares en imagen derecha
    for pt in puntos_izq:
        pt_hom = np.array([pt[0], pt[1], 1])
        l = F @ pt_hom
        a, b, c = l

        # Calcular puntos extremos de la línea
        h, w = img_der.shape[:2]
        if abs(b) > 1e-6:
            x0, y0 = 0, int(-c / b)
            x1, y1 = w, int(-(a * w + c) / b)
            cv.line(img_der, (x0, y0), (x1, y1), (0, 0, 255), 2)

    # Mostrar imágenes (redimensionadas al 50%)
    img_izq_resized = cv.resize(img_izq, None, fx=0.5, fy=0.5)
    img_der_resized = cv.resize(img_der, None, fx=0.5, fy=0.5)

    cv.imshow('Imagen Izquierda - Puntos seleccionados', img_izq_resized)
    cv.imshow('Imagen Derecha - Lineas Epipolares', img_der_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


def dibujar_lineas_epipolares_rectificadas(img1, img2, num_lineas=15):
    """Dibuja líneas epipolares horizontales en las imágenes rectificadas"""
    img1_color = img1.copy() if len(img1.shape) == 3 else cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2_color = img2.copy() if len(img2.shape) == 3 else cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    h, w = img1.shape[:2]

    # Dibujar líneas horizontales en ambas imágenes
    for i in range(1, num_lineas + 1):
        y = int(i * h / (num_lineas + 1))
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Línea en imagen izquierda
        cv.line(img1_color, (0, y), (w, y), color, 1)

        # Línea en imagen derecha
        cv.line(img2_color, (0, y), (w, y), color, 1)

    # Mostrar imágenes (redimensionadas al 50%)
    img1_resized = cv.resize(img1_color, None, fx=0.5, fy=0.5)
    img2_resized = cv.resize(img2_color, None, fx=0.5, fy=0.5)

    cv.imshow('Rectified Left - Epipolar Lines', img1_resized)
    cv.imshow('Rectified Right - Epipolar Lines', img2_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


# --------------------------- Programa principal MEJORADO ---------------------------
if __name__ == '__main__':
    # Calibración
    P, K = calibrate(showPics=True)
    if P is None:
        print("Error en calibración. Saliendo...")
        exit()

    _, R, t = descomponer_proyeccion(P)

    img_izquierda = cv.imread('foto3.jpg')
    img_derecha = cv.imread('foto4.jpg')

    if img_izquierda is None or img_derecha is None:
        print("No se pudieron cargar las imágenes de prueba.")
        exit()

    # Convertir a escala de grises para detección de características
    img_izquierda_gray = cv.cvtColor(img_izquierda, cv.COLOR_BGR2GRAY)
    img_derecha_gray = cv.cvtColor(img_derecha, cv.COLOR_BGR2GRAY)

    # Detectar correspondencias
    pts1, pts2 = detectar_correspondencias(img_izquierda_gray, img_derecha_gray)

    # Calcular matriz fundamental con RANSAC
    F, inliers1, inliers2 = ransac_fundamental(pts1, pts2)

    print("\n--- ANTES DE RECTIFICACIÓN ---")
    print("\nVerificación de la matriz fundamental F:")

    # Seleccionar 5 puntos manuales en la imagen izquierda
    print("\nSeleccione 5 puntos en la imagen IZQUIERDA:")
    puntos_izq = capturar_puntos_manuales(img_izquierda, "Seleccione 5 puntos - Imagen Izquierda", 5)

    # Dibujar puntos en izquierda y líneas en derecha
    dibujar_lineas_epipolares_manuales(img_izquierda, img_derecha, F, puntos_izq)

    # Dibujar líneas epipolares automáticas (15 puntos)
    print("\nVisualización automática con 15 puntos:")
    dibujar_lineas_epipolares(img_izquierda, img_derecha, F, inliers1, inliers2, 15)

    # Calcular matriz esencial
    E = calcular_matriz_esencial(F, K, K)
    print("\nMatriz esencial (E):\n", E)

    # Calcular homografías de rectificación MEJORADA
    print("\nCalculando rectificación sin calibración...")

    # Rectificación sin calibración (Algoritmo 20.1)
    H1, H2, img_izquierda_rect, img_derecha_rect = rectificacion_basada_puntos(
        F, img_izquierda, img_derecha, inliers1, inliers2)

    # Visualizar resultados
    visualizar_rectificacion(img_izquierda, img_derecha, img_izquierda_rect, img_derecha_rect)

    # Verificación de la rectificación
    print("\n--- DESPUÉS DE RECTIFICACIÓN ---")
    print("\nVerificación de la rectificación:")

    # Transformar puntos manuales a espacio rectificado
    puntos_izq_hom = np.hstack([puntos_izq, np.ones((len(puntos_izq), 1))]).T
    puntos_izq_rect = (H1 @ puntos_izq_hom).T[:, :2]

    # Dibujar puntos manuales en imágenes rectificadas
    img_izquierda_rect_con_puntos = img_izquierda_rect.copy()
    img_derecha_rect_con_puntos = img_derecha_rect.copy()

    for pt in puntos_izq_rect:
        pt_int = tuple(map(int, pt))
        cv.circle(img_izquierda_rect_con_puntos, pt_int, 8, (0, 255, 0), -1)

    # Mostrar imágenes con puntos (redimensionadas al 50%)
    cv.imshow('Imagen Izquierda Rectificada con Puntos',
              cv.resize(img_izquierda_rect_con_puntos, None, fx=0.5, fy=0.5))
    cv.waitKey(0)

    # Dibujar líneas epipolares horizontales en imágenes rectificadas
    print("\nLíneas epipolares horizontales en imágenes rectificadas:")
    dibujar_lineas_epipolares_rectificadas(img_izquierda_rect, img_derecha_rect, 15)

    # Verificar que los puntos están alineados con las líneas epipolares
    img_izquierda_rect_con_lineas = img_izquierda_rect.copy()
    img_derecha_rect_con_lineas = img_derecha_rect.copy()

    # Dibujar líneas horizontales
    h, w = img_izquierda_rect.shape[:2]
    for i in range(1, 16):
        y = int(i * h / 16)
        cv.line(img_izquierda_rect_con_lineas, (0, y), (w, y), (0, 0, 255), 1)
        cv.line(img_derecha_rect_con_lineas, (0, y), (w, y), (0, 0, 255), 1)

    # Dibujar puntos
    for pt in puntos_izq_rect:
        pt_int = tuple(map(int, pt))
        cv.circle(img_izquierda_rect_con_lineas, pt_int, 8, (0, 255, 0), -1)

    # Mostrar imágenes finales con puntos y líneas (redimensionadas al 50%)
    cv.imshow('Imagen Izquierda Rectificada - Puntos y Líneas',
              cv.resize(img_izquierda_rect_con_lineas, None, fx=0.4, fy=0.4))
    cv.imshow('Imagen Derecha Rectificada - Líneas',
              cv.resize(img_derecha_rect_con_lineas, None, fx=0.4, fy=0.4))
    cv.imshow('Rectified Left - Epipolar Lines', img_izquierda_rect_con_lineas)
    cv.imshow('Rectified Right - Epipolar Lines', img_derecha_rect_con_lineas)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("\nVerificación completada. Los puntos deberían estar alineados con las líneas epipolares horizontales.")