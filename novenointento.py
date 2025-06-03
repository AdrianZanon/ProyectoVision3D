import numpy as np
import cv2 as cv
import glob
import os
import random


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
    return P


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


# --------------------------- Etapa 6: Rectificación estereoscópica ---------------------------
def epipolo(F):
    """Calcula el epipolo derecho (o izquierdo si se pasa F.T) a partir de la matriz fundamental F.
    El epipolo es el vector en el núcleo de F (F*e=0), obtenido por SVD."""
    U, S, Vt = np.linalg.svd(F)
    e = Vt[-1]
    return e / e[2]  # Normaliza para tener coordenada homogénea 1


def construir_H_hartley(e, img_shape, centro=None):
    """Construye la homografía H que transforma la imagen de modo que el epipolo e quede en el infinito,
    siguiendo el método de Hartley. Esto es parte de la rectificación estéreo no calibrada."""
    h, w = img_shape[:2]
    if centro is None:
        centro = np.array([w / 2, h / 2])  # Centro de la imagen

    # Traslación para centrar la imagen en el origen
    Ttrans = np.array([
        [1, 0, -centro[0]],
        [0, 1, -centro[1]],
        [0, 0, 1]
    ])

    e_ = Ttrans @ e
    ex, ey = e_[0], e_[1]

    # Rotación para alinear epipolo con el eje x
    r = np.sqrt(ex ** 2 + ey ** 2)
    cos_alpha = ex / r
    sin_alpha = ey / r

    Trot = np.array([
        [cos_alpha, sin_alpha, 0],
        [-sin_alpha, cos_alpha, 0],
        [0, 0, 1]
    ])

    e_rot = Trot @ e_
    # Asegurar que el epipolo quede en el lado positivo del eje x
    if e_rot[0] < 0:
        cos_alpha = -cos_alpha
        sin_alpha = -sin_alpha
        Trot = np.array([
            [cos_alpha, sin_alpha, 0],
            [-sin_alpha, cos_alpha, 0],
            [0, 0, 1]
        ])
        e_rot = Trot @ e_

    # Homografía que envía el epipolo al infinito
    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1 / e_rot[0], 0, 1]
    ])

    H = G @ Trot @ Ttrans
    return H


def rectificacion_estereoscopica_hartley(F, img1, img2, pts1, pts2, centro=None):
    """Calcula las homografías de rectificación estéreo no calibrada usando el método de Hartley.
    Devuelve las homografías H1 y H2 para las dos imágenes."""
    e1 = epipolo(F.T)  # Epipolo izquierdo
    e2 = epipolo(F)  # Epipolo derecho

    # Construye H1 con el epipolo izquierdo
    H1 = construir_H_hartley(e1, img1.shape, centro)

    # Matriz M (similaridad o identidad, según Hartley)
    M = np.eye(3)

    # Homogeneizar puntos
    pts1_hom = np.hstack([pts1, np.ones((pts1.shape[0], 1))]).T
    pts2_hom = np.hstack([pts2, np.ones((pts2.shape[0], 1))]).T

    # Transformación provisional de puntos con H1 y M
    y_tilde_L = H1 @ pts1_hom
    y_tilde_R_temp = H1 @ M @ pts2_hom

    y_tilde_L /= y_tilde_L[2, :]
    y_tilde_R_temp /= y_tilde_R_temp[2, :]

    Y_tilde_L = y_tilde_L
    u_tilde_R = y_tilde_R_temp[0, :]

    # Resuelve la ecuación normal para encontrar coeficientes a_bar
    YTY = Y_tilde_L @ Y_tilde_L.T
    YTu = Y_tilde_L @ u_tilde_R.T
    try:
        a_bar = np.linalg.solve(YTY, YTu)
    except np.linalg.LinAlgError:
        print("Matriz singular en la ecuación normal, uso pseudoinversa")
        a_bar = np.linalg.pinv(YTY) @ YTu

    a, b, c = a_bar

    # Construye matriz A que ajusta la segunda homografía
    A = np.array([
        [a, b, c],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Homografía para la segunda imagen
    H2 = A @ H1 @ M

    return H1, H2


def aplicar_rectificacion_manual(img1, img2, H1, H2):
    """Aplica las homografías de rectificación H1 y H2 a las imágenes img1 e img2,
    realizando el warp con cv.warpPerspective y ajustando las imágenes para evitar recortes."""
    h, w = img1.shape[:2]

    # Esquinas de la imagen original
    corners = np.array([
        [0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]
    ]).T

    # Aplica las homografías a las esquinas para calcular bounding box
    corners1 = H1 @ corners
    corners2 = H2 @ corners

    # Normaliza para coordenadas homogéneas
    corners1 /= corners1[2, :]
    corners2 /= corners2[2, :]

    # Cálculo de dimensiones nuevas (unir ambas imágenes rectificadas)
    x_min1, x_max1 = np.min(corners1[0, :]), np.max(corners1[0, :])
    y_min1, y_max1 = np.min(corners1[1, :]), np.max(corners1[1, :])
    x_min2, x_max2 = np.min(corners2[0, :]), np.max(corners2[0, :])
    y_min2, y_max2 = np.min(corners2[1, :]), np.max(corners2[1, :])

    new_w = int(max(x_max1, x_max2) - min(x_min1, x_min2)) + 1
    new_h = int(max(y_max1, y_max2) - min(y_min1, y_min2)) + 1

    # Homografía de traslación para que todo quede en coordenadas positivas
    T1 = np.array([
        [1, 0, -min(x_min1, x_min2)],
        [0, 1, -min(y_min1, y_min2)],
        [0, 0, 1]
    ])
    H1_adjusted = T1 @ H1
    H2_adjusted = T1 @ H2

    img1_rect = cv.warpPerspective(img1, H1_adjusted, (new_w, new_h))
    img2_rect = cv.warpPerspective(img2, H2_adjusted, (new_w, new_h))

    return img1_rect, img2_rect


def calcular_F_rectificado(F, H1, H2):
    """Calcula la matriz fundamental rectificada F_rect después de aplicar las homografías H1 y H2:
    F_rect = H2.T * F * H1"""
    F_rect = H2.T @ F @ H1
    # Normaliza para evitar problemas de escala
    F_rect = F_rect / F_rect[2, 2] if F_rect[2, 2] != 0 else F_rect
    return F_rect


def dibujar_lineas_epipolares_rectificadas(img1, img2, F_rect, pts1, pts2, H1, H2, num_lineas=15):
    """Dibuja líneas epipolares horizontales en las imágenes rectificadas para verificar que la
    rectificación ha dejado las líneas epipolares paralelas y horizontales."""
    img1_color = img1.copy() if len(img1.shape) == 3 else cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2_color = img2.copy() if len(img2.shape) == 3 else cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    # Transformar puntos originales a puntos rectificados
    pts1_hom = np.hstack([pts1, np.ones((pts1.shape[0], 1))]).T
    pts2_hom = np.hstack([pts2, np.ones((pts2.shape[0], 1))]).T
    pts1_rect = (H1 @ pts1_hom)
    pts2_rect = (H2 @ pts2_hom)
    pts1_rect /= pts1_rect[2, :]
    pts2_rect /= pts2_rect[2, :]
    pts1_rect = pts1_rect.T[:, :2]
    pts2_rect = pts2_rect.T[:, :2]

    # Seleccionar puntos aleatorios para mostrar
    indices = random.sample(range(len(pts1)), min(num_lineas, len(pts1)))
    pts1_sample = pts1_rect[indices]
    pts2_sample = pts2_rect[indices]

    for pt1, pt2 in zip(pts1_sample, pts2_sample):
        # Línea epipolar en la imagen derecha
        pt1_hom = np.array([pt1[0], pt1[1], 1])
        l2 = F_rect @ pt1_hom  # l2: línea epipolar en img2
        a, b, c = l2
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Dibujar línea epipolar horizontal (b*y + c = 0) en img2
        if abs(b) > 1e-6:
            y = int(-c / b)
            cv.line(img2_color, (0, y), (img2.shape[1], y), color, 1)
        cv.circle(img2_color, (int(pt2[0]), int(pt2[1])), 5, color, -1)

        # Línea epipolar en la imagen izquierda
        pt2_hom = np.array([pt2[0], pt2[1], 1])
        l1 = F_rect.T @ pt2_hom
        a, b, c = l1
        if abs(b) > 1e-6:
            y = int(-c / b)
            cv.line(img1_color, (0, y), (img1.shape[1], y), color, 1)
        cv.circle(img1_color, (int(pt1[0]), int(pt1[1])), 5, color, -1)

    # Mostrar imágenes redimensionadas
    img1_resized = cv.resize(img1_color, None, fx=0.5, fy=0.5)
    img2_resized = cv.resize(img2_color, None, fx=0.5, fy=0.5)

    cv.imshow('Rectified Left - Epipolar Lines', img1_resized)
    cv.imshow('Rectified Right - Epipolar Lines', img2_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


# --------------------------- Funciones para selección manual de puntos ---------------------------
def capturar_puntos_manuales(imagen, titulo_ventana, num_puntos):
    """
    Permite al usuario seleccionar manualmente puntos en una imagen.

    Args:
        imagen: Imagen donde se seleccionarán los puntos
        titulo_ventana: Título de la ventana de selección
        num_puntos: Número de puntos a seleccionar

    Returns:
        Lista de puntos seleccionados [(x1, y1), (x2, y2), ...]
    """
    puntos = []

    def callback_raton(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if len(puntos) < num_puntos:
                puntos.append((x, y))
                cv.circle(imagen_mostrar, (x, y), 5, (0, 255, 0), -1)  # Verde
                cv.imshow(titulo_ventana, imagen_mostrar)
                print(f"Punto {len(puntos)}/{num_puntos}: ({x}, {y})")

    # Prepara la imagen para mostrar con puntos
    if len(imagen.shape) == 2:
        imagen_mostrar = cv.cvtColor(imagen, cv.COLOR_GRAY2BGR)
    else:
        imagen_mostrar = imagen.copy()

    # Reducir la imagen al 50% para visualización
    imagen_mostrar = cv.resize(imagen_mostrar, None, fx=0.5, fy=0.5)

    cv.namedWindow(titulo_ventana)
    cv.setMouseCallback(titulo_ventana, callback_raton)

    print(f"Seleccione {num_puntos} puntos en la imagen. Cierre la ventana cuando termine.")
    while True:
        cv.imshow(titulo_ventana, imagen_mostrar)
        key = cv.waitKey(1) & 0xFF
        if key == 27 or len(puntos) >= num_puntos:  # ESC para salir
            break

    cv.destroyWindow(titulo_ventana)
    return np.array(puntos)


def dibujar_lineas_epipolares_izq_a_der(img_izquierda, img_derecha, F, puntos_izq):
    """
    Dibuja líneas epipolares en la imagen derecha a partir de puntos seleccionados en la izquierda.

    Args:
        img_izquierda: Imagen izquierda (para referencia)
        img_derecha: Imagen derecha donde se dibujarán las líneas
        F: Matriz fundamental
        puntos_izq: Puntos seleccionados en la imagen izquierda
    """
    # Crear copias de las imágenes para dibujar
    img_izq_dib = img_izquierda.copy() if len(img_izquierda.shape) == 3 else cv.cvtColor(img_izquierda,
                                                                                         cv.COLOR_GRAY2BGR)
    img_der_dib = img_derecha.copy() if len(img_derecha.shape) == 3 else cv.cvtColor(img_derecha, cv.COLOR_GRAY2BGR)

    # Dibujar puntos en la imagen izquierda
    for pt in puntos_izq:
        pt = tuple(map(int, pt))
        cv.circle(img_izq_dib, pt, 8, (0, 255, 0), -1)  # Verde

    # Dibujar líneas epipolares en imagen derecha a partir de puntos izquierdos
    for pt in puntos_izq:
        pt_hom = np.array([pt[0], pt[1], 1])
        l2 = F @ pt_hom
        a, b, c = l2

        # Dibujar línea en imagen derecha
        h, w = img_derecha.shape[:2]
        if abs(b) > 1e-6:
            x0, y0 = 0, int(-c / b)
            x1, y1 = w, int(-(a * w + c) / b)
            color = (0, 0, 255)  # Rojo
            cv.line(img_der_dib, (x0, y0), (x1, y1), color, 2)

    # Reducir imágenes para visualización
    img_izq_resized = cv.resize(img_izq_dib, None, fx=0.5, fy=0.5)
    img_der_resized = cv.resize(img_der_dib, None, fx=0.5, fy=0.5)

    # Mostrar imágenes por separado
    cv.imshow('Imagen Izquierda - Puntos seleccionados', img_izq_resized)
    cv.imshow('Imagen Derecha - Lineas Epipolares', img_der_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


# --------------------------- Programa principal ---------------------------
if __name__ == '__main__':
    # Calibración y obtención de parámetros
    P = calibrate(showPics=True)
    if P is None:
        print("Error en calibración. Saliendo...")
        exit()

    K, R, t = descomponer_proyeccion(P)

    # Cargar imágenes
    img_izquierda_orig = cv.imread('foto3.jpg')
    img_derecha_orig = cv.imread('foto4.jpg')

    if img_izquierda_orig is None or img_derecha_orig is None:
        print("No se pudieron cargar las imágenes de prueba.")
        exit()

    # Reducir todas las imágenes al 50% para procesamiento
    img_izquierda = cv.resize(img_izquierda_orig, None, fx=0.5, fy=0.5)
    img_derecha = cv.resize(img_derecha_orig, None, fx=0.5, fy=0.5)

    # Convertir a escala de grises para detección de características
    img_izquierda_gray = cv.cvtColor(img_izquierda, cv.COLOR_BGR2GRAY)
    img_derecha_gray = cv.cvtColor(img_derecha, cv.COLOR_BGR2GRAY)

    # Detectar correspondencias
    pts1, pts2 = detectar_correspondencias(img_izquierda_gray, img_derecha_gray)

    # Calcular matriz fundamental con RANSAC
    F, inliers1, inliers2 = ransac_fundamental(pts1, pts2)

    print("\n--- ANTES DE RECTIFICACIÓN ---")

    # Seleccionar 5 puntos manuales en la imagen izquierda
    print("\nSeleccione 5 puntos en la imagen IZQUIERDA:")
    puntos_izq = capturar_puntos_manuales(img_izquierda, "Seleccione 5 puntos - Imagen Izquierda", 5)

    # Dibujar puntos en izquierda y líneas en derecha
    dibujar_lineas_epipolares_izq_a_der(
        img_izquierda,
        img_derecha,
        F,
        puntos_izq
    )

    # Dibujar líneas epipolares automáticas (15 puntos)
    print("\nVisualización automática con 15 puntos:")
    dibujar_lineas_epipolares(
        img_izquierda,
        img_derecha,
        F,
        inliers1,
        inliers2,
        15
    )

    # Calcular matriz esencial
    E = calcular_matriz_esencial(F, K, K)
    print("\nMatriz esencial (E):\n", E)

    # Calcular homografías de rectificación
    centro = np.array