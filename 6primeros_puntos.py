import numpy as np
import cv2 as cv
import glob
import os
import random
def calibrate(showPics=True):
    # Ruta de las imágenes de calibración
    calibrationDir = r'C:\Users\javip\OneDrive\Escritorio\Nueva carpeta (2)\ProyectoVision3D\Camara\Fotos_Javier'
    imgPathList = glob.glob(os.path.join(calibrationDir, 'Foto_*.jpg'))

    if not imgPathList:
        print("No se encontraron imágenes. Verifica la ruta y los nombres de los archivos.")
        return None, None, None, None, None

    # Parámetros ajedrez
    nCols = 8  #Columnas
    nRows = 6  #Filas
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

    print('Matriz intrínseca (K):\n', K)                     #Matriz intrínsica
    print("Error de reproyección (pixeles): {:.4f}".format(repError))

    # Guardar resultados
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'calibration.npz')
    np.savez(paramPath, repError=repError, camMatrix=K,      #Guardado de las variables de la calibración
             distCoeff=distCoeff, rvecs=rvecs, tvecs=tvecs)

    R, _ = cv.Rodrigues(rvecs[0])  #Cálculo de la matriz de rotación con Rodrigues
    Rt = np.hstack((R, tvecs[0]))  #Matriz de rotación y traslación combinadas para pasar de 3x3 a 3x4
    P = K @ Rt #Devolución de la matriz de P
    return P


'''def test_calibrate_output():
    P, K, dist, rvecs, tvecs = calibrate(showPics=False)
    assert P.shape == (3, 4)
    assert K.shape == (3, 3)
    assert isinstance(dist, np.ndarray)
    assert len(rvecs) == len(tvecs) }'''



def descomponer_proyeccion(P):
    M = P[:, :3]                               #Extracción de la última columna
    K, R = np.linalg.qr(np.linalg.inv(M))
    K = np.linalg.inv(K)                       #Inverisón linela de K
    R = np.linalg.inv(R)
    t = np.linalg.inv(K) @ P[:, 3]

    if K[2, 2] < 0:
        K *= -1
        R *= -1

    return K, R, t

'''def test_descomponer_proyeccion():
    P, K_gt, _, rvecs, tvecs = calibrate(showPics=False)
    K, R, t = descomponer_proyeccion(P)
    assert K.shape == (3, 3)
    assert R.shape == (3, 3)
    assert t.shape == (3,)'''




"----------------------------------- Etapa 3 -----------------------------------------"

def detectar_correspondencias(img1, img2):
    sift = cv.SIFT_create()                       #Busqueda de puntos en común
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)


    bf = cv.BFMatcher()                           #Emparejado de los puntos clave de ambas imágenes
    matches = bf.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []
                                                  #Encuentra las mejores coincidencias para los puntos comunes
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    return np.array(pts1), np.array(pts2)  #Devolución de los arrays de puntos de ambas imagenes

def normalizar_puntos(pts):
    mean = np.mean(pts, axis=0)            #Cálcula la media de los puntos heredados
    std = np.std(pts)                      #La desviación estandar es raiz de 2
    T = np.array([
        [np.sqrt(2)/std, 0, -mean[0]*np.sqrt(2)/std],
        [0, np.sqrt(2)/std, -mean[1]*np.sqrt(2)/std],
        [0, 0, 1]
    ])                                     #Construcción de una matriz de normalización
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])  #Converit los puntos a coordenadas homogeneas
    pts_norm = (T @ pts_hom.T).T           #Aplica la transformada
    return pts_norm, T

def estimar_F_8_puntos(pts1, pts2):
    pts1_norm, T1 = normalizar_puntos(pts1)
    pts2_norm, T2 = normalizar_puntos(pts2)

    A = []
    for p1, p2 in zip(pts1_norm, pts2_norm):
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])  #Cada fila de la matriz A es una ecuación lineal de un pto

    A = np.array(A)
    _, _, V = np.linalg.svd(A)    #Se resuelve el sistema utilizando SVD
    F = V[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt #Fuerza a aque la matriz F tenga dimensiones 2x2 para la proyección

    # Denormalizar
    F_final = T2.T @ F_rank2 @ T1
    return F_final / F_final[2, 2]  #Se devuelen los puntos al sistema normal porque se ha calculado F con puntos normalizados

def calcular_error_fundamental(F, pt1, pt2):
    pt1_h = np.append(pt1, 1)
    pt2_h = np.append(pt2, 1)
    return abs(pt2_h.T @ F @ pt1_h)

def ransac_fundamental(pts1, pts2, iteraciones=2000, umbral=0.01):
    max_inliers = []
    mejor_F = None

    for _ in range(iteraciones):                         #Prueba de las mejores soluciones posibles
        idx = random.sample(range(len(pts1)), 8)      #Elección aleatoria de pts
        F = estimar_F_8_puntos(pts1[idx], pts2[idx])

        inliers = []
        for i in range(len(pts1)):                       #Cálculo error punto - linea epipolar
            error = calcular_error_fundamental(F, pts1[i], pts2[i])
            if error < umbral:
                inliers.append(i)

        if len(inliers) > len(max_inliers):              #Actualizar al mejor modelo posible
            max_inliers = inliers
            mejor_F = F

    pts1_inliers = pts1[max_inliers]                     #Reconstrucción de los mejores resultados
    pts2_inliers = pts2[max_inliers]

    return mejor_F, pts1_inliers, pts2_inliers #Devolución de la matriz fuandamental

def dibujar_lineas_epipolares(img1, img2, F, pts1, pts2, num_lineas=10):
    # Convertir a color para dibujar en color
    img1_color = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    # Seleccionar num_lineas puntos aleatorios
    indices = random.sample(range(len(pts1)), min(num_lineas, len(pts1)))
    pts1_sample = pts1[indices]
    pts2_sample = pts2[indices]

    # Dibujar líneas epipolares en img1 para puntos de img2
    for pt1, pt2 in zip(pts1_sample, pts2_sample):
        # Línea epipolar en imagen 1 a partir de punto en imagen 2
        l1 = F.T @ np.array([pt2[0], pt2[1], 1])
        a, b, c = l1
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, x1 = 0, img1.shape[1]
        y0 = int(-c / b) if b != 0 else 0
        y1 = int(-(a * x1 + c) / b) if b != 0 else 0
        cv.line(img1_color, (x0, y0), (x1, y1), color, 1)
        cv.circle(img1_color, tuple(np.int32(pt1)), 5, color, -1)

        # Línea epipolar en imagen 2 a partir de punto en imagen 1
        l2 = F @ np.array([pt1[0], pt1[1], 1])
        a, b, c = l2
        x0, x1 = 0, img2.shape[1]
        y0 = int(-c / b) if b != 0 else 0
        y1 = int(-(a * x1 + c) / b) if b != 0 else 0
        cv.line(img2_color, (x0, y0), (x1, y1), color, 1)
        cv.circle(img2_color, tuple(np.int32(pt2)), 5, color, -1)

    # Mostrar imágenes


    img1_resized = cv.resize(img1_color, None, fx=0.5, fy=0.5)
    img2_resized = cv.resize(img2_color, None, fx=0.5, fy=0.5)

    cv.imshow('Epipolar Lines - Izq', img1_resized)
    cv.imshow('Epipolar Lines - Der', img2_resized)

    cv.waitKey(0)
    cv.destroyAllWindows()


'''def test_fundamental_matrix():
    img1 = cv.imread('foto3.jpg', 0)
    img2 = cv.imread('foto4.jpg', 0)
    pts1, pts2 = detectar_correspondencias(img1, img2)
    F, inl1, inl2 = ransac_fundamental(pts1, pts2)
    assert F.shape == (3, 3)
    assert np.allclose(np.linalg.matrix_rank(F), 2)'''



"--------------------------------- Pto 4 ------------------------"

def calcular_matriz_esencial(F, K1, K2):
    E= K2.T @ F @ K1
    return E

"--------------------------------- Pto 6 --------------------------"

def epipolo(F):
    # El epipolo está en el nullspace de F (para la imagen derecha) o F.T (para la izquierda)
    U, S, Vt = np.linalg.svd(F)
    e = Vt[-1]
    return e / e[2]

def construir_H(e, img_shape):
    """Construye la homografía H que envía el epipolo al infinito en dirección horizontal"""
    w, h = img_shape[1], img_shape[0]
    # Paso 1: trasladar origen al centro de imagen
    T = np.array([
        [1, 0, -w/2],
        [0, 1, -h/2],
        [0, 0,    1]
    ])

    e_ = T @ e
    ex, ey = e_[0], e_[1]

    r = np.sqrt(ex**2 + ey**2)
    alpha = np.arctan2(ey, ex)

    # Paso 2: rotar epipolo sobre el eje z para alinearlo con x
    R = np.array([
        [np.cos(alpha), np.sin(alpha), 0],
        [-np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])

    e_rot = R @ e_
    if abs(e_rot[0]) < 1e-6:
        print("Epipolo demasiado cerca del eje y; no se puede rectificar con esta técnica.")
        return np.eye(3)  # Retorna identidad como fallback

    # Paso 3: proyectar epipolo al infinito en dirección x (hacer que z = 0)
    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1/e_rot[0], 0, 1]
    ])

    # Homografía total: H = G * R * T
    H = G @ R @ T
    return H

def rectificacion_estereoscopica_no_calibrada_manual(F, img1, img2):

    # Epipolos
    e2 = epipolo(F)        # epipolo en imagen derecha (img2)
    e1 = epipolo(F.T)      # epipolo en imagen izquierda (img1)

    # Homografías que llevan los epipolos al infinito horizontalmente
    H2 = construir_H(e2, img2.shape)
    H1 = construir_H(e1, img1.shape)

    return H1, H2

def aplicar_rectificacion_manual(img1, img2, H1, H2):
    h, w = img1.shape
    img1_rect = cv.warpPerspective(img1, H1, (w, h))
    img2_rect = cv.warpPerspective(img2, H2, (w, h))
    return img1_rect, img2_rect


if __name__ == '__main__':
    P = calibrate(showPics=True)
    K, R, t = descomponer_proyeccion(P)
    img1 = cv.imread('foto3.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('foto4.jpg', cv.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("No se pudieron cargar las imágenes de prueba.")
        exit()

    pts1, pts2 = detectar_correspondencias(img1, img2)
    F, inliers1, inliers2 = ransac_fundamental(pts1, pts2)
    dibujar_lineas_epipolares(img1, img2, F, inliers1, inliers2)

    E = calcular_matriz_esencial(F, K, K)
    print("Matriz esencial (E):\n", E)

    H1, H2 = rectificacion_estereoscopica_no_calibrada_manual(F, img1, img2)
    img1_rect, img2_rect = aplicar_rectificacion_manual(img1, img2, H1, H2)

    # Mostrar resultados de rectificación
    img1_resized = cv.resize(img1_rect, None, fx=0.5, fy=0.5)
    img2_resized = cv.resize(img2_rect, None, fx=0.5, fy=0.5)
    cv.imshow('Imagen 1 Rectificada', img1_resized)
    cv.imshow('Imagen 2 Rectificada', img2_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()



