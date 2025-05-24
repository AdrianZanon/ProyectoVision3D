import numpy as np
import random
import pytest
from SeisPrimeros_puntos import descomponer_proyeccion, normalizar_puntos, estimar_F_8_puntos, calcular_error_fundamental, calcular_matriz_esencial, ransac_fundamental, construir_H

def test_descomponer_proyeccion():
    # Matriz P de ejemplo
    K = np.array([[1000, 0, 320],
                  [0, 1000, 240],
                  [0,    0,   1]])
    R = np.eye(3)
    t = np.array([[1], [2], [3]])
    Rt = np.hstack((R, t))
    P = K @ Rt

    K_est, R_est, t_est = descomponer_proyeccion(P)

    assert K_est.shape == (3, 3)
    assert R_est.shape == (3, 3)
    assert t_est.shape == (3,)
    assert np.allclose(K_est[2, 2], 1, atol=1e-1)

def test_normalizar_puntos():
    pts = np.array([[0, 0], [1, 0], [0, 1]])
    pts_norm, T = normalizar_puntos(pts)

    assert pts_norm.shape == (3, 3)
    assert T.shape == (3, 3)
    mean = np.mean(pts_norm[:, :2], axis=0)
    assert np.allclose(mean, [0, 0], atol=1e-1)

def test_estimar_F_8_puntos():
    pts1 = np.array([
        [10, 10],
        [20, 15],
        [30, 25],
        [40, 30],
        [50, 40],
        [60, 45],
        [70, 55],
        [80, 60]
    ])
    pts2 = np.array([
        [11, 10.5],
        [21, 15.3],
        [31, 25.2],
        [41, 30.4],
        [51, 40.1],
        [61, 45.6],
        [71, 55.4],
        [81, 60.2]
    ])

    F = estimar_F_8_puntos(pts1, pts2)

    assert F.shape == (3, 3)
    assert np.linalg.matrix_rank(F) == 2

def test_error_epipolar_cercano_a_0():
    F = np.array([[0, -0.0004, 0.08],
                  [0.0004, 0, -0.06],
                  [-0.09, 0.05, 1]])
    pt1 = np.array([100, 50])
    pt2 = np.array([102, 49])
    error = calcular_error_fundamental(F, pt1, pt2)
    assert error < 5.0

def test_ransac_fundamental():
    np.random.seed(0)
    random.seed(0)

    pts1 = np.array([[10, 20], [20, 30], [30, 40], [40, 50],
                     [50, 60], [60, 70], [70, 80], [80, 90],
                     [15, 25], [25, 35]])

    ruido = np.random.normal(0, 0.3, pts1.shape)
    pts2 = pts1 + ruido

    F_ransac, inliers1, inliers2 = ransac_fundamental(pts1, pts2, iteraciones=500, umbral=1.0)

    assert F_ransac.shape == (3, 3)
    assert inliers1.shape[0] == inliers2.shape[0]
    assert inliers1.shape[1] == 2
    assert inliers1.shape[0] >= 8 #Que hayan 8

    # Verifica que todos los inliers estén dentro del umbral
    for p1, p2 in zip(inliers1, inliers2):
        error = calcular_error_fundamental(F_ransac, p1, p2)
        assert error < 1.0

def test_calcular_matriz_esencial():
    F = np.array([[1e-6, 2e-6, -3e-3],
                  [-2e-6, 1e-6, 4e-3],
                  [2e-3, -1e-3, 1]])

    K1 = np.array([[1000, 0, 320],
                   [0, 1000, 240],
                   [0, 0, 1]])

    K2 = np.array([[950, 0, 300],
                   [0, 950, 250],
                   [0, 0, 1]])

    E = calcular_matriz_esencial(F, K1, K2)

    assert E.shape == (3, 3)
    assert np.linalg.matrix_rank(E) == 2
    assert not np.allclose(E, 0)

def test_construir_H():
    # Me invento posición fuera
    e = np.array([300, 200, 1])  # homogéneo
    img_shape = (400, 600)  # altura, anchura

    H = construir_H(e, img_shape)

    assert H.shape == (3, 3)

    e_transformado = H @ e
    e_transformado = e_transformado / e_transformado[2]  #Homo u know

    #Coordenada z cercana a 0
    assert abs(e_transformado[2]) < 1e-3 or abs(e_transformado[0]) > 1e4
    assert not np.allclose(H, np.eye(3))
