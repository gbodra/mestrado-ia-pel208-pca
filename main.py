import logging
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_file(filename):
    x_input = []
    y_input = []
    f = open("data/" + filename + ".txt")
    lines = f.readlines()

    for line in lines:
        if filename == "Books_attend_grade":
            x0, x1, y = line.split(";")
            x_input.append([float(x0), float(x1)])
            y_input.append(float(y))
        else:
            line = line.strip("\n")
            x, y = line.split(";")
            x_input.append(float(x))
            y_input.append(float(y))

    return x_input, y_input


def subtract_mean(x, y):
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    x_adjusted = []
    y_adjusted = []

    for i in range(len(x)):
        x_adjusted.append(x[i] - x_mean)
        y_adjusted.append(y[i] - y_mean)

    return x_adjusted, y_adjusted


def covariance(a, b):
    covariance_result = 0

    for i in range(len(a)):
        covariance_result += (a[i] * b[i]) / (len(a) - 1)

    return covariance_result


def covariance_matrix(x, y):
    cov_matrix = []
    cov_x_x = covariance(x, x)
    cov_x_y = covariance(x, y)
    cov_matrix.append([cov_x_x, cov_x_y])
    cov_y_x = covariance(y, x)
    cov_y_y = covariance(y, y)
    cov_matrix.append([cov_y_x, cov_y_y])

    return cov_matrix


def matrix_empty(rows, cols):
    """
    Cria uma matriz vazia
        :param rows: número de linhas da matriz
        :param cols: número de colunas da matriz

        :return: matriz preenchida com 0.0
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def matrix_transpose(m):
    """
    Retorna a matriz transposta
        :param m: matriz a ser transposta

        :return: resultado da matriz de entrada transposta
    """
    if not isinstance(m[0], list):
        m = [m]

    rows = len(m)
    cols = len(m[0])

    mt = matrix_empty(cols, rows)

    for i in range(rows):
        for j in range(cols):
            mt[j][i] = m[i][j]

    return mt


def matrix_multiply(a, b):
    """
    Retorna o produto da multiplicação da matriz a com b
        :param a: primeira matriz
        :param b: segunda matriz

        :return: matriz resultante
    """
    rows_a = len(a)
    cols_a = len(a[0])
    rows_b = len(b)
    cols_b = len(b[0])
    if cols_a != rows_b:
        raise ArithmeticError('O número de colunas da matriz a deve ser igual ao número de linhas da matriz b.')

    result_matrix = matrix_empty(rows_a, cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            total = 0
            for ii in range(cols_a):
                total += a[i][ii] * b[ii][j]
            result_matrix[i][j] = total

    return result_matrix


def vector_to_matrix(vector):
    if isinstance(vector[0], list):
        raise ArithmeticError('Não é um vetor')

    matrix = matrix_empty(len(vector), 1)
    for i in range(len(vector)):
        matrix[i][0] = vector[i]

    return matrix


def calc_pca(dataset):
    x, y = load_file(dataset)
    logging.info("*******INPUT*******")
    logging.info(x)
    logging.info(y)

    logging.info("*******MEAN SUBTRACTED*******")
    x_minus_mean, y_minus_mean = subtract_mean(x, y)
    logging.info(x_minus_mean)
    logging.info(y_minus_mean)

    logging.info("*******COVARIANCE MATRIX*******")
    cov = covariance_matrix(x_minus_mean, y_minus_mean)
    logging.info(cov)

    logging.info("*******AUTOVALOR / AUTOVETOR*******")
    cov_np = np.array(cov)
    w, v = np.linalg.eig(cov_np)
    logging.info(w)
    logging.info(v)

    logging.info("*******COMPONENTES*******")
    feature_vector = v[:, np.argmax(w)]
    gradient_1 = feature_vector[1] / feature_vector[0]
    feature_vector_2 = v[:, np.argmin(w)]
    gradient_2 = feature_vector_2[1] / feature_vector_2[0]
    logging.info(feature_vector)

    logging.info("*******VALORES FINAIS*******")
    feature_vector = vector_to_matrix(feature_vector)
    feature_vector_t = matrix_transpose(feature_vector)
    input_x_y = [list(a) for a in zip(x_minus_mean, y_minus_mean)]
    input_t = matrix_transpose(input_x_y)
    logging.info(feature_vector_t)
    logging.info(input_t)
    result = matrix_multiply(feature_vector_t, input_t)
    logging.info(result)

    logging.info("*******PLOT*******")
    plt.plot(x_minus_mean, y_minus_mean, '+', color='black')
    plt.plot(0, 0, 'x', color='blue')
    min_x = min(x_minus_mean)
    max_x = max(x_minus_mean)
    pca_1_x = [min_x, 0, max_x]
    pca_1_y = [min_x * gradient_1, 0, max_x * gradient_1]
    plt.plot(pca_1_x, pca_1_y, color='red')

    pca_2_x = [min_x, 0, max_x]
    pca_2_y = [min_x * gradient_2, 0, max_x * gradient_2]
    plt.plot(pca_2_x, pca_2_y, color='orange')
    plt.title("Dataset: " + dataset)
    plt.show()


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
calc_pca("dummy")

calc_pca("alpswater")

calc_pca("US_Census")

calc_pca("economic_growth")
