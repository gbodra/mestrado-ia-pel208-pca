import logging
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_file(filename):
    """
    Carrega o dataset para a memória
        :param filename: nome do arquivo

        :return: matriz x e y
    """
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
    """
    Subtrair a média de todo o dataset
        :param x: entrada x
        :param y: entrada y

        :return: matriz ajustada
    """
    if isinstance(x[0], list):
        x_zipped = list(zip(*x))
        x1_mean = sum(x_zipped[0]) / len(x_zipped[0])
        x2_mean = sum(x_zipped[1]) / len(x_zipped[1])
    else:
        x_mean = sum(x) / len(x)

    y_mean = sum(y) / len(y)
    x_adjusted = []
    y_adjusted = []

    for i in range(len(x)):
        if isinstance(x[0], list):
            x_adjusted.append([x[i][0] - x1_mean, x[i][1] - x2_mean])
        else:
            x_adjusted.append(x[i] - x_mean)

        y_adjusted.append(y[i] - y_mean)

    return x_adjusted, y_adjusted


def covariance(a, b):
    """
    Calcula a covariancia entre dois pontos
        :param a: ponto a
        :param b: ponto b

        :return: covariancia entre a e b
    """
    covariance_result = 0

    for i in range(len(a)):
        covariance_result += (a[i] * b[i]) / (len(a) - 1)

    return covariance_result


def covariance_matrix(x, y):
    """
    Calcula matriz de covariancia
        :param x: entrada x
        :param y: entrada y

        :return: matriz de covariancia
    """
    cov_matrix = []

    if not isinstance(x[0], list):
        cov_x_x = covariance(x, x)
        cov_x_y = covariance(x, y)
        cov_matrix.append([cov_x_x, cov_x_y])
        cov_y_x = covariance(y, x)
        cov_y_y = covariance(y, y)
        cov_matrix.append([cov_y_x, cov_y_y])
    else:
        x_zipped = list(zip(*x))
        x1 = x_zipped[0]
        x2 = x_zipped[1]
        cov_x_x = covariance(x1, x1)
        cov_x_y = covariance(x1, y)
        cov_x_x2 = covariance(x1, x2)
        cov_matrix.append([cov_x_x, cov_x_y, cov_x_x2])
        cov_y_x = covariance(y, x1)
        cov_y_y = covariance(y, y)
        cov_y_x2 = covariance(y, x2)
        cov_matrix.append([cov_y_x, cov_y_y, cov_y_x2])
        cov_x2_x = covariance(x2, x1)
        cov_x2_y = covariance(x2, y)
        cov_x2_x2 = covariance(x2, x2)
        cov_matrix.append([cov_x2_x, cov_x2_y, cov_x2_x2])

    return cov_matrix


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

    logging.info("*******PLOT*******")
    plt.plot(x_minus_mean, y_minus_mean, '+', color='black', label='Dados ajustados')
    plt.plot(0, 0, 'x', color='blue', label='Média')
    min_x = min(x_minus_mean)
    max_x = max(x_minus_mean)
    pca_1_x = [min_x, 0, max_x]
    pca_1_y = [min_x * gradient_1, 0, max_x * gradient_1]
    plt.plot(pca_1_x, pca_1_y, color='red', label='Componente principal')

    pca_2_x = [min_x, 0, max_x]
    pca_2_y = [min_x * gradient_2, 0, max_x * gradient_2]
    plt.plot(pca_2_x, pca_2_y, color='orange', label='Componente secundário')

    if dataset == "US_Census":
        x_mq = 2010 - (sum(x) / len(x))
        y_mq = 286.9128 - (sum(y) / len(y))
        plt.plot([x_minus_mean[0], x_mq], [y_minus_mean[0], y_mq], color='purple', label='Mínimos Quadrados')

    if dataset == "alpswater":
        x_mq = 31.06 - (sum(x) / len(x))
        y_mq = 214.3658 - (sum(y) / len(y))
        plt.plot([x_minus_mean[0], x_mq], [y_minus_mean[0], y_mq], color='purple', label='Mínimos Quadrados')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.title("Dataset: " + dataset)
    plt.show()


logging.basicConfig(stream=sys.stderr, level=logging.INFO)

calc_pca("alpswater")

calc_pca("US_Census")
