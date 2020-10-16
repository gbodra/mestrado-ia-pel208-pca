from sklearn.decomposition import PCA


def load_file(filename):
    x_input = []
    f = open("data/" + filename + ".txt")
    lines = f.readlines()

    for line in lines:
        if filename == "Books_attend_grade":
            x0, x1, y = line.split(";")
            x_input.append([float(x0), float(x1), float(y)])
        else:
            line = line.strip("\n")
            x, y = line.split(";")
            x_input.append([float(x), float(y)])
            # y_input.append(float(y))

    return x_input


x = load_file("dummy")
pca = PCA(n_components=2)
pca.fit(x)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)
