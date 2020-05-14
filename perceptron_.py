from util import *


def readfile(filename):
    csv_data = pd.read_csv(filename)
    cl = csv_data['Class']
    user = csv_data['User']
    xs = csv_data[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']]
    num = xs.count(axis=1)

    x_mean = xs.mean(axis=1)
    x_min = xs.min(axis=1)
    x_max = xs.max(axis=1)
    x_std = xs.std(axis=1)

    ys = csv_data[['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11']]
    y_mean = ys.mean(axis=1)
    y_min = ys.min(axis=1)
    y_max = ys.max(axis=1)
    y_std = ys.std(axis=1)

    zs = csv_data[['Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10', 'Z11']]
    z_mean = zs.mean(axis=1)
    z_min = zs.min(axis=1)
    z_max = zs.max(axis=1)
    z_std = zs.std(axis=1)

    new_data_feature = pd.DataFrame(
        {'num': num, 'x_mean': x_mean, 'x_std': x_std, 'x_min': x_min, 'x_max': x_max, 'y_mean': y_mean, 'y_std': y_std,
         'y_min': y_min, 'y_max': y_max, 'z_mean': z_mean, 'z_std': z_std, 'z_min': z_min, 'z_max': z_max})

    new_data_name = pd.DataFrame({'Class': cl, 'User': user})

    return new_data_feature, new_data_name


def preprocessing(way, train_feature, train_name, test_feature, test_name):
    names = train_feature.columns
    if way == 'none':
        std_train_feature = train_feature
        std_test_feature = test_feature

    else:
        if way == 'nor':
            # normalize the dataset
            std = Normalizer()

        elif way == 'std':
            # standardize the dataset
            std = StandardScaler()

        std.fit(train_feature)
        std_train_feature = pd.DataFrame(std.transform(train_feature), columns=names)
        std_test_feature = pd.DataFrame(std.transform(test_feature), columns=names)
        # pca
        # std_train_feature = pca_(std_train_feature)
        # std_test_feature = pca_(std_test_feature)

    new_data_train = pd.concat([train_name, std_train_feature], axis=1, join='inner')
    new_data_test = pd.concat([test_name, std_test_feature], axis=1, join='inner')

    return new_data_train, new_data_test


def cross_val(data, init_w, init_w0):
    user_list = [0, 1, 2, 5, 6, 8, 9, 10, 11]

    clf = Perceptron()
    acc_cross = []

    for i in user_list:
        val = data[data['User'] == i]
        val_feature = val.iloc[:, 2:]
        val_label = val.iloc[:, 0]
        train = data[data['User'] != i]
        train_feature = train.iloc[:, 2:]
        train_label = train.iloc[:, 0]

        clf.fit(train_feature, train_label, coef_init=init_w, intercept_init=init_w0)
        pred_label = pd.DataFrame(clf.predict(val_feature))
        acc_cross.append(accuracy_score(pred_label, val_label))

    # print(acc_cross)

    mean = np.mean(acc_cross)
    std = np.std(acc_cross)

    return mean, std


def find_bestpair(ACC, DEV, Cs, gammas):
    index_maxmean = np.argwhere(ACC == np.max(ACC))
    i1 = index_maxmean[:, 0]

    if len(i1) == 1 and DEV[i1[0]] == np.min(DEV):  # clear choice
        best_c = Cs[i1[0]]
        best_gamma = gammas[i1[0]]
        mean = ACC[i1[0]]
        deviation = DEV[i1[0]]
    else:  # max mean -> min dev
        pair = []
        for index in range(len(i1)):
            pair.append(DEV[i1[index]])
            # pair contains the dev with max mean, then we find the min dev among this
        best_index = np.argwhere(pair == np.min(pair))  # position in pair

        best_c = Cs[i1[best_index[0][0]]]
        best_gamma = gammas[i1[best_index[0][0]]]
        mean = ACC[i1[best_index[0][0]]]
        deviation = DEV[i1[best_index[0][0]]]

    return best_c, best_gamma, mean, deviation


def perceptron_mean(data):
    mean_mat = []
    std_mat = []

    w0_mat = np.zeros((50, 5))
    w_mat = np.zeros((50, 5, data.shape[1] - 2))
    for i in tqdm(range(50)):
        init_w0 = np.random.randn(5)
        init_w = np.random.randn(5, data.shape[1] - 2)
        mean, std = cross_val(data, init_w, init_w0)

        mean_mat.append(mean)
        std_mat.append(std)
        w_mat[i] = init_w
        w0_mat[i] = init_w0

    # find best
    best_w, best_w0, mean, deviation = find_bestpair(np.array(mean_mat), np.array(std_mat), w_mat, w0_mat)
    print('mean:', mean)
    print('standard deviation:', deviation)
    print('best_w:', best_w)
    print('best_w0:', best_w0)


train_feature, train_name = readfile('D_train.csv')
test_feature, test_name = readfile('D_test.csv')

# data, test = preprocessing('none', train_feature, train_name, test_feature, test_name)
# print('none')
# perceptron_mean(data)

data, test = preprocessing('std', train_feature, train_name, test_feature, test_name)
print('std')
perceptron_mean(data)

# data, test = preprocessing('nor', train_feature, train_name, test_feature, test_name)
# print('nor')
# perceptron_mean(data)
