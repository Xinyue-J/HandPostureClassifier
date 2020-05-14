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


def svm_(C, gamma, train, test):
    # process of svm classifier
    train_feature = train.iloc[:, 2:]
    train_label = train.iloc[:, 0]
    test_feature = test.iloc[:, 2:]
    test_label = test.iloc[:, 0]

    svm_model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    svm_model.fit(train_feature, train_label)
    pred_test_label = pd.DataFrame(svm_model.predict(test_feature))

    acc_svm = accuracy_score(pred_test_label[0], test_label)
    return acc_svm, svm_model


def cross_val(C, gamma, data):
    user_list = [0, 1, 2, 5, 6, 8, 9, 10, 11]
    acc_cross = []
    for i in user_list:
        val = data[data['User'] == i]
        train = data[data['User'] != i]
        acc, model = svm_(C, gamma, train, val)
        acc_cross.append(acc)
    mean = np.mean(acc_cross)
    std = np.std(acc_cross)

    # choose the best model in cross validation
    index_max = np.argwhere(acc_cross == np.max(acc_cross))  # user_list[index_max]
    val = data[data['User'] == user_list[index_max[0][0]]]
    train = data[data['User'] != user_list[index_max[0][0]]]
    acc_train, model_best = svm_(C, gamma, train, val)
    return acc_train, model_best, mean, std


def svm_main(data, test):
    Cs = np.logspace(0, 3, 10)
    gammas = np.logspace(-3, 0, 10)

    acc_mat = np.zeros([len(Cs), len(gammas)])

    for i in tqdm(range(len(Cs))):
        for j in range(len(gammas)):
            acc, drop, drop, drop = cross_val(Cs[i], gammas[j], data)
            acc_mat[i, j] = acc

    # find best c and gamma
    index_best = np.argwhere(acc_mat == np.max(acc_mat))

    index_x = index_best[:, 0]
    index_y = index_best[:, 1]
    best_c = Cs[index_x[0]]
    best_gamma = gammas[index_y[0]]
    print('c=', best_c)
    print('gamma=', best_gamma)

    # use the best parameter to calculate accuracy
    acc_train, model_best, mean, std = cross_val(best_c, best_gamma, data)

    print('accuracy on train:', acc_train)
    acc_test = test_acc(test, model_best)
    print('accuracy on test:', acc_test)
    print('mean:', mean)
    print('std:', std)


def test_acc(test, model):
    # calculate the accuracy on test
    test_feature = test.iloc[:, 2:]
    test_label = test.iloc[:, 0]
    pred_test_label = pd.DataFrame(model.predict(test_feature))
    acc_test = accuracy_score(pred_test_label[0], test_label)

    sns.set()
    f, ax = plt.subplots()

    C2 = confusion_matrix(test_label, pred_test_label, labels=[1, 2, 3, 4, 5])
    print(C2)
    sns.heatmap(C2, annot=True, ax=ax)

    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.savefig('1.png')
    return acc_test


train_feature, train_name = readfile('D_train.csv')
test_feature, test_name = readfile('D_test.csv')
data, test = preprocessing('std', train_feature, train_name, test_feature, test_name)
svm_main(data, test)
