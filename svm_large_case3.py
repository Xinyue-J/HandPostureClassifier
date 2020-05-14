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

    # pca
    # new_data_feature = pca_(new_data_feature)

    return new_data_train, new_data_test


def get_acc(data, test):
    # use the best parameter to calculate accuracy
    clf = svm.SVC(C=10.0, gamma=0.021544346900318832, kernel='rbf')
    clf.fit(data.iloc[:, 2:], data.iloc[:, 0])
    data_feature = data.iloc[:, 2:]
    data_label = data.iloc[:, 0]
    test_feature = test.iloc[:, 2:]
    test_label = test.iloc[:, 0]
    print('acc_train:', clf.score(data_feature, data_label))
    pred_test_label = pd.DataFrame(clf.predict(test_feature))
    print('acc_test:', accuracy_score(pred_test_label, test_label))
    print('acc_test:', clf.score(test_feature, test_label))

    # confusion matrix

    C = confusion_matrix(test_label, pred_test_label, labels=[1, 2, 3, 4, 5])
    print(C)
    # sns.set()
    # f, ax = plt.subplots()
    # sns.heatmap(C2, annot=True, ax=ax)
    # ax.set_title('confusion matrix')
    # ax.set_xlabel('predict')
    # ax.set_ylabel('true')
    # plt.savefig('larger.png')


train_feature, train_name = readfile('D_train_large.csv')
test_feature, test_name = readfile('D_test.csv')

data, test = preprocessing('std', train_feature, train_name, test_feature, test_name)
get_acc(data, test)
