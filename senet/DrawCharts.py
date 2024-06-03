import numpy as np
import matplotlib.pyplot as plt


# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)


# 不同长度数据，统一为一个标准，倍乘x轴
def multiple_equal(x, y):
    x_len = len(x)
    y_len = len(y)
    times = x_len/y_len
    y_times = [i * times for i in y]
    return y_times


if __name__ == "__main__":

    train_loss_path = r"train_loss.txt"
    train_acc_path = r"train_acc.txt"

    y_train_loss = data_read(train_loss_path)
    y_train_acc = data_read(train_acc_path)

    x_train_loss = range(len(y_train_loss))
    x_train_acc = multiple_equal(x_train_loss, range(len(y_train_acc)))

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')
    plt.ylabel('accuracy')

    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.plot(x_train_acc, y_train_acc,  color='red', linestyle="solid", label="train accuracy")
    plt.legend()

    plt.title('Accuracy curve')
    plt.show()
    plt.savefig("show.png")
