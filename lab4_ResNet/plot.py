import matplotlib.pyplot as plt

def plot():
    resnet18_pretrain_train = [0.7508, 0.7936, 0.8124, 0.8236, 0.8312, 0.8372, 0.8416, 0.8468, 0.8522, 0.8548]
    resnet18_pretrain_test = [0.7672, 0.8022, 0.8156, 0.8257, 0.8262, 0.8294, 0.8353, 0.8346, 0.8354, 0.8341]
    resnet18_train = [0.7347, 0.7351, 0.7350, 0.7349, 0.7347, 0.7347, 0.7346, 0.7349, 0.7343, 0.7348]
    resnet18_test = [0.7335, 0.7335, 0.7335, 0.7335, 0.7338, 0.7333, 0.7335, 0.7340, 0.7279, 0.7336]
    resnet50_pretrain_train = [0.7823, 0.8202, 0.8342, 0.8434, 0.8507, 0.8584, 0.8650, 0.8739, 0.8708, 0.8842]
    resnet50_pretrain_test = [0.8273, 0.8341, 0.8367, 0.8397, 0.8415, 0.8397, 0.8407, 0.8279, 0.8333, 0.8152]
    resnet50_train = [0.7247, 0.7326, 0.7332, 0.7337, 0.7339, 0.7340, 0.7344, 0.7345, 0.7343, 0.7343]
    resnet50_test = [0.7325, 0.7330, 0.7332, 0.7323, 0.7315, 0.7298, 0.7295, 0.7289, 0.7279, 0.7278]
    

    plt.title('resnet50')
    plt.plot(resnet50_pretrain_train, label='train(pretrained)')
    plt.plot(resnet50_pretrain_test, label='test(pretrained)')
    plt.plot(resnet50_train, label='train')
    plt.plot(resnet50_test, label='test')
    plt.legend(loc='upper left')
    plt.ylim(0.7, 1.0)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('resnet50_comp.jpg')

if __name__ == '__main__':
    plot()