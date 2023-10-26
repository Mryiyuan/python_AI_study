import mnist_train
import test_one_img

if __name__ == '__main__':
    # 第一步,训练模型
    print('Training model...')
    mnist_train.train()

    # 第二步,加载模型进行预测
    print('Loading model and testing...')
    test_one_img.test()