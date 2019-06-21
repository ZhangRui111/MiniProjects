import numpy as np
import tensorflow as tf
import time

from utils import extract_cifar10

data_path = '../data/'


def main():
    # 获取cifar10数据
    cifar10_data_set = extract_cifar10.Cifar10DataSet(data_path)
    test_images, test_labels = cifar10_data_set.test_data()

    # 定义会话
    sess = tf.InteractiveSession()

    # 保存模型训练数据
    saver = tf.train.Saver()

    # 所有变量进行初始化
    sess.run(tf.global_variables_initializer())

    # tensorboard: test&train分开记录
    merge_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('../logs/alexnet/' + '/train', sess.graph)  # 保存位置
    test_writer = tf.summary.FileWriter('../logs/alexnet/' + '/test', sess.graph)

    # 进行训练
    start_time = time.time()
    for i in range(60000):
        # 获取训练数据
        # print i,'1'
        batch_xs, batch_ys = cifar10_data_set.next_train_batch(50)
        # print i,'2'

        # 每迭代100个 batch，对当前训练数据进行测试，输出当前预测准确率
        if i % 1000 == 0:
            # Train_accuracy:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            # add to log
            _, result = sess.run([train_step, merge_op], {x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            train_writer.add_summary(result, i)
            print("step %d, training accuracy %g" % (i, train_accuracy))

            # 计算间隔时间
            end_time = time.time()
            print('time: ', (end_time - start_time))
            start_time = end_time

        if (i + 1) % 5000 == 0:
            # Test_accuracy
            avg = 0
            for j in range(20):
                avg += accuracy.eval(
                    feed_dict={x: test_images[j * 50:j * 50 + 50], y_: test_labels[j * 50:j * 50 + 50], keep_prob: 1.0})
            avg /= 20
            print("test accuracy %g" % avg)
            # add to log
            _, result = sess.run(
                [train_step, merge_op], {x: test_images[j * 50:j * 50 + 50], y_: test_labels[j * 50:j * 50 + 50], keep_prob: 1.0})
            test_writer.add_summary(result, i)

            # 保存模型参数
            if not tf.gfile.Exists('model_data'):
                tf.gfile.MakeDirs('model_data')
            save_path = saver.save(sess, "model_data/model.ckpt")
            print("Model saved in file: ", save_path)

        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    # 输出整体测试数据的情况
    avg = 0
    for i in range(200):
        avg += accuracy.eval(
            feed_dict={x: test_images[i * 50:i * 50 + 50], y_: test_labels[i * 50:i * 50 + 50], keep_prob: 1.0})
    avg /= 200
    print("test accuracy %g" % avg)

    # 关闭会话
    sess.close()


if __name__ == '__main__':
    main()
