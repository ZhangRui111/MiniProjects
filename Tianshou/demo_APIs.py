import numpy as np
import tianshou as ts

from tianshou.data import Batch, ReplayBuffer


def test_Batch():
    """
    batch.split()
    batch.append()
    len(batch)
    :return:
    """
    # data is a batch involves 4 transitions
    data = Batch(
        obs=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]),
        rew=np.array([0, 0, 0, 1])
    )
    index = [0, 1]  # pick the first 2 transition
    print(data[0])
    print(len(data))
    print("--------------------")
    data.append(Batch(
        obs=np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0]]),
        rew=np.array([-1, -1, -1, -1])
    ))
    print(data)
    print(len(data))
    print("--------------------")
    # the last batch might has size less than 3
    for mini_batch in data.split(size=3, permute=False):
        print(mini_batch)


def test_ReplayBuffer():
    """
    tianshou.data.ReplayBuffer
    buf.add()
    buf.get()
    buf.update()
    buf.sample()
    buf.reset()
    len(buf)
    :return:
    """
    buf1 = ReplayBuffer(size=15)
    for i in range(3):
        buf1.add(obs=i, act=i, rew=i, done=i, obs_next=i+1, info={}, weight=None)
    print(len(buf1))
    print(buf1.obs)
    buf2 = ReplayBuffer(size=10)
    for i in range(15):
        buf2.add(obs=i, act=i, rew=i, done=i, obs_next=i+1, info={}, weight=None)
    print(buf2.obs)
    buf1.update(buf2)
    print(buf1.obs)
    index = [1, 3, 5]
    # key is an obligatory args
    print(buf2.get(index, key='obs'))
    print('--------------------')
    sample_data, indice = buf2.sample(batch_size=4)
    print(sample_data, indice)
    print(sample_data.obs == buf2[indice].obs)
    print('--------------------')
    # buf.reset() only resets the index, not the content.
    print(len(buf2))
    buf2.reset()
    print(len(buf2))
    print(buf2)
    print('--------------------')


def main():
    # test_Batch()
    test_ReplayBuffer()


if __name__ == '__main__':
    main()
