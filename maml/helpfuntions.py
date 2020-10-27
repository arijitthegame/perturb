import numpy as np
import random


def make_datas(data,select):
    datas = {}
    for idx in range(len(select)):
        datas[idx] = data[select[idx]]
    return datas


def sample_once(datas, support_shot=5, query_shot=20, shuffle=True, plus = 0):
    np.random.seed(0)
    order = list(range(len(datas)))
    if shuffle:
        random.shuffle(order)

    target_support = np.repeat(order, support_shot)
    target_query = np.repeat(order, query_shot)

    select = random.sample(range(len(datas[order[0] + plus])), support_shot + query_shot)
    sample = datas[order[0]+plus][select, :]
    inputs_support = sample[:support_shot]
    inputs_query = sample[support_shot:]

    for idx in order[1:]:
        select = random.sample(range(len(datas[idx+plus])), support_shot + query_shot)
        sample = datas[idx + plus][select, :]
        inputs_support = np.append(inputs_support, sample[:support_shot], axis=0)
        inputs_query = np.append(inputs_query, sample[support_shot:], axis=0)

    return inputs_support, inputs_query, target_support, target_query



