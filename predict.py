from model import Graph
from train import read_data, extract_entity

train_data, dev_data, test_data, id2class, class2id = read_data()
_, model = Graph(0, 0, 0, 0)
model.load_weights('output/subject_model.weights')


def predict(content, cls):
    res = extract_entity(content, cls, class2id, model)
    return res


if __name__ == '__main__':
    while 1:
        content = input('content:')
        cls = input('cls:')
        r = predict(content, cls)
        print(r)
