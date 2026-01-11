import dataset
import matplotlib.pyplot as plt

train, test, eval = dataset.create_datasets([0, 1, 2, 3, 4], 1000, 100, 10)

for data in eval.dataset:
    plt.imshow(data)
    plt.show()
