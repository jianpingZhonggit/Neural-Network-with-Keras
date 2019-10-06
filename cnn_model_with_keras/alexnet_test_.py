from cnn_model_with_keras.alexnet_model import get_alex_model
from cnn_model_with_keras.data import ImageGenerator


def train(generator, model_, epochs):
    for epoch in range(epochs):
        for _ in range(len(generator.img_list)//generator.batch):
            gen = generator.get_batch()
            # print(train_x)
            # break
            # model_.train_on_batch(train_x, train_y)
            model.fit_generator(gen, steps_per_epoch=100, epochs=2)


if __name__ == "__main__":
    model = get_alex_model('./alex_net.h5', (224, 224, 3))
    img_generator = ImageGenerator('../dataset/train/', 224, 224, 32, 5)
    train(img_generator, model, 1)

