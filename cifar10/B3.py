import keras
from keras.datasets import mnist
from umlaut import UmlautCallback
import tensorflow as tf
#from umlaut import UmlautCallback

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()


train_images = train_images / 255.0
test_images = test_images / 255.0


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3), dtype=tf.float32),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])



model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

cb = UmlautCallback(
    model,
    session_name='eb',
    offline=True,
)

# train the model
model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=128,
    callbacks=[cb],
    #validation_split=0.2,
    validation_data=(train_images[:100], train_labels[:100])
)

results = model.evaluate(test_images, test_labels, batch_size=4096)
print('test loss, test acc: ', results)