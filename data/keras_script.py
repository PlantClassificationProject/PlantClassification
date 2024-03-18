from tensorflow import keras

from generate_model import make_model
from train_model import train_model

image_size = (180, 180)
batch_size = 128

# Load training data
train_ds = keras.preprocessing.image_dataset_from_directory(
    "split_ttv_dataset_type_of_plants/Train_Set_Folder",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=(180, 180),
    batch_size=32,
)

# Load validation data
val_ds = keras.preprocessing.image_dataset_from_directory(
    "split_ttv_dataset_type_of_plants/Validation_Set_Folder",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(180, 180),
    batch_size=32,
)

# # Can remove this eventually, for now just if people are interested
# # Visualize a subset of the training data
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(int(labels[i]))
#         plt.axis("off")
# plt.show()

model = make_model(input_shape=image_size + (3,), num_classes=2)

train_model(model, image_size, train_ds, val_ds)