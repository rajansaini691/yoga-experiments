"""
Steps:
    - Get pretrained MobileNet
    - Train with pruning on MPII
    - Convert to tflite
"""
import tensorflow as tf
import tensorflow_model_optimization as tfmot
print('finish imports')

# Read dataset
# for context, OV5640 outputs this: (2592, 1944)

img_height, img_width = (224, 224)

batch_size = 2
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './Yoga-82/data/train/',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

print('load part of dataset')

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './Yoga-82/data/train/',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

num_classes = 8

"""
Construct model
"""
inputs = tf.keras.layers.Input(shape=(img_width, img_height, 3))


# Prune mobilenet
mobilenet = tf.keras.applications.MobileNet(weights='imagenet', include_top=False)

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0, final_sparsity=0.8, begin_step=0, end_step=600)

pruning_policy = tfmot.sparsity.keras.PruneForLatencyOnXNNPack()

pruned_mobilenet = tfmot.sparsity.keras.prune_low_magnitude(
        mobilenet,
        #pruning_policy=pruning_policy           <--- Uncomment when XNN bug fixed
        pruning_schedule=pruning_schedule)

# Convert mobilenet logits into predictions
avg = tf.keras.layers.GlobalAveragePooling2D()(pruned_mobilenet(inputs))
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(avg)

# Create model
model = tf.keras.Model(inputs, outputs)

# TODO Modify params


train_model = model         # Set to model_for_pruning

train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'], loss='sparse_categorical_crossentropy')

print('start training')

# Train
train_model.fit(train_ds, epochs=11,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries('./out')],
        validation_data=test_ds)

# Try inference
