import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

IMAGE_SIZE = [224, 224]
#random shuffle
train_path = r"data_3categories/train"
valid_path = r"data_3categories/test"

resnet = ResNet152(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in resnet.layers:
    layer.trainable_model = False

folders = glob("data_3categories/train/*")
x = tf.keras.layers.BatchNormalization()(resnet.output)
x = Flatten()(x)
prediction = Dense(len(folders), activation='softmax', kernel_initializer='he_normal')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
  loss= tf.keras.losses.CategoricalCrossentropy(),
  optimizer= opt,
  metrics=['accuracy']
)
model.summary()

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data_3categories/train',
                                                 target_size = (224, 224),
                                                 shuffle = True,
                                                 batch_size = 8,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('data_3categories/test',
                                            target_size = (224, 224),
                                            shuffle = False,
                                            class_mode = 'categorical')

r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=200,
)

#accuracy report
y_pred = model.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
fnames = test_set.filenames ## fnames is all the filenames/samples used in testing
errors = np.where(y_pred != test_set.classes)[0] ## misclassifications done on the test data where y_pred is the predicted values
#for i in errors:
#    print(fnames[i])

#confusion matrix
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())
print(class_labels)
print(confusion_matrix(test_set.classes, y_pred))
report = classification_report(true_classes, y_pred, target_names=class_labels)
print(report)

# plot the loss
plt.figure(1)
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('try2_loss.png')


# plot the accuracy
plt.figure(2)
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.yticks(np.arange(0,1,0.1))
plt.savefig('try2_accuracy.png')


model.save('model_try2.h5')

# make a prediction
# set plot figure size
fig, c_ax = plt.subplots(1,1, figsize = (12, 8))

# function for scoring roc auc score for multi-class
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(class_labels):
        fpr, tpr, thresholds = metrics.roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, metrics.auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    return metrics.roc_auc_score(y_test, y_pred, average=average)

print('ROC AUC score:', multiclass_roc_auc_score(test_set.classes, y_pred))

c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
plt.savefig("try2_roc.png")