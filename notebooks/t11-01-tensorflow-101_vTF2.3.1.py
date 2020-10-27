
#APRENDIZAJE NEURONAL DE LAS SEÑALES DE TRÁFICO

#Desarrollado con Tensorflow v2.3.1
#IMPORTANTE: Pendiente de solucionar para adaptar a la versión 2.3.1


#IMPORTACIÓN DE LIBRERÍAS
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray
import tensorflow as tf
import random



#CARGA DE DATASET
def load_ml_data(data_directory):

    dirs = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory,d))]

    labels = []
    images = []

    for d in dirs:
        label_dir = os.path.join(data_directory, d).replace("\\", "/")
        file_names = [os.path.join(label_dir, f).replace("\\", "/") for f in os.listdir(label_dir) if f.endswith('.ppm')]

        for f in file_names:
            images.append(imageio.imread(f))
            labels.append(int(d))
    
    return images, labels

main_directory = 'C:/Users/usuario/Downloads/python-ml-course-master/datasets/belgian/'

train_data_directory = os.path.join(main_directory, 'Training').replace("\\", "/")
test_data_directory = os.path.join(main_directory, 'Testing').replace("\\", "/")

train_images, train_labels = load_ml_data(train_data_directory)
test_images, test_labels = load_ml_data(test_data_directory)

train_images = np.array(train_images, dtype=object)
train_labels = np.array(train_labels, dtype=int)

test_images = np.array(test_images, dtype=object)
test_labels = np.array(test_labels, dtype=int)


#TRANSFORMACIÓN DE IMÁGENES A 30x30
train_images_30 = [transform.resize(image, (30, 30)) for image in train_images]
test_images_30 = [transform.resize(image, (30, 30)) for image in test_images]


#TRANSFORMACIÓN DE IMÁGENES A ESCALA DE GRISES
train_images_30 = np.array(train_images_30)
train_images_30 = rgb2gray(train_images_30)
test_images_30 = np.array(test_images_30)
test_images_30 = rgb2gray(test_images_30)


#PREPARACIÓN DEL MODELO
x = tf.placeholder(dtype = tf.float32, shape = [None, 30, 30])
y = tf.placeholder(dtype = tf.int32, shape = [None])
images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
train_opt = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
final_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(final_pred, tf.float32))


#ENTRENAMIENTO DEL MODELO
tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(601):
    _, accuracy_val = sess.run([train_opt, accuracy], feed_dict = {x: train_images_30, y:list(train_labels)})
    _, loss_val = sess.run([train_opt, loss], feed_dict = {x: train_images_30, y:list(train_labels)})
    if i % 50 == 0:
        print('EPOCH', i)
        print('Eficacia:', accuracy_val)
        print('Loss:', loss)
    

#EVALUACIÓN DE LA RED NEURONAL
sample_idx = random.sample(range(len(train_images_30)), 40)
sample_images = [train_images_30[i] for i in sample_idx]
sample_labels = [train_labels[i] for i in sample_idx]
prediction = sess.run([final_pred], feed_dict = {x: sample_images})[0]
plt.figure(figsize = (16, 20))
for i in rang(len(sample_images)):
    truth = sample_labels[i]
    predi = prediction[i]
    plt.subplot(10,4, i + 1)
    plt.axis('off')
    color = 'green' if truth == predi else 'red'
    plt.text(32, 15, 'Real:          {0}\nPredicción: {1}'.format(truth, predi), fontsize = 14, color = color)
    plt.imshow(sample_images[i], cmap = 'gray')
plt.show()
prediction = sess.run([final_pred], feed_dict = {x: test_images_30})[0]
match_count = sum([int(l0 == lp) for l0, lp in zip(test_labels, prediction)])
acc = match_count / len(test_labels) * 100
print('Eficacia de la red neuronal: {:.3f}'.format(acc))