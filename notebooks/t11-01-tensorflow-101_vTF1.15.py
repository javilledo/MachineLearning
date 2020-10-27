
#Aprendizaje neuronal de las señales de tráfico

#Desarrollado con Tensorflow v1.15
import os

def load_ml_data(data_directory):

    import os
    import imageio

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

print()
print('  CARGA DE IMÁGENES')
print('---------------------')
print('Se han cargado', len(train_images), 'imágenes en el conjunto de entrenamiento')
print('Se han cargado', len(test_images), 'imágenes en el conjunto de testing')
print()

# El tipo de las variables con los datos es tipo 'list'

import numpy as np

train_images = np.array(train_images, dtype=object)
train_labels = np.array(train_labels, dtype=int)

test_images = np.array(test_images, dtype=object)
test_labels = np.array(test_labels, dtype=int)

# El tipo de las variables con los datos es tipo 'array'

print('  COMPROBACIÓN DE DATOS')
print('-------------------------')
print('El archivo train_images tiene', train_images.size, 'datos')
print('El archivo train_labels tiene', train_labels.size, 'datos')
print('El número de señales diferentes en el conjunto de entrenamiento es de', len(set(train_labels)))
print('El archivo test_images tiene', test_images.size, 'datos')
print('El archivo test_labels tiene', test_labels.size, 'datos')
print('El número de señales diferentes en el conjunto de testing es de', len(set(test_labels)))
print()

print('Como ejemplo el tamaño de la primera fotografía es', train_images[0].shape)
print()

#Puede saberse información sobre los datos cargados con .flags, el tamaño de las imágenes con .itemsize y .nbytes

import matplotlib.pyplot as plt

#HISTOGRAMA DE CANTIDAD DE DATOS EN CADA CATEGORÍA
# plt.hist(train_labels, len(set(train_labels)))
# plt.show()

#Aquí puede verse que hay categorías con muy pocas fotos, y otras con hasta 300 fotos

#VISUALIZACIÓN DE IMÁGENES AL AZAR
import random

rand_signs = random.sample(range(0, len(train_labels)), 6)
print('Las señales aleatorias que vamos a pintar son', rand_signs)
print()

# for i in range(len(rand_signs)):
#     temp_im = train_images[rand_signs[i]]
#     plt.subplot(1, 6, i+1)
#     plt.axis('off')
#     plt.imshow(temp_im)
#     plt.subplots_adjust(wspace = 0.5)
#     print('Forma: {0}, min: {1}, max:{2}'.format(temp_im.shape, temp_im.min(), temp_im.max()))
# plt.show()

#VISUALIZACIÓN DE UNA IMAGEN POR CATEGORÍA
# unique_labels = set(train_labels) #Obtiene las uniques
# plt.figure(figsize = (16, 16))
# i = 1
# for label in unique_labels:
#     temp_im = train_images[list(train_labels).index(label)]
#     plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
#     plt.subplot(8, 8, i)
#     plt.axis('off')
#     plt.title('Clase {0} ({1})'.format(label, list(train_labels).count(label)))
#     i = i + 1
#     plt.imshow(temp_im)
# plt.show()

#MODELO DE RED NEURONAL CON TENSORFLOW
#  + No todas las imágenes son del mismo tamaño
#  + Hay 62 categorías de imágenes (desde la 0 hasta la 61)
#  + La distribución de señales de tráfico no es uniforme (algunas salen más veces que otras)

#Convertimos imágenes a escala de grises, ya que usar el color es poco útil, porque el nivel de iluminación no es estable
from skimage import transform

#Queremos saber el tamaño mínimo de anchura y altura, para no hacer unas imágenes por debajo de esos valores
w = 9999 
h = 9999
for image in train_images:
    if image.shape[0] < h:
        h = image.shape[0] 
    if image.shape[1] < w:
        w = image.shape[1]
print('Tamaño mínimo: {0}x{1}'.format(h, w))  

train_images_30 = [transform.resize(image, (30, 30)) for image in train_images]
test_images_30 = [transform.resize(image, (30, 30)) for image in test_images]

print('Como ejemplo se muestra la primera de las imágenes')
print(train_images_30[0])

#VISUALIZACIÓN DE UNA IMAGEN POR CATEGORÍA TRAS EL REDIMENSIONADO
# unique_labels = set(train_labels) #Obtiene las uniques
# plt.figure(figsize = (16, 16))
# i = 1
# for label in unique_labels:
#     temp_im = train_images_30[list(train_labels).index(label)]
#     plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
#     plt.subplot(8, 8, i)
#     plt.axis('off')
#     plt.title('Clase {0} ({1})'.format(label, list(train_labels).count(label)))
#     i = i + 1
#     plt.imshow(temp_im)
# plt.show()

#Las imágenes ya se encuentra normalizadas, entre 0 y 1

# A continuación se debe transformar a escala de grises
from skimage.color import rgb2gray

train_images_30 = np.array(train_images_30)
train_images_30 = rgb2gray(train_images_30)
test_images_30 = np.array(test_images_30)
test_images_30 = rgb2gray(test_images_30)

#VISUALIZACIÓN DE UNA IMAGEN POR CATEGORÍA TRAS EL REDIMENSIONADO Y PASO A GRAYSCALE
# unique_labels = set(train_labels) #Obtiene las uniques
# plt.figure(figsize = (16, 16))
# i = 1
# for label in unique_labels:
#     temp_im = train_images_30[list(train_labels).index(label)]
#     plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
#     plt.subplot(8, 8, i)
#     plt.axis('off')
#     plt.title('Clase {0} ({1})'.format(label, list(train_labels).count(label)))
#     i = i + 1
#     plt.imshow(temp_im, cmap = 'gray') #Ojo que aquí se le indica que debe estar en escala de grises
# plt.show()

#PREPARACIÓN DEL MODELO
import tensorflow as tf

x = tf.placeholder(dtype = tf.float32, shape = [None, 30, 30])
y = tf.placeholder(dtype = tf.int32, shape = [None])

images_flat = tf.contrib.layers.flatten(x)

logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

train_opt = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

final_pred = tf.argmax(logits, 1)

accuracy = tf.reduce_mean(tf.cast(final_pred, tf.float32))

print(images_flat)
print(logits)
print(loss)
print(final_pred)

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
    
#Evaluación de la red neuronal
sample_idx = random.sample(range(len(train_images_30)), 40)
sample_images = [train_images_30[i] for i in sample_idx]
sample_labels = [train_labels[i] for i in sample_idx]

prediction = sess.run([final_pred], feed_dict = {x: sample_images})[0]

print(sample_labels)
print(prediction)

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