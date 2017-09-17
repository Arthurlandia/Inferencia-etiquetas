# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import fnmatch

import numpy as np
import tensorflow as tf

from NodeLookup import NodeLookup
from pyspark import SparkContext, SparkConf
from pyspark.sql import Row
from pyspark.sql import SparkSession


FLAGS = None

# pylint: disable=line-too-long
#DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

# Numero de imagenes por batch
image_batch_size = 1000

# Numero de predicciones
num_top_predictions = 5


# Ruta salida del json en HDFS
output_path = 'hdfs://spark-master:9000/utad/output'


def run_inference_on_image_spark(sess, image_name, image_path, node_lookup):
    """Corre la inferencia sobre una imagen.

    Args:
        sess: la session de Tensorflow
        image_name: el nombre de la imagen
        image_path: la ruta de la imagen
        node_lookup: el nodelookup

    Returns:
        [(image_name, scores)]
    """

    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('Imagen no existe %s', image_path)

    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})

    predictions = np.squeeze(predictions)
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    scores = []
    for node_id in top_k:
        human_string = node_lookup[node_id]
        score = predictions[node_id]
        scores.append((human_string, score))
    print (image_name, scores)
    return image_name, scores



def process_batch(batch):
    """Aplica la inferencia a un batch de imagenes

    Args:
        batch: un lote de imagenes

    Returns:
        [Row(imagen=xxx, labels=xxx, score=xxx)]

    """
    with tf.Graph().as_default() as g:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_bc.value)
        tf.import_graph_def(graph_def, name='')
        with tf.Session() as sess:
            labeled = []
            labeled_format = []
            salida = []
            for (image_id, image_path) in batch:
                labeled.append((run_inference_on_image_spark(sess, image_id, image_path, node_lookup_bc.value)))
            for image, tup_scores in labeled:
                labeled_format.append((format_resul(image, tup_scores)))
            for tupla in labeled_format:
                for tup in tupla:
                    salida.append(tup)
            return salida



def format_resul(image, tup_scores):
    """Formatea la salida para formar un json

    Args:
        image: el nombre de una imagen
        tup_scores: [(labels, scores)]

    Returns:
        [Row(image, labels, score)]

    """
    return [Row(imagen=image, etiquetas=i[0], score=float(i[1])) for i in tup_scores]


def get_path_image(num):
    """Genera las rutas de las imagenes

    Args:
        num: numero maximo de imagenes

    Returns:
        [(image_id, path_image), (...)]
    """

    imagenes = []
    for i in range(1, num):
        imagenes.append((i, FLAGS.image_file+"im"+str(i)+".jpg"))
    return imagenes



def generate_batches(imagenes):
    """Separa la lista en lotes

    Args:
        imagenes: lista de todas las rutas de las imagenes

    Returns:
        Una lista con las rutas de todas las imagenes separadas por lotes
    """
    return [imagenes[i:i + image_batch_size] for i in range(0, len(imagenes), image_batch_size)]


def main(_):

    num = len(fnmatch.filter(os.listdir("/home/arthurlandia/Downloads/mirflickr25k/"), "*.jpg"))
    #num = 100
    print("Hay un total de %d imagenes" % num)

    # Primero se generan los lotes
    batched_data = generate_batches(get_path_image(num))
    print("Hay %d batches" % len(batched_data))

    # Se paralelizan los lotes y se procesan
    fotosRDD = sc.parallelize(batched_data)
    imagenesRDD = fotosRDD.flatMap(process_batch)

    # La salida se guarda en formato json dentro del sistema de archivos HDFS
    # Para ello se convierte el rdd en dataframe
    spark = SparkSession(sc)
    imagenes_df = imagenesRDD.toDF()
    imagenes_df.write.json(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classify_image_graph_def.pb:
    #   Binary representation of the GraphDef protocol buffer.
    # imagenet_synset_to_human_label_map.txt:
    #   Map from synset ID to a human readable string.
    # imagenet_2012_challenge_label_map_proto.pbtxt:
    #   Text representation of a protocol buffer mapping a label to synset ID.
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/home/arthurlandia/Documents/imagenet',
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='/home/arthurlandia/Downloads/mirflickr25k/',
        help='Absolute path to image file.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    conf = SparkConf().setAppName("Image classify").setMaster("spark://192.168.1.9:7077")
    sc = SparkContext(conf=conf)

    # Se carga el NodeLookup y el modelo
    node_lookup_bc = sc.broadcast(NodeLookup().node_lookup)
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        model_bc = sc.broadcast(f.read())

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
