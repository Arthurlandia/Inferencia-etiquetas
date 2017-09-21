import itertools

from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import *

# beatuifulsoup
conf = SparkConf().setAppName("FlickrToImagenet").setMaster("spark://192.168.1.9:7077")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
path_hdfs = "hdfs://spark-master:9000/utad/"

def entrenar(word):
    """ Se entrena un modelo para una etiqueta dada.

    Args
        word: la etiqueta dada

    Returns:
         (etiqueta, modelo)
    """

    datos = trainingRDD \
        .map(lambda (image, tags, features): Row(int(1 if word in tags else 0), features))\
        .toDF(["label", "features"])

    dt = DecisionTreeClassifier(
        labelCol="label",
        featuresCol="features",
        maxDepth=4)

    model = dt.fit(datos)
    return word, model


def predecir(modelos):
    """ Realiza las prediciones. Devuelve una lista de tuplas que contienen la etiqueta a predecir
        junto con la imagen que el modelo ha predicho.

    Args:
        modelos: son los modelos obtenidos por cada etiqueta -> (etiqueta, modelo)

    Returns:
        [Row(image, etiqueta)]
    """
    predicciones = []
    for modelo in modelos:
        model = modelo[1].transform(testData)
        image_predict = model.select("image").filter(model.prediction == 1).collect()
        for imagen in image_predict:
            predicciones.append(Row(imagen.image, modelo[0]))
    return predicciones


if __name__ == '__main__':

    # Se transforma el json de la clasificacion realizada por imagenet
    # {"etiquetas":"sunglass","imagen":1,"score":0.5427044034004211}
    # a una tupla (imagen, list(etiquetas))
    tags_imagenetRDD = spark.read.json(path_hdfs+"classify/output") \
        .rdd \
        .map(lambda line: (line[1], [word.strip().lower() for word in line[0].split(",")])) \
        .groupByKey() \
        .map(lambda x: (x[0], list(itertools.chain.from_iterable(x[1]))))


    # Se transforma el json generado con todas las etiquetas de mirflickr
    # {"etiqueta":"balloon","imagen":"11243"}
    # a una tupla (imagen, list(etiquetas))
    tags_mirflickrRDD = spark.read.json(path_hdfs+"mirflickr/tags") \
        .rdd \
        .map(lambda line: (line[1], line[0])) \
        .groupByKey() \
        .map(lambda x: (int(x[0]), list(x[1])))


    # Se hace el join para juntar tanto las clasificaiones de imagenet como las etiquetas de flickr
    # (imagen, list(etiquetas)).join((imagen, list(etiquetas))
    # a (imagen, labels, etiquetas)
    # y se convierte a dataframe para generar las caracteristicas
    tags_join = tags_imagenetRDD\
        .join(tags_mirflickrRDD)\
        .map(lambda (k, v): (k, v[0], v[1]))\
        .toDF(["image", "tags", "labels"])


    # Se generan las caracteristicas, para ello se usa CountVectorizer, que genera un label con sus features,
    # estas son un SparseVector de las 20000 mas comunes
    tags_vector = CountVectorizer(
        inputCol="labels", outputCol="features", vocabSize=20000)\
        .fit(tags_join)\
        .transform(tags_join)


    # Se generan los datasets de entrenamiento y de test
    trainingData, testData = tags_vector.select("image", "tags", "features").randomSplit([0.8, 0.2])
    trainingRDD = trainingData.rdd.cache()
    testRDD = testData.rdd.cache()

    # Se cogen las etiquetas mas comunes de la clasificacion de imagenet
    comunes = spark.sparkContext.textFile(path_hdfs+"image_net/commons").collect()

    # Se generan los modelos por cada etiqueta comun
    modelos = [entrenar(word) for word in comunes]
    predicciones = predecir(modelos)

    # Se convierte el resultado a un RDD
    prediccionesRDD = sc.parallelize(predicciones)

    # Luego se convierten en dataframe para guardarlo en HDFS
    prediccionesRDD\
        .map(lambda (image, label): Row(imagen=image, etiqueta=label))\
        .toDF()\
        .coalesce(4)\
        .write.mode("overwrite")\
        .json(path_hdfs + "image_net/predictions")
