

from pyspark import SparkContext, SparkConf
from HTMLParser import HTMLParser
from pyspark.sql import SparkSession
from pyspark.sql import Row

# beatuifulsoup
classify_path = 'hdfs://spark-master:9000/utad/classify/output'
conf = SparkConf().setAppName("Prepare tags").setMaster("spark://192.168.1.9:7077")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

path_hdfs = "hdfs://spark-master:9000/utad/"
path_tags = "/home/arthurlandia/Downloads/mirflickr25k/meta/tags_raw/"
common_tags = "/home/arthurlandia/Downloads/mirflickr25k/doc/common_tags.txt"

def tags_mirflickr_to_json():
    """ Convierte todos los ficheros con las tags de los usuarios de mirflickr a un JSON
    """
    sc.wholeTextFiles(path_tags)\
        .flatMap(lambda (ruta, content): load_tags(ruta, content))\
        .toDF()\
        .write.mode("overwrite")\
        .json(path_hdfs+"mirflickr/tags")


def load_tags(path, content):
    """ Coge todas las etiquetas de un archivo (que hace referencia a una imagen) y lo transforma
        en un Row de tuplas

    Args:
        path: es la ruta del archivo
        content: es el contenido del archivo
    Return:
        [Row(image, etiqueta)]
    """

    paths = path.split("/")
    num = paths[7].replace("tags", "")
    num = num.replace(".txt", "")

    htmlparser = HTMLParser()
    contenido = htmlparser.unescape(content).lower().split("\r\n")
    resul = []
    for tag in contenido:
        if tag != "":
            resul.append((Row(imagen=num, etiqueta=tag)))
    return resul


def tags_mirflickr_commons():
    """ Transfofrma el archivo common_tags.txt en un .csv y lo guarda en HDFS

    """

    sc.textFile(common_tags)\
        .flatMap(lambda line: Row(line.split(" ")))\
        .toDF(["label", "number"]) \
        .select("label") \
        .limit(50) \
        .coalesce(4) \
        .write.mode("overwrite")\
        .text(path_hdfs+"mirflickr/commons")



def tags_imagenet_commons():
    """ Coge los tags comunes de la clasificacion realizada por tensorflow y lo guarda en .csv
        en HDFS

    """

    spark.read.json(path_hdfs+"classify/output") \
        .rdd \
        .flatMap(lambda line: split_tags(line[0])) \
        .reduceByKey(lambda a, b: a+b) \
        .sortBy(lambda (k, v): v, False) \
        .flatMap(lambda val: Row(val)) \
        .toDF(["label", "number"]) \
        .select("label") \
        .orderBy("number", ascending=False) \
        .limit(50) \
        .coalesce(4) \
        .write.mode("overwrite")\
        .text(path_hdfs+"image_net/commons")


def split_tags(line):
    """ Se procesa cada linea, por si hubiera varias palabras en cada linea, tambien se
        pasan todas las palabras a minusculas y se eliminan los espacios

    Args
        line: el valor correspondiente al nodo etiquetas

    Returns:
        [(etiqueta, 1)]
    """

    output = []
    for tag in line.split(","):
        output.append((tag.strip().lower(), 1))
    return output


if __name__ == '__main__':
    tags_mirflickr_commons()
    tags_imagenet_commons()
    tags_mirflickr_to_json()
