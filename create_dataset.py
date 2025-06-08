# Imports necessary libraries
from nltk import corpus
from nltk.tokenize import word_tokenize
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import udf, row_number
from pyspark.sql.types import StringType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import logging
import nltk
import numpy as np
import os
import sys
import traceback


# Ensure NLTK data path is set
nltk.data.path.append('/usr/local/share/nltk_data')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    datefmt='%d/%m/%y %H:%M:%S',
)
logger = logging.getLogger(__name__)


####################################################################################################
# Helper functions
####################################################################################################

def create_spark_session() -> SparkSession:
    """Create and configure a Spark session with S3 support."""
    logger.info("Creating Spark session")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    assert aws_access_key is not None, "AWS_ACCESS_KEY_ID is not set"
    assert aws_secret_key is not None, "AWS_SECRET_ACCESS_KEY is not set"
    try:
        spark = SparkSession.builder \
            .appName("RecipesReader") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
            ) \
            .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
            .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
            .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
            .config("spark.driver.extraClassPath", "/opt/hadoop/lib/*") \
            .config("spark.executor.extraClassPath", "/opt/hadoop/lib/*") \
            .getOrCreate()
        logger.info("Spark session created successfully")
        spark.sparkContext.setLogLevel("INFO")
        logger.info("Spark context log level set to INFO")
        return spark
    except Exception as e:
        logger.info(f"Failed to create Spark session: {str(e)}")
        logger.info(traceback.format_exc())
        sys.exit(1)


# Creates a set of stopwords
stopwords = set(corpus.stopwords.words('english'))
# Adds units of measurement, quantities, and very common ingredients to stopwords
stopwords.update([
    "1/2", "1/4", "3/4", "tbsp", "lbs", "cup", "tsp", "tablespoon", "teaspoon",
    "teaspoons", "tablespoons", "cups", "pounds", "ounces", "chopped", "salt",
    "fresh", "sliced", "large", "ounce", "pkg", "optional", "tbsp", "pound",
    "small", "medium", "taste", "1/3", "1/8", "tbs", "minced", "cut", "finely",
    "whole", "melted", "freshly", "pieces", "thinly", "recipe", "divided", "inch",
    "cubes", "half", "grated", "diced", "shredded", "peeled", "juice", "drained",
    "cooked", "slices", "cans", "package", "pinch", "plus", "crushed", "mix",
    "stick", "powder", "dash", "box", "use", "coarsely", "lightly", "possibly",
    "slightly", "roughly", "firmly", "preferably", "sugar", "pepper", "cheese",
    "butter", "oil", "cream"
])


def tokenizeWords(text: str) -> List[str]:
    """
    Tokenizes words in a text and returns a list of words.
    params:
        text: str
    returns:
        List[str]: list of words
    """
    return [word for word in word_tokenize(text.lower()) if (
        (word not in stopwords) and (not word.isnumeric()) and (len(word) > 2)
    )]


def clean_ingredients(text: str) -> str:
    """
    Cleans ingredients text by removing double quotes, commas, square brackets, and extra whitespace
    params:
        text: str
    returns:
        str: cleaned text
    """
    # Remove any double quotes that might be inside the string
    return text.replace('"', "").replace("'", "").replace(",", "").replace("[", "")\
        .replace("]", "").strip()


####################################################################################################
# Main script
####################################################################################################

if __name__ == "__main__":

    try:
        # Create Spark session
        spark = create_spark_session()

        # Creates dataframe from recipes data
        logger.info("Reading recipes data")
        df = spark.read.option("header", "true").csv(
            "s3a://recipes-recommendations/recipes_data.csv"
        )
        logger.info("Recipes data read")

        # Drops unnecessary columns
        logger.info("Dropping unnecessary columns")
        df = df.drop("source", "site", "NER", "directions")
        logger.info("Unnecessary columns dropped")

        # Drops rows with missing values
        logger.info("Dropping rows with missing values")
        df = df.dropna()
        logger.info("Rows with missing values dropped")

        # Splits the data into 10 random subsets
        logger.info("Splitting data into 10 random subsets")
        subsets = df.randomSplit([0.1] * 10)
        logger.info("Data split into 10 random subsets")

        for index, df in enumerate(subsets):

            # Write dataframe to S3
            logger.info(f"Writing subset {index} to S3")
            df.write.mode("overwrite").parquet(
                f"s3a://recipes-recommendations/subsets/recipes_data_subset_{index}.parquet"
            )
            logger.info("Dataframe written to S3")

        for index, df in enumerate(subsets):
            logger.info(f"Processing subset {index}")

            # Adds an index column
            logger.info("Adding index column")
            df = df.withColumn("index", row_number().over(Window.orderBy("link")) - 1)
            logger.info("Index column added")

            # Sorts the data by index
            logger.info("Sorting data by index")
            df = df.orderBy("index")
            logger.info("Data sorted by index")

            # Converts ingredients to list
            logger.info("Cleaning ingredients")
            df = df.withColumn(
                "ingredients",
                udf(lambda x: clean_ingredients(x))(df["ingredients"])
            )
            logger.info("Cleaned ingredients")

            # Creates columns with ingredients and titles as lists of strings
            logger.info("Creating title_ingredients_combined column")
            tokenize_title_and_ingredients = udf(
                lambda x, y: " ".join(tokenizeWords(x)) + " " + " ".join(tokenizeWords(y))
            )
            df = df.withColumn(
                "title_ingredients_combined",
                tokenize_title_and_ingredients(df["ingredients"], df["title"])
            )
            logger.info("title_ingredients_combined column created")

            # Collects the links
            logger.info("Collecting links")
            links = [row.link for row in df.select("link").collect()]
            logger.info("Links collected")

            # Collect the data into a Python list first, preserving order
            logger.info("Collecting title_ingredients_combined column")
            title_ingredients_list = (
                df.select("title_ingredients_combined")
                .orderBy("index")  # Ensure consistent ordering
                .rdd
                .map(lambda row: row.title_ingredients_combined)
                .collect()
            )
            logger.info("title_ingredients_combined column collected")

            # Creates a TF-IDF vectorizer
            logger.info("Creating TF-IDF vectorizer")
            vectorizer = TfidfVectorizer().fit(title_ingredients_list)
            logger.info("TF-IDF vectorizer created")

            # Creates a TF-IDF matrix
            logger.info("Creating TF-IDF matrix")
            vectors = vectorizer.transform(title_ingredients_list)
            logger.info("TF-IDF matrix created")

            # Creates a function to get the most and least similar links for a given row
            def get_links(index: int) -> str:
                """
                Gets the most and least similar links for a given row.
                params:
                    index: int
                returns:
                    str: list of similar and different links
                """
                cosineSimilarities = cosine_similarity(vectors[index], vectors)[0]
                topIndices = np.argpartition(cosineSimilarities, -4)[-4:]
                topIndices = topIndices[topIndices != index]
                bottomIndices = np.argpartition(cosineSimilarities, 3)[:3]
                return str([topIndices, bottomIndices])

            # Creates a column with the cosine similarity between the vector and each row
            logger.info("Creating similar and different links column")
            df = df.withColumn(
                "similar_and_different_links",
                udf(get_links, StringType())(df["index"])
            )
            logger.info("similar and different links columns created")

            # Gets list of similar and different links
            logger.info("Getting list of similar and different links")
            similar_and_different_links = (
                df.select("similar_and_different_links")
                .rdd
                .map(lambda x: x.similar_and_different_links)
                .collect()
            )
            logger.info("List of similar and different links collected")
            logger.info(f"similar_and_different_links: {similar_and_different_links[:10]}")
            logger.info(f"similar_and_different_links: {len(similar_and_different_links)}")
            logger.info(f"similar_and_different_links: {type(similar_and_different_links[0])}")

            # Saves the data to S3
            logger.info("Saving data to S3")
            df.write.mode("overwrite").parquet(
                f"s3a://recipes-recommendations/results/recipes_data_with_similar_and_different_links_{index}.parquet"  # noqa: E501
            )
            logger.info("Data saved to S3")

    except Exception as e:
        logger.info(f"Error in main function: {str(e)}")
        logger.info(traceback.format_exc())
    finally:
        logger.info("Stopping Spark session")
        try:
            spark.stop()
            logger.info("Spark session stopped")
        except Exception as e:
            logger.info(f"Error stopping Spark session: {str(e)}")
            logger.info(traceback.format_exc())
            sys.exit(1)  # Force exit on error
