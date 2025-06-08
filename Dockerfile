FROM apache/spark:3.5.6-python3

USER root

# Create a folder for the Hadoop AWS libraries
RUN mkdir -p /opt/hadoop/lib

# Download Hadoop AWS and AWS SDK jars (compatible with Hadoop 3.3.5 used in Spark 3.5.x)
ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.5/hadoop-aws-3.3.5.jar /opt/hadoop/lib/
ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.5/hadoop-common-3.3.5.jar /opt/hadoop/lib/
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.517/aws-java-sdk-bundle-1.12.517.jar /opt/hadoop/lib/
ADD https://repo1.maven.org/maven2/com/fasterxml/woodstox/woodstox-core/6.2.4/woodstox-core-6.2.4.jar /opt/hadoop/lib/
ADD https://repo1.maven.org/maven2/org/codehaus/woodstox/stax2-api/4.2.1/stax2-api-4.2.1.jar /opt/hadoop/lib/

# Apache Commons Configuration (required by Hadoop metrics)
ADD https://repo1.maven.org/maven2/org/apache/commons/commons-configuration2/2.8.0/commons-configuration2-2.8.0.jar /opt/hadoop/lib/
ADD https://repo1.maven.org/maven2/org/apache/commons/commons-text/1.9/commons-text-1.9.jar /opt/hadoop/lib/

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Optional: fix permissions
RUN chmod -R 755 /opt/hadoop/lib

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')"

# Copy Spark script and requirements
COPY create_dataset.py /app/create_dataset.py
