# Recipe Recommendations Dataset Creation

This project contains a PySpark script that processes a large dataset of recipes to generate similarity scores between recipes based on their titles and ingredients. The script uses TF-IDF vectorization and cosine similarity to identify similar and different recipes, which can be used for training vector embeddings models for dense vector search.

## Features

- Processes recipes data using PySpark for distributed computing
- Implements TF-IDF vectorization for text analysis
- Calculates cosine similarity between recipes
- Identifies top 3 similar and 3 different recipes for each recipe
- Handles data in chunks for efficient processing
- Saves results in Parquet format for efficient storage and querying

## Prerequisites

- Python 3.8+
- PySpark 3.5.0
- Access to an AWS EKS cluster
- AWS credentials configured with access to the S3 bucket
- Docker (for containerized deployment)
- NLTK data (punkt and stopwords)

## AWS Setup

1. Create an S3 bucket:
```bash
aws s3 mb s3://your-bucket-name --region your-region
```

2. Set up bucket permissions:
```bash
aws s3api put-bucket-versioning --bucket your-bucket-name --versioning-configuration Status=Enabled
```

3. Create an IAM policy for S3 access:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name",
                "arn:aws:s3:::your-bucket-name/*"
            ]
        }
    ]
}
```

4. Attach the policy to your IAM user or role.

## Dataset Preparation

1. Download the RecipeNLP dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/recipenlg)

2. Process the dataset to match the required format:
   - The CSV file should have the following columns:
     - `title`: Recipe title
     - `ingredients`: List of ingredients
     - `link`: Unique identifier for the recipe
     - `source`: Source of the recipe
     - `site`: Website where the recipe was found
     - `NER`: Named entity recognition tags
     - `directions`: Cooking instructions

3. Upload the processed dataset to your S3 bucket:
```bash
aws s3 cp recipes_data.csv s3://your-bucket-name/recipes_data.csv
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up AWS credentials:
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

3. Update the S3 bucket name in the script:
   - Open `create_dataset.py`
   - Replace `recipes-recommendations` with your bucket name in the S3 paths

## Script Overview

The script performs the following steps:

1. **Data Loading and Preprocessing**
   - Reads recipes data from S3 (`s3://your-bucket-name/recipes_data.csv`)
   - Drops unnecessary columns
   - Removes rows with missing values
   - Splits data into 10 random subsets for parallel processing

2. **Text Processing**
   - Cleans ingredient lists
   - Tokenizes titles and ingredients
   - Removes stopwords and common cooking terms
   - Combines title and ingredients for analysis

3. **Similarity Analysis**
   - Creates TF-IDF vectors for each recipe
   - Calculates cosine similarity between recipes
   - Identifies top 3 similar and 3 different recipes for each recipe

4. **Data Storage**
   - Saves processed data in Parquet format
   - Stores results in S3 for further use

## Docker Build and Run

1. Build the Docker image:
```bash
docker build -t recipes-processor .
```

2. Run the container locally:
```bash
docker run \
    -e AWS_ACCESS_KEY_ID="your-access-key" \
    -e AWS_SECRET_ACCESS_KEY="your-secret-key" \
    -v ~/.aws:/root/.aws:ro \
    recipes-processor
```

## Kubernetes Deployment

The script can be deployed to Kubernetes using Argo Workflows. The workflow is defined in `recipes-spark.yaml`.

1. Create the AWS credentials secret:
```bash
kubectl create secret generic aws-credentials \
    --from-literal=aws-access-key-id="your-access-key" \
    --from-literal=aws-secret-access-key="your-secret-key" \
    -n argo
```

2. Set up RBAC permissions:
```bash
kubectl apply -f spark-rbac.yaml
```

This will create:
- A ClusterRole (`spark-role`) that grants permissions for:
  - Managing pods, services, and configmaps
  - Managing deployments
  - Managing Spark applications and scheduled spark applications
- A ClusterRoleBinding that binds these permissions to the default service account in the argo namespace

3. Apply the Argo workflow:
```bash
kubectl apply -f recipes-spark.yaml
```

The workflow will:
- Delete any existing Spark job
- Create a new Spark job with the following configuration:
  - 5 executors with 32GB memory each
  - 4 cores per executor
  - AWS credentials mounted from secrets
  - S3 access configured
  - Hadoop AWS libraries included

## Output

The script generates the following outputs in your S3 bucket:
- `s3://your-bucket-name/subsets/recipes_data_subset_*.parquet`: Split datasets
- `s3://your-bucket-name/results/recipes_data_with_similar_and_different_links_*.parquet`: Processed results

Each result file contains:
- Original recipe data
- Index for reference
- List of similar recipe indices
- List of different recipe indices

## Error Handling

The script includes comprehensive error handling:
- Logging of all major operations
- Exception handling for Spark session creation
- Graceful cleanup of resources
- Detailed error messages for debugging

## Original Dataset

The dataset used for this project is from the [RecipeNLP Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/recipenlg). The script processes this data to create similarity scores between recipes based on their titles and ingredients using TF-IDF vectorization and cosine similarity.

