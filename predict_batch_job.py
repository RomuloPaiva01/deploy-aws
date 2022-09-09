import sagemaker
from sagemaker.transformer import Transformer

BUCKET = 'iris-demo-s3'
MODEL = 'iris-demo-model-yqzyt0nlsesp9bfptmffabg'

INPUT_DATA = 's3://iris-demo-s3/test.csv'

sess = sagemaker.Session()

transformer_job = Transformer(
    MODEL,
    1,
    "ml.m4.xlarge",
    output_path=f"s3://{BUCKET}/output_transform",
    sagemaker_session=sess,
    strategy="MultiRecord",
    assemble_with="Line",
)

transformer_job.transform(INPUT_DATA, content_type="text/csv", split_type="Line")