from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import os

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Machine Learning Training Pipeline with DVC and MLflow',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'training', 'dvc', 'mlflow']
)

# Define environment variables
env_vars = {
    'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
    'AWS_ACCESS_KEY_ID': Variable.get('AWS_ACCESS_KEY_ID', default_var='minioadmin'),
    'AWS_SECRET_ACCESS_KEY': Variable.get('AWS_SECRET_ACCESS_KEY', default_var='minioadmin'),
    'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000',
    'MLFLOW_S3_IGNORE_TLS': 'true'
}

# Define tasks
check_dvc_changes = BashOperator(
    task_id='check_dvc_changes',
    bash_command='''
        cd /opt/airflow/workspace/gfas/yolov11_mlflow && \
        git pull && \
        dvc status
    ''',
    env=env_vars,
    dag=dag
)

pull_data = BashOperator(
    task_id='pull_data',
    bash_command='''
        cd /opt/airflow/workspace/gfas/yolov11_mlflow && \
        dvc pull --force
    ''',
    env=env_vars,
    dag=dag
)

train_model = BashOperator(
    task_id='train_model',
    bash_command='''
        cd /opt/airflow/workspace/gfas/yolov11_mlflow && \
        python scripts/pipeline.py
    ''',
    env=env_vars,
    dag=dag
)

# Define task dependencies
check_dvc_changes >> pull_data >> train_model 