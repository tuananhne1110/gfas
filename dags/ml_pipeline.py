from datetime import datetime
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
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Create DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Machine Learning Training Pipeline',
    schedule_interval=None,
    catchup=False
)

# Define tasks
checkout = BashOperator(
    task_id='checkout',
    bash_command='''
        if [ -d "/opt/airflow/workspace/gfas" ]; then
            cd /opt/airflow/workspace/gfas && \
            git fetch && \
            git reset --hard origin/main
        else
            git clone --depth 1 https://github.com/tuananhne1110/gfas.git /opt/airflow/workspace/gfas
        fi
    ''',
    dag=dag
)

setup_env = BashOperator(
    task_id='setup_environment',
    bash_command='''
        cd /opt/airflow/workspace/gfas && \
        if [ ! -d "venv" ]; then
            python -m venv venv && \
            . venv/bin/activate && \
            pip install -r requirements.txt
        else
            . venv/bin/activate && \
            if [ requirements.txt -nt venv/.requirements_installed ]; then
                pip install -r requirements.txt && \
                touch venv/.requirements_installed
            fi
        fi
    ''',
    dag=dag
)

configure_git = BashOperator(
    task_id='configure_git',
    bash_command='''
        if ! git config --global user.name > /dev/null; then
            git config --global user.name "Airflow"
        fi
        if ! git config --global user.email > /dev/null; then
            git config --global user.email "airflow@example.com"
        fi
    ''',
    dag=dag
)

configure_dvc = BashOperator(
    task_id='configure_dvc',
    bash_command='''
        cd /opt/airflow/workspace/gfas && \
        . venv/bin/activate && \
        # Initialize DVC if not already initialized
        if [ ! -d .dvc ]; then
            dvc init --no-scm
        fi
        # Configure DVC remote if not already configured
        if [ ! -f .dvc/.credentials_configured ] || [ ! -f .dvc/config.local ]; then
            dvc remote modify --local minio access_key_id {{ var.value.MINIO_ACCESS_KEY }} && \
            dvc remote modify --local minio secret_access_key {{ var.value.MINIO_SECRET_KEY }} && \
            touch .dvc/.credentials_configured
        fi
    ''',
    dag=dag
)

pull_data = BashOperator(
    task_id='pull_data',
    bash_command='''
        cd /opt/airflow/workspace/gfas && \
        . venv/bin/activate && \
        # Check if data has changed on remote
        dvc status || dvc pull
    ''',
    dag=dag
)

train_model = BashOperator(
    task_id='train_model',
    bash_command='''
        cd /opt/airflow/workspace/gfas && \
        . venv/bin/activate && \
        python scripts/pipeline.py
    ''',
    dag=dag
)

# Define task dependencies
checkout >> setup_env >> configure_git >> configure_dvc >> pull_data >> train_model 