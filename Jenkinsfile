pipeline {
    agent any
    
    environment {
        MINIO_ACCESS_KEY = credentials('minio-access-key')
        MINIO_SECRET_KEY = credentials('minio-secret-key')
    }
    
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/tuananhne1110/gfas.git',
                    credentialsId: 'github-credentials'
            }
        }
        
        stage('Setup Python') {
            steps {
                sh '''
                    docker pull python:3.10-slim
                    docker run --rm -v ${WORKSPACE}:/workspace -w /workspace python:3.10-slim pip install -r requirements.txt
                '''
            }
        }
        
        stage('Configure Git') {
            steps {
                sh '''
                    git config --global user.name "Jenkins"
                    git config --global user.email "jenkins@example.com"
                '''
            }
        }
        
        stage('Configure DVC') {
            steps {
                sh '''
                    dvc remote modify --local minio access_key_id $MINIO_ACCESS_KEY
                    dvc remote modify --local minio secret_access_key $MINIO_SECRET_KEY
                '''
            }
        }
        
        stage('Pull Data') {
            steps {
                sh 'dvc pull'
            }
        }
        
        stage('Train Model') {
            steps {
                sh '''
                    docker run --rm \
                        -v ${WORKSPACE}:/workspace \
                        -w /workspace \
                        -e MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY} \
                        -e MINIO_SECRET_KEY=${MINIO_SECRET_KEY} \
                        python:3.10-slim \
                        python scripts/pipeline.py
                '''
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'logs/**', allowEmptyArchive: true
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
} 
