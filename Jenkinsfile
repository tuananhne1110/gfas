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
                    # Check Python version
                    python3 --version || apt-get update && apt-get install -y python3 python3-venv
                    
                    # Create and activate virtual environment
                    python3 -m venv venv
                    . venv/bin/activate
                    
                    # Install requirements
                    pip install -r requirements.txt
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
                    . venv/bin/activate
                    dvc remote modify --local minio access_key_id $MINIO_ACCESS_KEY
                    dvc remote modify --local minio secret_access_key $MINIO_SECRET_KEY
                '''
            }
        }
        
        stage('Pull Data') {
            steps {
                sh '''
                    . venv/bin/activate
                    dvc pull
                '''
            }
        }
        
        stage('Train Model') {
            steps {
                sh '''
                    . venv/bin/activate
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
