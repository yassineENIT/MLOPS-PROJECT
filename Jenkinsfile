pipeline {
  agent any

  environment {
    VENV_DIR = 'venv'
    GCP_PROJECT = 'manifest-wind-474003-j9'
    GCLOUD_PATH = '/var/jenkins_home/gcloud/google-cloud-sdk/bin'
  }

  stages {
    stage('Cloning Github repo to Jenkins') {
      steps {
        script {
          echo 'Cloning github repo to Jenkins ........'
          checkout scmGit(
            branches: [[name: '*/main']],
            extensions: [],
            userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/yassineENIT/MLOPS-PROJECT.git']]
          )
        }
      }
    }

    stage('Setting up our Virtual Environment and installing dependencies') {
      steps {
        script {
          echo 'Setting up our Virtual Environment and installing dependencies ........'
          sh '''
            set -eu
            python3 -m venv "${VENV_DIR}"
            . "${VENV_DIR}/bin/activate"
            pip install --upgrade pip
            pip install -e .
          '''
        }
      }
    }

    stage('Building and pushing Docker image to GCR') {
      steps {
        withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
          script {
            echo 'Building and pushing Docker image to GCR ........'
            sh '''
              set -eu
              export PATH="$PATH:${GCLOUD_PATH}"
              gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
              gcloud config set project "${GCP_PROJECT}"
              gcloud auth configure-docker -q
              docker build -t "gcr.io/${GCP_PROJECT}/mlops-project:latest" .
              docker push "gcr.io/${GCP_PROJECT}/mlops-project:latest"
            '''
          }
        }
      }
    }
  }
}
