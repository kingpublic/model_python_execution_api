pipeline {
    agent any
    environment {
        IMAGE_NAME = 'model-python-execution-api'
        IMAGE_TAG  = 'latest'
        DEPLOY_DIR = '/home/moodbites/Model_Python_Execution_API'
        HOST_IP    = '103.185.52.161'        // ← ganti jika IP berbeda
        HOST_USER  = 'moodbites'
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Build & Deploy') {
            steps {
                withCredentials([
                    sshUserPrivateKey(
                        credentialsId: 'moodbites-host-ssh',
                        keyFileVariable: 'SSH_KEY',
                        usernameVariable: 'SSH_USER'
                    )
                ]) {
                    sh '''
                        # ── 1. Reset repo di host ke origin/main ──────────────────────
                        ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
                            $SSH_USER@${HOST_IP} "
                                cd ${DEPLOY_DIR} &&
                                git fetch origin &&
                                git reset --hard origin/main &&
                                git clean -fd
                            "

                        # ── 2. Kirim docker-compose ke host ───────────────────────────
                        scp -i $SSH_KEY -o StrictHostKeyChecking=no \
                            docker-compose.yml \
                            $SSH_USER@${HOST_IP}:${DEPLOY_DIR}/docker-compose.yml

                        # ── 3. Build image & deploy di host ───────────────────────────
                        ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
                            $SSH_USER@${HOST_IP} "
                                docker compose -f ${DEPLOY_DIR}/docker-compose.yml down || true &&
                                docker build -t ${IMAGE_NAME}:${IMAGE_TAG} \
                                    ${DEPLOY_DIR} &&
                                docker compose -f ${DEPLOY_DIR}/docker-compose.yml up -d
                            "
                    '''
                }
            }
        }
    }
    post {
        success {
            echo 'Pipeline berhasil! Model Python API berjalan di port 8067.'
        }
        failure {
            echo 'Pipeline gagal! Periksa log di atas.'
        }
    }
}