pipeline {
    agent {
        node {
            label 'Docker'
        }
    }

    stages {
        stage('build') {
            steps {
                withCredentials([string(credentialsId: 'ded95f1b-c18f-4a17-adb1-c6bd53933dc3', variable: 'GITHUB_TOKEN')]) {
                    sh 'curl -H "Authorization: token $GITHUB_TOKEN" -X POST \
                    --data  "{\\"state\\": \\"pending\\", \\"description\\": \\"Build Pending \\", \
                    \\"target_url\\": \\"${BUILD_URL}\\", \\"context\\": \\"continuous-integration/jenkins\\" }" \
                    --url https://api.github.com/repos/bsc-wdc/dislib/statuses/${GIT_COMMIT}'
                }
                sh "git lfs pull origin"
                sh "docker stop dislib || true"
                sh "docker rm dislib || true"
                sh "docker build --tag bscwdc/dislib ."
                sh '''#!/bin/bash
                docker run $(bash <(curl -s https://codecov.io/env)) -d --name dislib bscwdc/dislib
                '''
            }
        }
        stage('test') {
            steps {
                sh "docker exec dislib /dislib/run_ci_checks.sh"
            }
        }
        stage('deploy') {
            when {
                branch "master"
            }
            steps {
                withCredentials([[$class: 'UsernamePasswordMultiBinding', credentialsId:'dockerhub_compss', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD']]) {
                    sh 'docker login -u "$USERNAME" -p "$PASSWORD"'
                }
                sh 'docker tag bscwdc/dislib bscwdc/dislib:latest'
                sh 'docker push bscwdc/dislib:latest'
            }
        }
    }
    post{
        always {
            sh "docker exec dislib /dislib/bin/print_tests_logs.sh"
            sh "docker images"
            sh "docker stop dislib || true"
            sh "docker rm dislib || true"
            /*
            sh "printenv"
            sh 'echo ${BUILD_URL}'
            sh 'echo ${GIT_URL}'
            sh 'echo ${GIT_COMMIT}'
            sh 'echo ${BRANCH_NAME}'
            sh 'echo ${BUILD_TAG}'
            */
        }
        success {
            withCredentials([string(credentialsId: 'ded95f1b-c18f-4a17-adb1-c6bd53933dc3', variable: 'GITHUB_TOKEN')]) {
                sh '''curl -H "Authorization: token $GITHUB_TOKEN" -X POST \
                --data  "{\\"state\\": \\"success\\", \\"description\\": \\"Build Successful \\", \
                \\"target_url\\": \\"${BUILD_URL}\\", \\"context\\": \\"continuous-integration/jenkins\\" }" \
                --url https://api.github.com/repos/bsc-wdc/dislib/statuses/${GIT_COMMIT}'''
            }
        }
        failure {
            withCredentials([string(credentialsId: 'ded95f1b-c18f-4a17-adb1-c6bd53933dc3', variable: 'GITHUB_TOKEN')]) {
                sh 'curl -H "Authorization: token $GITHUB_TOKEN" -X POST \
                --data  "{\\"state\\": \\"failure\\", \\"description\\": \\"Build Failure \\", \
                \\"target_url\\": \\"${BUILD_URL}\\", \\"context\\": \\"continuous-integration/jenkins\\" }" \
                --url https://api.github.com/repos/bsc-wdc/dislib/statuses/${GIT_COMMIT}'
            }
        }
    }
}