pipeline {
    agent {
        node {
            label 'Docker'
        }
    }

    stages {
        stage('tests') {
            steps {
                sh '''#!/bin/bash
                git lfs pull origin
                docker stop dislib
                docker rm dislib
                docker build --tag bscwdc/dislib .
                docker run $(bash <(curl -s https://codecov.io/env)) -d --name dislib bscwdc/dislib
                # docker exec dislib /dislib/run_ci_checks.sh
                docker images
                docker exec dislib /dislib/bin/print_tests_logs.sh'''
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
            sh '''#!/bin/bash
            docker stop dislib
            docker rm dislib'''
        }
        success {
            echo '${env.BUILD_URL}'
            echo '${env.BUILD_TAG}'
            echo '${env.GIT_URL}'
            echo '${env.GIT_COMMIT}'
            echo '${env.BRANCH_NAME}'
            withCredentials([string(credentialsId: 'ded95f1b-c18f-4a17-adb1-c6bd53933dc3', variable: 'GITHUB_TOKEN')]) {
                sh 'curl -H "Authorization: token $GITHUB_TOKEN" -X POST --data  "{\\"state\\": \\"success\\", \\"description\\": \\"Build Successful \\", \\"target_url\\": \\"$BUILD_URL\\", \\"context\\": \\"$BUILD_TAG\\" }" --url ${env.GIT_URL}/statuses/$GIT_COMMIT'
            }
        }
        failure {
            withCredentials([string(credentialsId: 'ded95f1b-c18f-4a17-adb1-c6bd53933dc3', variable: 'GITHUB_TOKEN')]) {
                sh 'curl -H "Authorization: token $GITHUB_TOKEN" -X POST --data  "{\\"state\\": \\"failure\\", \\"description\\": \\"Build Failure \\", \\"target_url\\": \\"$BUILD_URL\\", \\"context\\": \\"$BUILD_TAG\\" }" --url ${env.GIT_URL}/statuses/$GIT_COMMIT'
            }
        }
    }
}