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
                docker build --tag bscwdc/dislib .
                docker rmi -f $(docker images |grep 'dislib')
                docker run $(bash <(curl -s https://codecov.io/env)) -d --name dislib bscwdc/dislib
                docker exec dislib /dislib/run_ci_checks.sh
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
}