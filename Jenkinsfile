pipeline {
    agent {
        node {
            label 'Docker'
        }
    }

    stages {
        stage('before_script') {
            steps {
                bash 'docker build --tag bscwdc/dislib .'
                bash 'docker run $(bash <(curl -s https://codecov.io/env)) -d --name dislib bscwdc/dislib'
            }
        }
        stage('script') {
            steps {
                sh 'docker images'
                sh 'docker exec dislib /dislib/bin/print_tests_logs.sh'
            }
        }
        stage('after_script') {
            steps {
            sh 'docker images'
            sh 'docker exec dislib /dislib/bin/print_tests_logs.sh'
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