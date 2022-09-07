def setGithubCommitStatus(state, description) {
    withEnv(["STATE=$state", "DESCRIPTION=$description"]) {
         withCredentials([string(credentialsId: 'Compsupescalar Github secret token', variable: 'GITHUB_TOKEN')]) {
            sh 'curl -H "Authorization: token $GITHUB_TOKEN" -X POST \
            --data  "{\\"state\\": \\"$STATE\\", \\"description\\": \\"$DESCRIPTION\\", \
            \\"target_url\\": \\"${BUILD_URL}\\", \\"context\\": \\"continuous-integration/jenkins\\" }" \
            --url https://api.github.com/repos/bsc-wdc/dislib/statuses/${GIT_COMMIT}'
        }
    }
}

pipeline {
    options {
        timeout(time: 5, unit: 'HOURS')
    }
    agent {
        node {
            label 'Docker'
        }
    }
    stages {
        stage('build') {
            steps {
                setGithubCommitStatus('pending', 'The Jenkins build is in progress')
                sh 'git pull origin'
                sh 'docker rm -f dislib &> /dev/null || true'
                sh 'docker rmi -f bscwdc/dislib &> /dev/null || true'
                sh 'docker build --pull --no-cache --tag bscwdc/dislib .'
                sh '''#!/bin/bash
                docker run $(bash <(curl -s https://codecov.io/env)) -d --name dislib bscwdc/dislib'''
            }
        }
        stage('test') {
            steps {
                sh 'docker exec dislib /dislib/run_ci_checks.sh'
            }
        }
        stage('deploy') {
            when {
                branch 'master'
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
            sh 'docker images'
            sh 'docker rm -f dislib &> /dev/null || true'
            sh 'docker rmi -f bscwdc/dislib &> /dev/null || true'
        }
        success {
            setGithubCommitStatus('success', 'Build Successful')
        }
        failure {
            setGithubCommitStatus('failure', 'Build Failure')
        }
    }
}
