Edge device (nvidia jetson, raspberry pi with coral TPU)에서 tfserving을 이용한 추론 성능을 시험합니다.

## 모델 다운로드
- model은 models/README.md를 참고하여 다운로드 가능합니다.

## 각 장비에서 사용가능한 도커파일
- xavier, tx 장비에서 사용가능한 Dockerfile과 nano 장비에서 사용가능한 Dockerfile이 각각 따로 있습니다. (Dockerfile_gRPC_xavier_tx, Dockerfile_gRPC_nano)


## 도커파일 빌드 및 실행파일
- docker_build_gRPC_xavier_tx.sh 또는 docker_build_gRPC_nano.sh로 도커파일을 빌드할 수 있습니다.
- docker_run_gRPC.sh로 빌드한 이미지를 실행 할 수 있습니다.
- 실행 하기 전 각 파일의 실행 권한을 확인 후 권한이 없으면 권한을 주어야합니다.
  - xavier, tx 장비 예시
    ```shell
    chmod +x docker_build_gRPC_xavier_tx.sh docker_run_gRPC.sh 
    ```
- 권한이 있다면 다음과 같이 실행 할 수 있습니다.
  - xavier, tx 장비 예시
    ```shell
    ./docker_build_gRPC_xavier_tx.sh && ./docker_run_gRPC.sh 
    ```