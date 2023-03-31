### Fully-managed Cloud Service (AWS App Runner, GCP Cloud Run)에서 tfserving을 이용한 추론 성능을 시험합니다.

- 각 모델 Docker container 빌드 전, build하는 호스트에서 model_download.sh를 참고하여 model을 다운로드 후 진행해야 합니다.

- Docker Image build 전, 같은 폴더에 위치한 model_download.sh를 실행 후 진행해야 합니다.
  ```shell
  chmod +x ./model_download.sh && ./model_download.sh
  ```

- Docker bulld & run 방법 (예시)
  ```shell
  tag="tfserving-cloud"
  version="latest"
  API="gRPC" #gPRC or REST
  HOST_PORT=8080
  CONTAINER_PORT=8500 #gRPC = 8500, REST = 8501
  docker build -t $tag:$version -f Dockerfile-$API .
  docker run -it -p $HOST_PORT:$CONTAINER_PORT $tag:$version
  ```