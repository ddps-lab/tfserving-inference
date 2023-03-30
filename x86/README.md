### Fully-managed Cloud Service (AWS App Runner, GCP Cloud Run)에서 tfserving을 이용한 추론 성능을 시험합니다.

- 각 모델 Docker container 빌드 전, build하는 호스트에서 model_download.sh를 참고하여 model을 다운로드 후 진행해야 합니다.

- Docker Image build시, Download한 model 폴더를 같은 Directory에 복사 후 진행해야 합니다.
  - Edge 장비의 경우, 각각 model을 COPY 후 빌드할 필요 없이, docker -v option을 사용하여도 됩니다.