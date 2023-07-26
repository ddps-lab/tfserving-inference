1. 먼저 maxmul.cu를 공유 라이브러리로 컴파일한다.
```
nvcc --ptxas-options=-v --compiler-options '-fPIC' -o libmaxmul.so --shared maxmul.cu
```

2. maxmul.go 파일 실행
```
go run maxmul.go
```
