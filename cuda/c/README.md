1. 먼저 maxmul.cu를 공유 라이브러리로 컴파일한다.
```
nvcc --ptxas-options=-v --compiler-options '-fPIC' -o libmaxmul.so --shared maxmul.cu
```

2. 위 라이브러리를 포함하여 maxmul.c를 컴파일한다.
```
gcc -o maxmul maxmul.c -L. -lpthread -lmaxmul
```

3. 실행
```
./maxmul
```