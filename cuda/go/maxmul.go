package main

/*
#cgo LDFLAGS: -L. -lmaxmul
void maxmul(int *A, int* B, int *C, int size);
*/
import "C"

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

func Maxmul(a []C.int, b []C.int, size int, wg *sync.WaitGroup, reqIdx int, standardTime time.Time) {
	wg.Add(1)
	defer wg.Done()

	var startTime = time.Since(standardTime)
	var totalSize = size * size
	var c = make([]C.int, totalSize)

	C.maxmul(&a[0], &b[0], &c[0], C.int(size))

	var endTime = time.Since(standardTime)
	var totalTime = endTime - startTime
	fmt.Printf("[%d] c[0]=%d, start time: %s, end time: %s, processing time: %s\n", reqIdx, c[0], startTime, endTime, totalTime)
}

func main() {
	var waitGroup sync.WaitGroup

	var reqNumPerSec = 100
	var reqSec = 10
	var totalReq = reqNumPerSec * reqSec
	var rowSize int = 600
	var totalSize = rowSize * rowSize

	var aa = make([]C.int, totalSize)
	var bb = make([]C.int, totalSize)

	for i := 0; i < totalSize; i++ {
		aa[i] = C.int(rand.Intn(100))
		bb[i] = C.int(rand.Intn(100))
	}

	var sleepTime = float64(1) / float64(reqNumPerSec)
	var sleepDuration = time.Second
	if sleepTime < 1 {
		sleepTime *= 1000
		sleepDuration = time.Millisecond
	}

	var standardTime = time.Now()
	for i := 0; i < totalReq; i++ {
		go Maxmul(aa, bb, rowSize, &waitGroup, i, standardTime)
		time.Sleep(sleepDuration * time.Duration(sleepTime))
	}

	waitGroup.Wait()
}
