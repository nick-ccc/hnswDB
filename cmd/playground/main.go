package main

import (
	"fmt"
	"math/rand"
	"sort"
	"time"

	"github.com/nick-ccc/hnswDB/pkg"
	"github.com/nick-ccc/hnswDB/pkg/data"
	"github.com/nick-ccc/hnswDB/pkg/vector"
)

func main() {
	build := 500000
	dimensions := 10

	myHNSW := data.MakeHNSWEuclidean[int](
		10,
		15,
		5,
		.4,
		500001,
		uint64(dimensions),
		8,
	)

	start := time.Now()
	for i := 0; i < build; i++ {
		vector := make([]int, dimensions)
		for j := range vector {
			vector[j] = rand.Intn(10) // 0 or 1
		}

		input := data.InputPairing[int]{
			Vector: vector,
			Key:    pkg.LabelType(fmt.Sprintf("vec-%d", i)),
		}

		myHNSW.Add(input)
	}
	elapsed := time.Since(start) // calculate elapsed time
	fmt.Printf("Building index of %d|%d took %s\n", build, dimensions, elapsed)

	// Search Speed
	searchVector := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	const runs = 10
	times := make([]time.Duration, 0, runs)

	for i := 0; i < runs; i++ {
		start := time.Now()

		d, err := myHNSW.Search(searchVector, 10)
		if err != nil {
			fmt.Println("Search error:", err)
			continue
		}

		_ = d // just to avoid unused variable warning

		elapsed := time.Since(start)
		times = append(times, elapsed)
		fmt.Printf("Run %d: %s\n", i+1, elapsed)
	}

	// Compute average
	var total time.Duration
	for _, t := range times {
		total += t
	}
	average := total / time.Duration(len(times))

	// Compute median
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	median := times[len(times)/2]
	if len(times)%2 == 0 {
		median = (times[len(times)/2-1] + times[len(times)/2]) / 2
	}

	fmt.Printf("\nAverage time: %s\n", average)
	fmt.Printf("Median time:  %s\n", median)

	// One last time and print results
	d, err := myHNSW.Search(searchVector, 10)
	if err != nil {
		fmt.Println(err.Error())
	}

	for i, v := range d {
		r, _ := vector.EuclideanDistance(v.Vector, searchVector)
		fmt.Printf("[%d | %f] %+v\n", i, r, v)
	}
}
