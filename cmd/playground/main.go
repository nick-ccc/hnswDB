package main

import (
	"fmt"

	"github.com/nick-ccc/hnswDB/pkg/data"
)

func main() {
	myHNSW := data.MakeHNSWEuclidean[int](
		10,
		15,
		5,
		.3,
		5000,
		5,
		8,
	)

	input := data.InputPairing[int]{
		Vector: []int{1, 0, 1, 1, 1},
		Key:    "Hello World",
	}

	myHNSW.Add(input)
	d, err := myHNSW.Search([]int{1, 1, 1, 1, 1}, 10)
	if err != nil {
		fmt.Println(err.Error())
	}

	fmt.Println(d)
}
