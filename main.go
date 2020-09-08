package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/renatobrittoaraujo/img-classificator/ann"
	"github.com/renatobrittoaraujo/img-classificator/dataset"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	conversions := make([]dataset.ImageConversion, 1)
	conversions[0] = dataset.CreatePooler(5)
	// DATASET NAMES
	dataset1 := dataset.GetFeatureMaps("women", conversions)
	dataset2 := dataset.GetFeatureMaps("men", conversions)
	fmt.Println("DATASET1 SIZE: ", len(dataset1))
	fmt.Println("DATASET2 SIZE: ", len(dataset2))
	fmt.Println("IMAGE SIZE:", len(dataset1[0]), len(dataset2[0]))
	for i := 0; i < 1; i++ {
		fmt.Println("CREATE ANN")
		a := ann.CreateANN("annname", []int{len(dataset1[0]), 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 1})
		for i := 0; i < 1; i++ {
			a.Train(dataset1, dataset2)
		}
		a.Test(dataset1, dataset2)
	}
}
