package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/renatobrittoaraujo/img-classificator/dataset"

	"github.com/renatobrittoaraujo/img-classificator/ann"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	conversions := make([]dataset.ImageConversion, 2)
	conversions[0] = dataset.CreateFilter(
		[][]float64{
			[]float64{1, 2, 1},
			[]float64{2, 2, 2},
			[]float64{1, 2, 1},
		},
	)
	conversions[1] = dataset.CreatePooler(20)
	dataset1 := dataset.GetFeatureMaps("black", conversions)
	dataset2 := dataset.GetFeatureMaps("white", conversions)
	fmt.Println("DATASET1 SIZE: ", len(dataset1))
	fmt.Println("DATASET1 IMAGE SIZE:", len(dataset1[0]))
	for i := 0; i < 1; i++ {
		fmt.Println("CREATE ANN batata")
		a := ann.CreateANN("batata", []int{len(dataset1[0]), 10, 10, 1})
		for i := 0; i < 1000; i++ {
			a.Train(dataset2, dataset1)
		}
	}
}
