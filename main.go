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
			[]float64{0, 5, 0},
			[]float64{5, 1, 5},
			[]float64{0, 5, 0},
		},
	)
	conversions[1] = dataset.CreatePooler(13)
	dataset1 := dataset.GetFeatureMaps("batata", conversions)
	dataset2 := dataset.GetFeatureMaps("cenoura", conversions)
	fmt.Println("DATASET1 SIZE: ", len(dataset1))
	fmt.Println("DATASET1 IMAGE SIZE:", len(dataset1[0]))
	for i := 0; i < 1; i++ {
		fmt.Println("CREATE ANN batata")
		a := ann.CreateANN("batata", []int{len(dataset1[0]), 40, 80, 10, 10, 1})
		for i := 0; i < 1000; i++ {
			a.Train(dataset1, dataset2)
		}
	}
}
