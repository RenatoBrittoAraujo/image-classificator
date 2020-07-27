package main

import (
	"math/rand"
	"time"

	"github.com/renatobrittoaraujo/img-classificator/dataset"

	"github.com/renatobrittoaraujo/img-classificator/ann"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	// args := os.Args[1:]
	// if args[0] == "train" {
	// 	dataset1name := args[1]
	// 	dataset2name := args[2]
	// 	fmt.Println("Training with [", dataset1name, dataset2name, "] datasets...")
	// 	// dataset1 := dataset.GetDataset(dataset1name)
	// 	// dataset2 := dataset.GetDataset(dataset2name)
	// 	for {
	// 		fmt.Println("AHAH")
	// 	}
	// } else {
	// 	fmt.Println("What the fuck is this command?")
	// }
	for i := 0; i < 1; i++ {
		a := ann.CreateANN("batata", []int{10, 10, 10, 1})
		// v := a.FowardProgation([]float64{1, 2, 3})[0]
		dataset1 := dataset.GetDataset("batata")
		dataset2 := dataset.GetDataset("cenoura")
		a.TrainImages(dataset2, dataset1)
		a.TrainImages(dataset1, dataset2)
	}
}
