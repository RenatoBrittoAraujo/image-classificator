package main

import (
	"fmt"
	"math/rand"
	"time"

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
	cats := make([]int, 100)
	for i := 0; i < 1000000; i++ {
		a := ann.CreateANN("batata", []int{3, 4, 3, 1})
		v := a.FowardProgation([]float64{1, 2, 3})[0]
		cat := int(v * 100.0)
		cats[cat]++
	}
	for i := range cats {
		fmt.Println("{\"category\":\"", i, "\",\"column-1\":", cats[i], "},")
	}
}
