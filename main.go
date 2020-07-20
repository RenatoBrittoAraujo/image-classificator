package main

import (
	"fmt"

	"github.com/renatobrittoaraujo/img-classificator/ann"
	"github.com/renatobrittoaraujo/img-classificator/dataset"
)

func main() {
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
	for i := 0; i < 10; i++ {
		image := dataset.GetDataset("batata")[i]
		a := ann.Ann{}
		fmt.Println(a.Convert(image))
	}
}
