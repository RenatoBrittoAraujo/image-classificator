package main

import (
	"fmt"
	"os"
)

func main() {
	args := os.Args[1:]
	if args[0] == "train" {
		dataset1name := args[1]
		dataset2name := args[2]
		fmt.Println("Training with [", dataset1name, dataset2name, "] datasets...")
		// dataset1 := dataset.GetDataset(dataset1name)
		// dataset2 := dataset.GetDataset(dataset2name)
		for {
			fmt.Println("AHAH")
		}
	} else {
		fmt.Println("What the fuck is this command?")
	}
}
