package main

import (
	"fmt"

	"github.com/renatobrittoaraujo/img-classificator/dataset"
)

func main() {
	fmt.Println(dataset.GetDataset("batata"))
}
