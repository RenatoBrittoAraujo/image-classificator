package ann

/*
	All public functions of a neural network are contained here
*/

import (
	"fmt"
	"image"

	"github.com/renatobrittoaraujo/img-classificator/helpers"
)

// Ann holds all relevant data for an artificial neural network
type Ann struct {
	name       string
	path       string
	trDataset1 []image.Image
	trDataset2 []image.Image
	teDataset  []image.Image
	layers     []layer
}

// CreateANN creates a new ANN with data
func CreateANN(name string, layerSizes []int) Ann {
	if helpers.FileExists("savedneuralnetworks/" + name) {
		panic("Neural network with name " + name + " already exists")
	}
	ann := Ann{
		name: name,
		path: "savedneuralnetworks/" + name,
	}
	inputLayer := layer{
		activationFunction: actfcodeSIGMOID,
		dropoutRate:        0.1,
	}
	inputLayer.init(layerSizes[0], nil)
	ann.layers = append(ann.layers, inputLayer)
	for i := 1; i < len(layerSizes)-1; i++ {
		newLayer := layer{
			activationFunction: actfcodeSIGMOID,
			dropoutRate:        0.1,
		}
		newLayer.init(layerSizes[i], &(ann.layers[i-1]))
		ann.layers = append(ann.layers, newLayer)
	}
	outputLayer := layer{
		activationFunction: actfcodeSIGMOID,
		dropoutRate:        0.1,
	}
	outputLayer.init(layerSizes[len(layerSizes)-1], &ann.layers[len(ann.layers)-1])
	ann.layers = append(ann.layers, outputLayer)
	return ann
}

// TrainImages trains the ann with given images
// NOTE: the order of the datasets matter
func (a *Ann) TrainImages(dataset1 []image.Image, dataset2 []image.Image) {
	a.trDataset1 = dataset1
	a.trDataset2 = dataset2
	order := helpers.Permutation(len(dataset1) + len(dataset2))
	for i := range order {
		var featureMap []float64
		if order[i] >= len(dataset1) {
			featureMap = a.convertImage(dataset2[order[i]-len(dataset1)])
		} else {
			featureMap = a.convertImage(dataset1[order[i]])
		}
		a.trainCase(featureMap)
	}
}

// Test runs a test on given dataset and returns an ordered list of names from the data
// NOTE: names are associated with
// func (a *Ann) TestImages(dataset []image.Image, label0 string, label1 string) {
// 	a.teDataset = dataset
// 	order := helpers.Permutation(len(dataset))
// }

/* Private functions */

func (a *Ann) trainCase(input []float64) {
	// Do foward propagation to get output
	// Calculate loss
	// Run backpropagation
}

func (a *Ann) FowardProgation(data []float64) []float64 {
	if len(data) != len(a.layers[0].nodes) {
		panic(fmt.Sprint("Invalid input size for furst layer, shoould be", len(a.layers[0].nodes), "but is", len(data)))
	}
	data = a.layers[0].flOutput(data)
	for i := range a.layers {
		if i == 0 {
			continue
		}
		// fmt.Println("ON LAYER", i, "DATA =", data)
		data = a.layers[i].output(data)
	}
	return data
}

/* TODO */

// // LoadANN loads the metadata of an ANN plus it's weights and biases
// func (a *Ann) LoadANN(path string) bool {

// }

// // SaveANN saves all metadata of an ANN plus it's current weights and biases
// func (a *Ann) SaveANN(folder string) bool {

// }

// // ListANNs returns a list of ANN files in a folder
// func (a *Ann) ListANNs(folder string) []string {

// }
