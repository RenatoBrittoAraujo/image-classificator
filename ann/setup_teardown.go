package ann

/*
	Functions involving the setup and teardown of the neural network
*/

import (
	"github.com/renatobrittoaraujo/img-classificator/helpers"
)

// testData holds a test item for a neural network
type testData struct {
	label      string
	attributes []float64
}

// Ann holds all relevant data for an artificial neural network
type Ann struct {
	name         string
	path         string
	layers       []layer
	layerOutputs [][]float64
	layerSums    [][]float64

	outputNodeName    map[int]string
	trainingDatasets  map[string][][]float64
	testDataset       []testData
	validationDataset []testData
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
		activationFunction: actfcodeTANH,
		dropoutRate:        0.1,
	}
	inputLayer.init(layerSizes[0], nil)
	ann.layers = append(ann.layers, inputLayer)
	for i := 1; i < len(layerSizes)-1; i++ {
		newLayer := layer{
			activationFunction: actfcodeTANH,
			dropoutRate:        0.1,
		}
		newLayer.init(layerSizes[i], &(ann.layers[i-1]))
		ann.layers = append(ann.layers, newLayer)
	}
	outputLayer := layer{
		activationFunction: actfcodeTANH,
		dropoutRate:        0.1,
	}
	outputLayer.init(layerSizes[len(layerSizes)-1], &ann.layers[len(ann.layers)-1])
	ann.layers = append(ann.layers, outputLayer)
	ann.layerOutputs = make([][]float64, len(layerSizes))
	ann.layerSums = make([][]float64, len(layerSizes))
	return ann
}

// SaveAnn creates a file that contains all relevant
// data (including parameters) of the ann and saves it
// to a fiel
func (a *Ann) SaveAnn(path string) bool {
	return true
}

// LoadTrainingDataset loads a new dataset for the neural network
// this dataset is not saved in memory, so should you ever
// want to use it again, you will have to load all of them
func LoadTrainingDataset(name string, data [][]float64) bool {

	return true
}

// LoadTestingDataset loads a new dataset for the neural network
// this dataset is not saved in memory, so should you ever
// want to use it again, you will have to load all of them
func LoadTestingDataset(data []float64) bool {

	return true
}

// LoadValidationDataset loads a new dataset for the neural network
// this dataset is not saved in memory, so should you ever
// want to use it again, you will have to load all of them
func LoadValidationDataset(data []float64) bool {

	return true
}

// IsReady returns a boolean if neural network object is ready for being trained/tested
func (a *Ann) IsReady() (bool, string) {

	return true, ""
}
