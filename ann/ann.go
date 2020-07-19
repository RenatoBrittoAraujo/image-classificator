package ann

/*
	All public functions of the neural network are contained here
*/

import "image"

// Ann holds all relevant data for an artificial neural network
type Ann struct {
	name       string
	path       string
	trDataset1 []image.Image
	trDataset2 []image.Image
	teDataset  []image.Image
}

// CreateANN creates a new ANN with data
func CreateANN(name string, inputSize int, outputSize int, layerSizes []int) Ann {

}

// LoadANN loads the metadata of an ANN plus it's weights and biases
func (a *Ann) LoadANN(path string) bool {

}

// SaveANN saves all metadata of an ANN plus it's current weights and biases
func (a *Ann) SaveANN(folder string) bool {

}

// ListANNs returns a list of ANN files in a folder
func (a *Ann) ListANNs(folder string) []string {

}

// LoadDatasets loads a pair of datasets into memory
func LoadDatasets(dataset1 []image.Image, dataset2 []image.Image) {

}

// Train trains the artificial network loaded by annpath with the two dataset given
// NOTE: the order of the datasets matter
func (a *Ann) Train(dataset1 []image.Image, dataset2 []image.Image) {

}

// Test runs a test on given dataset and returns an ordered list of names from the data
// NOTE: names are associated with
func (a *Ann) Test(dataset []image.Image, label0 string, label1 string) {

}
