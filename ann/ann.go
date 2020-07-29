package ann

/*
	All public functions of a neural network are contained here
*/

import (
	"fmt"
	"image"
	"math"

	"github.com/renatobrittoaraujo/img-classificator/helpers"
)

// Ann holds all relevant data for an artificial neural network
type Ann struct {
	name         string
	path         string
	trDataset1   []image.Image
	trDataset2   []image.Image
	teDataset    []image.Image
	layers       []layer
	layerOutputs [][]float64
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
	ann.layerOutputs = make([][]float64, len(layerSizes))
	return ann
}

// TrainImages trains the ann with given images
// NOTE: the order of the datasets matter
func (a *Ann) TrainImages(dataset1 []image.Image, dataset2 []image.Image) {
	a.trDataset1 = dataset1
	a.trDataset2 = dataset2
	order := helpers.Permutation(len(dataset1) + len(dataset2))
	okcases := 0
	notokcases := 0
	for i := range order {
		var featureMap []float64
		expected := []float64{0}
		if order[i] >= len(dataset1) {
			featureMap = a.convertImage(dataset2[order[i]-len(dataset1)])
			expected = []float64{1.0}
		} else {
			featureMap = a.convertImage(dataset1[order[i]])
			expected = []float64{0.0}
		}
		ok := a.trainCase(featureMap, expected)
		if expected[0] == 0 {
			fmt.Println("Expect batata")
		} else {
			fmt.Println("Expect cebola")
		}
		if ok {
			fmt.Println("Got right")
		} else {
			fmt.Println("Got wrong")
		}
		if ok {
			okcases++
		} else {
			notokcases++
		}
	}
	fmt.Println("OK: ", okcases, " NOTOK: ", notokcases)
}

// Test runs a test on given dataset and returns an ordered list of names from the data
// NOTE: names are associated with
// func (a *Ann) TestImages(dataset []image.Image, label0 string, label1 string) {
// 	a.teDataset = dataset
// 	order := helpers.Permutation(len(dataset))
// }

/* Private functions */

func (a *Ann) trainCase(input []float64, expected []float64) bool {
	// Do foward propagation to get output
	res := a.FowardProgation(input)[0]
	// Run backpropagation
	a.BackPropagation(expected)
	fmt.Println("OUTPUT WAS ", fmt.Sprintf("%.2f", a.layerOutputs[len(a.layerOutputs)-1][0]))
	fmt.Println("LAYERS:")
	fmt.Println(a.layerOutputs[0])
	fmt.Println(a.layerOutputs[1])
	fmt.Println(a.layerOutputs[2])
	fmt.Println(a.layerOutputs[3])
	if res > 0.5 && expected[0] == 1 {
		return true
	}
	if res <= 0.5 && expected[0] == -1.0 {
		return true
	}
	return false
}

func (a *Ann) FowardProgation(data []float64) []float64 {
	if len(data) > len(a.layers[0].nodes) {
		data = data[0:len(a.layers[0].nodes)]
	}
	if len(data) != len(a.layers[0].nodes) {
		panic(fmt.Sprint("Invalid input size for first layer, shoould be", len(a.layers[0].nodes), "but is", len(data)))
	}
	for i := range a.layers {
		if i == 0 {
			data = a.layers[0].flOutput(data)
		} else {
			data = a.layers[i].output(data)
		}
		a.layerOutputs[i] = data
	}
	return data
}

func (a *Ann) BackPropagation(expected []float64) {
	learningRate := 0.01
	for i := len(a.layers) - 1; i > 0; i-- {
		output := a.layerOutputs[i]
		// Update weights
		for j := 0; j < len(a.layers[i].nodes); j++ {
			for k := 0; k < len(a.layers[i].nodes[j].inEdges); k++ {
				fmt.Println("OLD WEIGHT", a.layers[i].nodes[j].inEdges[k].weight)
				fmt.Println("EXPECTED[J]", expected[j])
				fmt.Println("activationFunctionDerivative", activationFunctionDerivative(actfcodeSIGMOID, output[j]))
				a.layers[i].nodes[j].inEdges[k].weight = a.layers[i].nodes[j].inEdges[k].weight +
					2*(expected[j]-activationFunctionDerivative(actfcodeSIGMOID, output[j]))*learningRate
				fmt.Println("NEW WEIGHT", a.layers[i].nodes[j].inEdges[k].weight)
				if math.IsNaN(a.layers[i].nodes[j].inEdges[k].weight) {
					panic("WHAT THE FUCK")
				}
			}
		}
		// Update biases
		for j := 0; j < len(a.layers[i].nodes); j++ {
			a.layers[i].nodes[j].bias = a.layers[i].nodes[j].bias +
				2*(expected[j]-activationFunctionDerivative(actfcodeSIGMOID, output[j]))*learningRate
		}
		expectedList := make([]float64, len(a.layers[i-1].nodes))
		// Update expected list
		for j := 0; j < len(a.layers[i].nodes); j++ {
			for k := 0; k < len(a.layers[i].nodes[j].inEdges); k++ {
				expectedList[k] += (+2*expected[j] +
					-2*activationFunctionDerivative(actfcodeSIGMOID, output[j])) * learningRate
			}
		}
		expected = expectedList
	}
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
