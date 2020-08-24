package ann

/*
	All public functions of a neural network are contained here
*/

import (
	"fmt"

	"github.com/renatobrittoaraujo/img-classificator/helpers"
)

// Ann holds all relevant data for an artificial neural network
type Ann struct {
	name         string
	path         string
	layers       []layer
	layerOutputs [][]float64
	layerSums    [][]float64
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
	ann.layerSums = make([][]float64, len(layerSizes))
	return ann
}

// Train trains the ann with given inputs
// NOTE: the order of the datasets matter
func (a *Ann) Train(dataset1 [][]float64, dataset2 [][]float64) {
	order := helpers.Permutation(len(dataset1) + len(dataset2))
	okcases := 0
	notokcases := 0
	dataset1misses := 0
	dataset2misses := 0
	for i := range order {
		var featureMap []float64
		expected := []float64{0}
		datasetIndex := 0
		if order[i] >= len(dataset1) {
			datasetIndex = 1
		}
		if datasetIndex == 1 {
			featureMap = dataset2[order[i]-len(dataset1)]
			expected = []float64{1.0}
		} else {
			featureMap = dataset1[order[i]]
			expected = []float64{0.0}
		}
		ok := a.trainCase(featureMap, expected)
		if ok {
			okcases++
		} else {
			notokcases++
			if datasetIndex == 0 {
				dataset1misses++
			} else {
				dataset2misses++
			}
		}
	}
	fmt.Println("DATASET1 MISSES:", dataset1misses, "DATASET2 MISSES:", dataset2misses)
	fmt.Println("OK: ", okcases, " NOTOK: ", notokcases)
	fmt.Println("PRECISION: ", float64(okcases)/float64(okcases+notokcases))
}

/* Private functions */
func (a *Ann) trainCase(input []float64, expected []float64) bool {
	res := a.FowardProgation(input)
	a.BackPropagation(expected)
	if res[0] > 0.5 && expected[0] == 1 {
		return true
	}
	if res[0] <= 0.5 && expected[0] == 0.0 {
		return true
	}
	return false
}

func (a *Ann) FowardProgation(data []float64) []float64 {
	if len(data) > len(a.layers[0].nodes) {
		data = data[0:len(a.layers[0].nodes)]
	}
	if len(data) != len(a.layers[0].nodes) {
		panic(fmt.Sprint("Invalid input size for first layer, should be", len(a.layers[0].nodes), "but is", len(data)))
	}
	for i := range a.layers {
		if i == 0 {
			data = a.layers[0].flOutput(data)
		} else {
			data = a.layers[i].sumOutput(data)
		}
		a.layerSums[i] = append([]float64(nil), data...)
		for i := range data {
			data[i] = activationFunction(actfcodeSIGMOID, data[i])
		}
		a.layerOutputs[i] = data
	}
	return data
}

func (a *Ann) BackPropagation(expected []float64) {
	learningRate := 0.1
	for i := len(a.layers) - 1; i > 0; i-- {
		output := a.layerOutputs[i]

		// Update weights
		for j := 0; j < len(a.layers[i].nodes); j++ {
			for k := 0; k < len(a.layers[i].nodes[j].inEdges); k++ {
				v := a.layerOutputs[i-1][k]
				s := a.layerSums[i][j]
				o := output[j]
				derivativeNudge :=
					-2 * (expected[j] - o) *
						activationFunctionDerivative(actfcodeSIGMOID, s) *
						v * learningRate
				a.layers[i].nodes[j].inEdges[k].weight -= derivativeNudge
			}
		}

		// Update biases
		for j := 0; j < len(a.layers[i].nodes); j++ {
			s := a.layerSums[i][j]
			o := output[j]
			derivativeNudge := (-2 * (expected[j] - o) *
				activationFunctionDerivative(actfcodeSIGMOID, s))
			a.layers[i].nodes[j].bias -= derivativeNudge
		}

		// Update expected list
		expectedList := make([]float64, len(a.layers[i-1].nodes))
		for j := 0; j < len(a.layers[i].nodes); j++ {
			for k := 0; k < len(a.layers[i].nodes[j].inEdges); k++ {
				w := a.layers[i].nodes[j].inEdges[k].weight
				s := a.layerSums[i][j]
				o := output[j]
				derivativeNudge :=
					-2 * (expected[j] - o) *
						activationFunctionDerivative(actfcodeSIGMOID, s) *
						w * learningRate
				expectedList[k] = a.layerOutputs[i-1][k] - derivativeNudge
			}
		}
		expected = expectedList
	}
}
