package ann

/*
	All public functions of a neural network are contained here
*/

import (
	"fmt"
	"image"
	"os"
	"os/exec"
	"time"

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
	order := helpers.Permutation(10000)
	okcases := 0
	notokcases := 0
	list := make([]float64, 0)
	for i := range order {
		var featureMap []float64
		expected := []float64{0}
		if order[i] >= 5000 /*len(dataset1)*/ {
			// featureMap = a.convertImage(dataset2[order[i]-len(dataset1)])
			featureMap = []float64{0, 1}
			expected = []float64{0.0}
		} else {
			// featureMap = a.convertImage(dataset1[order[i]])
			featureMap = []float64{1, 0}
			expected = []float64{1.0}
		}
		fmt.Println("PUT:", featureMap, "EXPECT:", expected)
		ok, res := a.trainCase(featureMap, expected)
		list = append(list, res)
		// if expected[0] == 0.0 {
		// 	fmt.Println("Expect white")
		// } else {
		// 	fmt.Println("Expect black")
		// }
		// if ok {
		// 	fmt.Println("Got right")
		// } else {
		// 	fmt.Println("Got wrong")
		// }
		if ok {
			okcases++
		} else {
			notokcases++
		}
	}
	// fmt.Println(list)
	fmt.Println("OK: ", okcases, " NOTOK: ", notokcases)
}

/* Private functions */
func (a *Ann) trainCase(input []float64, expected []float64) (bool, float64) {
	// Do foward propagation to get output
	res := a.FowardProgation(input)
	// Run backpropagation
	a.BackPropagation([]float64{lossFunction(lossfcodeMSE, res, expected)})
	fmt.Println("OUTPUT WAS ", fmt.Sprintf("%.2f", a.layerOutputs[len(a.layerOutputs)-1][0]))
	fmt.Println(a.layerOutputs[0])
	fmt.Println(a.layerOutputs[1])
	// fmt.Println(a.layerOutputs[2])
	onems, _ := time.ParseDuration("1s")
	time.Sleep(onems)
	cmd := exec.Command("clear") //Linux example, its tested
	cmd.Stdout = os.Stdout
	cmd.Run()
	if res[0] > 0.5 && expected[0] == 1 {
		return true, res[0]
	}
	if res[0] <= 0.5 && expected[0] == 0.0 {
		return true, res[0]
	}
	return false, res[0]
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
	learningRate := 1.0
	for i := len(a.layers) - 1; i >= 0; i-- {
		output := a.layerOutputs[i]
		expectedList := make([]float64, 0)

		// Update weights
		for j := 0; j < len(a.layers[i].nodes) && i != 0; j++ {
			for k := 0; k < len(a.layers[i].nodes[j].inEdges); k++ {
				fmt.Println("NUDGE TO EDGE ", k, j, "IS", -(-2*(expected[j]-output[j])*
					activationFunctionDerivative(actfcodeSIGMOID, output[j])*
					a.layerOutputs[i-1][k])*learningRate)
				a.layers[i].nodes[j].inEdges[k].weight = a.layers[i].nodes[j].inEdges[k].weight -
					(-2*(expected[j]-output[j])*
						activationFunctionDerivative(actfcodeSIGMOID, output[j])*
						a.layerOutputs[i-1][k])*learningRate
				fmt.Println("EDGE IS NOW", a.layers[i].nodes[j].inEdges[k].weight)
			}
		}

		// Update biases
		// for j := 0; j < len(a.layers[i].nodes); j++ {
		// 	a.layers[i].nodes[j].bias = a.layers[i].nodes[j].bias -
		// 		(-2*(expected[j]-output[j])*
		// 			activationFunctionDerivative(actfcodeSIGMOID, output[j]))*learningRate
		// }

		// Update expected list
		if i != 0 {
			expectedList = make([]float64, len(a.layers[i-1].nodes))
			for j := 0; j < len(a.layers[i].nodes); j++ {
				for k := 0; k < len(a.layers[i].nodes[j].inEdges); k++ {
					expectedList[k] = a.layerOutputs[i-1][k] - (-2*(expected[j]-output[j])*
						activationFunctionDerivative(actfcodeSIGMOID, output[j])*
						a.layers[i].nodes[j].inEdges[k].weight)*learningRate
				}
			}
		}

		expected = expectedList
	}
}
