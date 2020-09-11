package ann

import "fmt"

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
			data[i] = activationFunction(actfcodeTANH, data[i])
		}
		a.layerOutputs[i] = data
	}
	return data
}
