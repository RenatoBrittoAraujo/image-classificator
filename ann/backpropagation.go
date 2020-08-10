package ann

func (a *Ann) BackPropagation(expected []float64) {
	learningRate := 0.01
	for i := len(a.layers) - 1; i > 0; i-- {
		output := a.layerOutputs[i]

		// Update weights
		for j := 0; j < len(a.layers[i].nodes); j++ {
			for k := 0; k < len(a.layers[i].nodes[j].inEdges); k++ {
				v := a.layerOutputs[i-1][k]
				s := a.layerSums[i][j]
				o := output[j]
				derivativeNudge :=
					-2.0 * (expected[j] - o) *
						activationFunctionDerivative(actfcodeTANH, s) *
						v * learningRate
				a.layers[i].nodes[j].inEdges[k].weight -= derivativeNudge
			}
		}

		// Update biases
		for j := 0; j < len(a.layers[i].nodes); j++ {
			s := a.layerSums[i][j]
			o := output[j]
			derivativeNudge := (-2 * (expected[j] - o) *
				activationFunctionDerivative(actfcodeTANH, s)) *
				learningRate
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
						activationFunctionDerivative(actfcodeTANH, s) *
						w * learningRate
				expectedList[k] = a.layerOutputs[i-1][k] - derivativeNudge
			}
		}
		expected = expectedList
	}
}
