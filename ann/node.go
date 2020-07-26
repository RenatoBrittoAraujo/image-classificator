package ann

type node struct {
	inEdges []edge
	bias    float64
}

func (n *node) output(inputData []float64, actfcode int) float64 {
	sum := n.bias
	for i := 0; i < len(inputData); i++ {
		sum += n.inEdges[i].weight * inputData[i]
	}
	return activationFunction(actfcode, sum)
}

func (n *node) flOutput(inputData float64, actfcode int) float64 {
	sum := n.bias + inputData
	return activationFunction(actfcode, sum)
}
