package ann

type node struct {
	inEdges []edge
	bias    float64
}

func (n *node) getOutput(inputData []float64) float64 {
	sum := n.bias
	for i := 0; i < len(inputData); i++ {
		sum += n.inEdges[i].weight * inputData[i]
	}
	return sum
}
