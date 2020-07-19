package ann

type node struct {
	inEdges  []edge
	outEdges []edge
	bias     float64
}

func createNode(argInEdges []edge, argOutEdges []edge, argActivationFunction int) node {
	return node{
		inEdges:            argInEdges,
		outEdges:           argOutEdges,
		activationFunction: argActivationFunction,
	}
}

func getOutput(inputData []float64) float64 {
	return 1.0
}
