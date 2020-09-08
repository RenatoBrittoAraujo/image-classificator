package ann

import (
	"math"

	"github.com/renatobrittoaraujo/img-classificator/helpers"
)

type layer struct {
	nodes              []node
	activationFunction int
	dropoutRate        float64 // only active on training mode
	backwardLayer      *layer
}

func (l *layer) init(numberOfNodes int, lastLayer *layer) {
	for i := 0; i < numberOfNodes; i++ {
		edges := make([]edge, 0, 0)
		if lastLayer != nil {
			for j := 0; j < len(lastLayer.nodes); j++ {
				newEdge := edge{
					weight: helpers.RandomFloat(-1.0, 1.0) * 0.1,
				}
				edges = append(edges, newEdge)
			}
		}
		newNode := node{
			bias:    0.0,
			inEdges: edges,
		}
		l.nodes = append(l.nodes, newNode)
	}
}

func (l *layer) sumOutput(input []float64) []float64 {
	data := make([]float64, len(l.nodes))
	usedNodesCount := int(math.Ceil(1.0-l.dropoutRate) * float64(len(l.nodes)))
	notDropoutNodes := helpers.Permutation(len(l.nodes))[0:usedNodesCount]
	for i := range notDropoutNodes {
		data[i] = l.nodes[i].output(input, l.activationFunction)
	}
	return data
}

func (l *layer) flOutput(input []float64) []float64 {
	for i := range input {
		input[i] = l.nodes[i].flOutput(input[i], l.activationFunction)
	}
	return input
}
