package ann

import (
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
					weight: helpers.RandomFloat(-1.0, 1.0),
				}
				edges = append(edges, newEdge)
			}
		}
		newNode := node{
			bias:    helpers.RandomFloat(-1.0, 1.0),
			inEdges: edges,
		}
		l.nodes = append(l.nodes, newNode)
	}
}

// func (l *layer) getTrainingOutput(input []float64) []float64 {

// }

// func (l *layer) getTestingOutput(input []float64) []float64 {

// }
