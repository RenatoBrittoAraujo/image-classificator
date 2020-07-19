package ann

type layer struct {
	nodes              []node
	activationFunction int
	dropoutRate        float64
	fowardLayer        *layer
	backwardLayer      *layer
}

func FowardsPropagation(input []float64) []float64 {

}

func BackwardsPropagation(input []float64) []float64 {

}
