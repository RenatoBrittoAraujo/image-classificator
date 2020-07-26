package ann

import "math"

const (
	lossfcodeMSE = iota
	lossfcodeMAE
	lossfcodeMBE
	lossfcodeCrossEntropy
)

func lossFunction(lossCode int, outputs []float64, corrects []float64) float64 {
	switch lossCode {
	case lossfcodeMSE:
		return lossfMSE(outputs, corrects)
	case lossfcodeMAE:
		return lossfMAE(outputs, corrects)
	case lossfcodeMBE:
		return lossfMBE(outputs, corrects)
	case lossfcodeCrossEntropy:
		return lossfCrossEntropy(outputs, corrects)
	default:
		panic("Invalid loss function argument")
	}
}

func lossFunctionDerivative(lossCode int, outputs []float64, corrects []float64) float64 {
	switch lossCode {
	case lossfcodeMSE:
		return lossdMSE(outputs, corrects)
	case lossfcodeMAE:
		return lossdMAE(outputs, corrects)
	case lossfcodeMBE:
		return lossdMBE(outputs, corrects)
	case lossfcodeCrossEntropy:
		return lossdCrossEntropy(outputs, corrects)
	default:
		panic("Invalid activation function argument")
	}
}

/* LOSS FUNCTIONS */

func lossfMSE(outputs []float64, corrects []float64) float64 {
	sum := 0.0
	for i := 0; i < len(corrects); i++ {
		sum += (corrects[i] - outputs[i]) * (corrects[i] - outputs[i])
	}
	return sum / float64(len(corrects))
}

func lossfMAE(outputs []float64, corrects []float64) float64 {
	sum := 0.0
	for i := 0; i < len(corrects); i++ {
		sum += math.Abs(corrects[i] - outputs[i])
	}
	return sum / float64(len(corrects))
}

func lossfMBE(outputs []float64, corrects []float64) float64 {
	sum := 0.0
	for i := 0; i < len(corrects); i++ {
		sum += (corrects[i] - outputs[i])
	}
	return sum / float64(len(corrects))
}

func lossfCrossEntropy(outputs []float64, corrects []float64) float64 {
	sumc := 0.0
	sumo := 0.0
	for i := 0; i < len(corrects); i++ {
		sumc += corrects[i]
	}
	for i := 0; i < len(outputs); i++ {
		sumo += outputs[i]
	}
	return -(sumc*math.Log(sumo) + (1-sumc)*math.Log(1-sumo))
}

/* LOSS FUNCTIONS DERIVATIVES */

func lossdMSE(outputs []float64, corrects []float64) float64 {
	sum := 0.0
	for i := 0; i < len(corrects); i++ {
		sum += (corrects[i] - outputs[i]) * (corrects[i] - outputs[i])
	}
	return sum / float64(len(corrects))
}

func lossdMAE(outputs []float64, corrects []float64) float64 {
	sum := 0.0
	for i := 0; i < len(corrects); i++ {
		sum += math.Abs(corrects[i] - outputs[i])
	}
	return sum / float64(len(corrects))
}

func lossdMBE(outputs []float64, corrects []float64) float64 {
	sum := 0.0
	for i := 0; i < len(corrects); i++ {
		sum += (corrects[i] - outputs[i])
	}
	return sum / float64(len(corrects))
}

func lossdCrossEntropy(outputs []float64, corrects []float64) float64 {
	sumc := 0.0
	sumo := 0.0
	for i := 0; i < len(corrects); i++ {
		sumc += corrects[i]
	}
	for i := 0; i < len(outputs); i++ {
		sumo += outputs[i]
	}
	return -(sumc*math.Log(sumo) + (1-sumc)*math.Log(1-sumo))
}
