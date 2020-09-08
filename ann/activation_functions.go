package ann

import "math"

const (
	actfcodeCUBE = iota
	actfcodeELU
	actfcodeIDENTITY
	actfcodeLEAKYRELU
	actfcodeRATIONALTANH
	actfcodeRELU
	actfcodeSIGMOID
	actfcodeSOFTPLUS
	actfcodeTANH
)

func activationFunction(acivationCode int, data float64) float64 {
	switch acivationCode {
	case actfcodeCUBE:
		return actfCube(data)
	case actfcodeELU:
		return actfElu(data)
	case actfcodeIDENTITY:
		return actfIdentity(data)
	case actfcodeRELU:
		return actfRelu(data)
	case actfcodeSIGMOID:
		return actfSigmoid(data)
	case actfcodeSOFTPLUS:
		return actfSoftplus(data)
	case actfcodeTANH:
		return actfTanh(data)
	default:
		panic("Invalid activation function argument")
	}
}

func activationFunctionDerivative(acivationCode int, data float64) float64 {
	switch acivationCode {
	case actfcodeCUBE:
		return actdCube(data)
	case actfcodeELU:
		return actdElu(data)
	case actfcodeIDENTITY:
		return actdIdentity(data)
	case actfcodeRELU:
		return actdRelu(data)
	case actfcodeSIGMOID:
		return actdSigmoid(data)
	case actfcodeSOFTPLUS:
		return actdSoftplus(data)
	case actfcodeTANH:
		return actdTanh(data)
	default:
		panic("Invalid activation function argument")
	}
}

/* ACTIVATION FUNCTIONS */

func actfCube(val float64) float64 {
	return math.Cbrt(val)
}

func actfElu(val float64) float64 {
	if val > 0 {
		return val
	}
	return math.Exp(val) - 1.0
}

func actfIdentity(val float64) float64 {
	return val
}

func actfRelu(val float64) float64 {
	if val < 0 {
		return 0
	}
	return val
}

func actfSigmoid(val float64) float64 {
	return 1.0 / (1.0 + math.Exp(-val))
}

func actfSoftplus(val float64) float64 {
	return math.Log(1 + math.Exp(val))
}

func actfTanh(val float64) float64 {
	return math.Tanh(val)
}

/* ACTIVATION FUNCTIONS DERIVATIVES */

func actdCube(val float64) float64 {
	return 1.0 / (3.0 * math.Pow(val, 2.0/3.0))
}

func actdElu(val float64) float64 {
	if val > 0 {
		return 1
	}
	return math.Exp(val)
}

func actdIdentity(val float64) float64 {
	return 1
}

func actdRelu(val float64) float64 {
	if val > 0 {
		return 1
	}
	return 0
}

func actdSigmoid(val float64) float64 {
	return actfSigmoid(val) * (1.0 - actfSigmoid(val))
}

func actdSoftplus(val float64) float64 {
	return math.Exp(val) / (1 + math.Exp(val))
}

func actdTanh(val float64) float64 {
	return (1 - math.Tanh(val)*math.Tanh(val))
}
