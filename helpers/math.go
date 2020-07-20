package helpers

// IMax returns max of given integers
func IMax(a int, b int) int {
	if a >= b {
		return a
	}
	return b
}

// IMin returns min of given integers
func IMin(a int, b int) int {
	if a >= b {
		return b
	}
	return a
}
