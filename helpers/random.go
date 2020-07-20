package helpers

import (
	"math/rand"
)

// RandomFloat returns random float64 inside interval [min,max)
func RandomFloat(min float64, max float64) float64 {
	return rand.Float64()*(max-min) + min
}

// Permutation retuns a permutation of integer of given input
func Permutation(size int) []int {
	output := make([]int, size, size)
	for i := 0; i < size; i++ {
		output[i] = i
	}
	for i := 0; i < size; i++ {
		swapPos := RandomInteger(0, size)
		output[i] ^= output[swapPos]
		output[swapPos] ^= output[i]
		output[i] ^= output[swapPos]
	}
	return output
}

// RandomInteger returns random int inside interval [min,max]
func RandomInteger(min int, max int) int {
	return rand.Int()%(min+max) + min
}
