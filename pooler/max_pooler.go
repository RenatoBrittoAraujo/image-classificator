package pooler

import (
	"math"

	"github.com/renatobrittoaraujo/img-classificator/helpers"
)

func MaxPool(image [][]float64, gridCut int) [][]float64 {
	dy := len(image)
	dx := len(image[0])
	newImage := make([][]float64, dy/gridCut)
	for i := 0; i < len(newImage); i++ {
		newImage[i] = make([]float64, dx/gridCut)
	}
	for y := 0; y < dy/gridCut; y++ {
		for x := 0; x < dx/gridCut; x++ {
			val := 0.0
			for vy := y * gridCut; vy < helpers.IMin((y+1)*gridCut, dy); vy++ {
				for vx := x * gridCut; vx < helpers.IMin((x+1)*gridCut, dx); vx++ {
					val = math.Max(val, image[vy][vx])
				}
			}
			newImage[y][x] = val
		}
	}
	return newImage
}
