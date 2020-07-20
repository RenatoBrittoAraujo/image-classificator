package pooler

import "github.com/renatobrittoaraujo/img-classificator/helpers"

func AveragePool(image [][]float64, gridCut int) [][]float64 {
	dy := len(image)
	dx := len(image[0])
	newImage := make([][]float64, dy/gridCut)
	for i := 0; i < len(newImage); i++ {
		newImage[i] = make([]float64, dx/gridCut)
	}
	for y := 0; y < dy/gridCut; y++ {
		for x := 0; x < dx/gridCut; x++ {
			sum := 0.0
			for vy := y * gridCut; vy < helpers.IMin((y+1)*gridCut, dy); vy++ {
				for vx := x * gridCut; vx < helpers.IMin((x+1)*gridCut, dx); vx++ {
					sum += float64(image[vy][vx])
				}
			}
			newImage[y][x] = sum / float64(gridCut*gridCut)
		}
	}
	return newImage
}
