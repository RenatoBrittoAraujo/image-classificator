package ann

import (
	"image"

	"github.com/renatobrittoaraujo/img-classificator/filter"
	"github.com/renatobrittoaraujo/img-classificator/pooler"
)

func (a *Ann) convertImage(image image.Image) []float64 {
	avgBitmap := make([][]float64, image.Bounds().Dy())
	for i := range avgBitmap {
		avgBitmap[i] = make([]float64, image.Bounds().Dx())
	}
	for y := 0; y < image.Bounds().Dy(); y++ {
		for x := 0; x < image.Bounds().Dx(); x++ {
			r, g, b, _ := image.At(x, y).RGBA()
			avgBitmap[y][x] = float64(r+g+b) / (3.0 * 65535.0)
			// fmt.Println(r, g, b, avgBitmap[y][x])
			// tm, _ := time.ParseDuration("1s")
			// time.Sleep(tm)
		}
	}
	filterMap := [][]float64{
		{0, 0, 0},
		{0, 1, 0},
		{0, 0, 0},
	}
	filteredImage := filter.Filter(avgBitmap, filterMap)
	pooledImage := pooler.AveragePool(filteredImage, 70)
	featureMap := make([]float64, len(pooledImage)*len(pooledImage[0]))
	i := 0
	for y := 0; y < len(pooledImage); y++ {
		for x := 0; x < len(pooledImage[0]); x++ {
			featureMap[i] = pooledImage[y][x]
			i++
		}
	}
	// DATA SMOOTHING
	// min := +10000000000.0
	// max := -10000000000.0
	// for i := range featureMap {
	// 	if featureMap[i] < min {
	// 		min = featureMap[i]
	// 	}
	// 	if featureMap[i] > max {
	// 		max = featureMap[i]
	// 	}
	// }
	// min = 0
	// max = 1000
	// if min == max {
	// 	max = min + 1
	// }
	// for i := range featureMap {
	// 	featureMap[i] = (featureMap[i]-min)*2.0/(max-min) - 1.0
	// }
	return featureMap
}
