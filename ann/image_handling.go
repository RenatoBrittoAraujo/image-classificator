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
			avgBitmap[y][x] = float64(r+g+b) / 3.0
		}
	}
	filterMap := [][]float64{
		{-1, 1, -1},
		{1, 1, 1},
		{-1, 1, -1},
	}
	filteredImage := filter.Filter(avgBitmap, filterMap)
	pooledImage := pooler.MaxPool(filteredImage, 70)
	featureMap := make([]float64, len(pooledImage)*len(pooledImage[0]))
	i := 0
	for y := 0; y < len(pooledImage); y++ {
		for x := 0; x < len(pooledImage[0]); x++ {
			featureMap[i] = pooledImage[y][x]
			i++
		}
	}
	return featureMap
}
