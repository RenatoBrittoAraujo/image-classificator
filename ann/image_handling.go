package ann

import (
	"image"

	"github.com/renatobrittoaraujo/img-classificator/filter"
	"github.com/renatobrittoaraujo/img-classificator/pooler"
)

const (
	imageFilter = iota
	imagePool
)

// Creates a two dimensional filter for image conversion
func CreateFilter(filter [][]float64) imageConversion {
	return imageConversion{
		conversionType: imageFilter,
		filter:         filter,
	}
}

// Creates a pooler for image conversion
func CreatePooler(poolSize int) imageConversion {
	return imageConversion{
		conversionType: imagePool,
		poolSize:       poolSize,
	}
}

type imageConversion struct {
	conversionType int
	filter         [][]float64
	poolSize       int
}

func (a *Ann) convertImage(image image.Image, conversions []imageConversion) []float64 {
	redBitmap := make([][]float64, image.Bounds().Dy())
	greenBitmap := make([][]float64, image.Bounds().Dy())
	blueBitmap := make([][]float64, image.Bounds().Dy())
	for i := range redBitmap {
		redBitmap[i] = make([]float64, image.Bounds().Dx())
		greenBitmap[i] = make([]float64, image.Bounds().Dx())
		blueBitmap[i] = make([]float64, image.Bounds().Dx())
	}
	for y := 0; y < image.Bounds().Dy(); y++ {
		for x := 0; x < image.Bounds().Dx(); x++ {
			r, g, b, _ := image.At(x, y).RGBA()
			redBitmap[y][x] = float64(r)
			greenBitmap[y][x] = float64(g)
			blueBitmap[y][x] = float64(b)
		}
	}
	for _, conversion := range conversions {
		switch conversion.conversionType {
		case imagePool:
			redBitmap = pooler.AveragePool(redBitmap, conversion.poolSize)
			greenBitmap = pooler.AveragePool(greenBitmap, conversion.poolSize)
			blueBitmap = pooler.AveragePool(blueBitmap, conversion.poolSize)
		case imageFilter:
			redBitmap = filter.Filter(redBitmap, conversion.filter)
			greenBitmap = filter.Filter(greenBitmap, conversion.filter)
			blueBitmap = filter.Filter(blueBitmap, conversion.filter)
		default:
			panic("Unknown conversion type")
		}
	}
	imageLength := len(redBitmap) * len(redBitmap[0])
	featureMap := make([]float64, imageLength*3)
	i := 0
	for y := 0; y < len(redBitmap); y++ {
		for x := 0; x < len(redBitmap[0]); x++ {
			featureMap[i] = redBitmap[y][x]
			featureMap[i+imageLength] = greenBitmap[y][x]
			featureMap[i+imageLength*2] = blueBitmap[y][x]
			i++
		}
	}
	return featureMap
}
