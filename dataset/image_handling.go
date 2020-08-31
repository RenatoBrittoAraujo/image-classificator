package dataset

import (
	"image"
	"math"

	"github.com/renatobrittoaraujo/img-classificator/filter"
	"github.com/renatobrittoaraujo/img-classificator/pooler"
)

const (
	imageFilter = iota
	imagePool
)

// Creates a two dimensional filter for image conversion
func CreateFilter(filter [][]float64) ImageConversion {
	return ImageConversion{
		conversionType: imageFilter,
		filter:         filter,
	}
}

// Creates a pooler for image conversion
func CreatePooler(poolSize int) ImageConversion {
	return ImageConversion{
		conversionType: imagePool,
		poolSize:       poolSize,
	}
}

type ImageConversion struct {
	conversionType int
	filter         [][]float64
	poolSize       int
}

func FeatureMap(image image.Image, conversions []ImageConversion) []float64 {
	red, green, blue := imageToColorBitmaps(image)
	for _, conversion := range conversions {
		switch conversion.conversionType {
		case imagePool:
			red = pooler.AveragePool(red, conversion.poolSize)
			green = pooler.AveragePool(green, conversion.poolSize)
			blue = pooler.AveragePool(blue, conversion.poolSize)
		case imageFilter:
			red = filter.Filter(red, conversion.filter)
			green = filter.Filter(green, conversion.filter)
			blue = filter.Filter(blue, conversion.filter)
		default:
			panic("Unknown conversion type")
		}
	}
	imageLength := len(red) * len(red[0])
	featureMap := make([]float64, imageLength*3)
	i := 0
	for y := 0; y < len(red); y++ {
		for x := 0; x < len(red[0]); x++ {
			featureMap[i] = red[y][x]
			featureMap[i+imageLength] = green[y][x]
			featureMap[i+imageLength*2] = blue[y][x]
			i++
		}
	}
	return normalize(featureMap)
}

func imageToColorBitmaps(image image.Image) (red [][]float64, green [][]float64, blue [][]float64) {
	red = make([][]float64, image.Bounds().Dy())
	green = make([][]float64, image.Bounds().Dy())
	blue = make([][]float64, image.Bounds().Dy())
	for i := range red {
		red[i] = make([]float64, image.Bounds().Dx())
		green[i] = make([]float64, image.Bounds().Dx())
		blue[i] = make([]float64, image.Bounds().Dx())
	}
	for y := 0; y < image.Bounds().Dy(); y++ {
		for x := 0; x < image.Bounds().Dx(); x++ {
			r, g, b, _ := image.At(x, y).RGBA()
			red[y][x] = float64(r)
			green[y][x] = float64(g)
			blue[y][x] = float64(b)
		}
	}
	return red, green, blue
}

func normalize(input []float64) []float64 {
	normal := 0.0
	for _, v := range input {
		normal += v * v
	}
	if normal == 0.0 {
		return input
	}
	normal = math.Sqrt(normal)
	for i := range input {
		input[i] = input[i] / normal
	}
	return input
}
