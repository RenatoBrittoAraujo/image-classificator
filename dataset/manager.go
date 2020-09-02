package dataset

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
)

// GetDataset returns a slice containing the jpg images of the dataset as bitmaps
func GetDataset(datasetName string) []image.Image {
	var dataset []image.Image
	var fileNames []string
	err := filepath.Walk("dataset/"+datasetName, func(path string, info os.FileInfo, err error) error {
		if path == "dataset/"+datasetName {
			return nil
		}
		fileNames = append(fileNames, path)
		return nil
	})
	if err != nil {
		fmt.Println("Failure at finding dataset " + datasetName + ": " + err.Error())
		panic(err)
	}
	for _, file := range fileNames {
		imageFile, err := os.Open(file)
		if err != nil {
			fmt.Println("Failure at opening file " + file + ": " + err.Error())
			panic(err)
		}
		loadedImage, err := png.Decode(imageFile)
		if err != nil {
			loadedImage, err = jpeg.Decode(imageFile)
			if err != nil {
				fmt.Println("Failure at decoding " + file + ": " + err.Error())
				panic(err)
			}
		}
		dataset = append(dataset, loadedImage)
		imageFile.Close()
	}
	return dataset
}

func GetFeatureMaps(datasetName string, conversions []ImageConversion) [][]float64 {
	dataset := GetDataset(datasetName)
	featureMaps := make([][]float64, len(dataset))
	for i := range dataset {
		featureMaps[i] = FeatureMap(dataset[i], conversions)
	}
	return featureMaps
}
