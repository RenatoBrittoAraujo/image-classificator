package ann

/*
	Functions involving actions you can do with of the
	setup and configured neural network
*/

import (
	"fmt"

	"github.com/renatobrittoaraujo/img-classificator/helpers"
)

// Train trains the ann with given inputs
// NOTE: the order of the datasets matter
func (a *Ann) Train(dataset1 [][]float64, dataset2 [][]float64) {
	order := helpers.Permutation(len(dataset1) + len(dataset2))
	okcases := 0
	notokcases := 0
	dataset1misses := 0
	dataset2misses := 0
	sum1 := 0.0
	sum2 := 0.0
	for i := range order {
		var featureMap []float64
		expected := []float64{0}
		datasetIndex := 0
		if order[i] >= len(dataset1) {
			datasetIndex = 1
		}
		if datasetIndex == 1 {
			featureMap = dataset2[order[i]-len(dataset1)]
			expected = []float64{-1.0}
		} else {
			featureMap = dataset1[order[i]]
			expected = []float64{1.0}
		}
		ok, res := a.trainCase(featureMap, expected)
		if datasetIndex == 0 {
			sum1 += res[0]
		} else {
			sum2 += res[0]
		}
		if ok {
			okcases++
		} else {
			notokcases++
			if datasetIndex == 0 {
				dataset1misses++
			} else {
				dataset2misses++
			}
		}
	}
	fmt.Println("_______________________________________________________________________________")
	fmt.Println("| DATASET 1 MISSES:", dataset1misses, "\t\t| DATASET 2 MISSES:", dataset2misses)
	fmt.Println("| AVG OUTPUT 1:", sum1/float64(len(dataset1)), "\t| AVG OUTPUT 2:", sum2/float64(len(dataset2)))
	fmt.Println("| OK:", okcases, "\t\t\t\t| NOTOK:", notokcases)
	fmt.Println("| PRECISION:", float64(okcases)/float64(okcases+notokcases))
	fmt.Println("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
}

/* Private functions */
func (a *Ann) trainCase(input []float64, expected []float64) (flg bool, res []float64) {
	res = a.FowardProgation(input)
	a.BackPropagation(expected)
	if res[0] > 0.0 && expected[0] == 1 {
		flg = true
		return
	}
	if res[0] <= 0.0 && expected[0] == -1.0 {
		flg = true
		return
	}
	flg = false
	return
}

func (a *Ann) Test(dataset1 [][]float64, dataset2 [][]float64) {
	acc1 := 0
	acc2 := 0
	err1 := 0
	err2 := 0
	for i := 0; i < len(dataset1)+len(dataset2); i++ {
		testCase := []float64{}
		expected := []float64{}
		if i < len(dataset1) {
			testCase = dataset1[i]
			expected = []float64{-1.0}
		} else {
			testCase = dataset2[i-len(dataset1)]
			expected = []float64{1.0}
		}
		res := a.FowardProgation(testCase)
		if res[0] >= 0 && expected[0] == 1.0 {
			acc1++
		} else if res[0] < 0 && expected[0] == -1.0 {
			acc2++
		} else if res[0] >= 0 && expected[0] == -1.0 {
			err2++
		} else {
			err1++
		}
	}
	fmt.Println("CONFUSION MATRIX:")
	fmt.Println("\tGroup 1\tGroup 2")
	fmt.Println("ACC\t", acc1, "\t", acc2)
	fmt.Println("ERR\t", err1, "\t", err2)
}
