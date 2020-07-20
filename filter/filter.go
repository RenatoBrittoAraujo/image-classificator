package filter

func Filter(image [][]float64, filter [][]float64) [][]float64 {
	fx := len(filter[0])
	fy := len(filter)
	if fx&1+fy&1 != 2 {
		panic("Filter dimesions must be odd")
	}
	filtered := make([][]float64, len(image)-fy/2)
	for i := range filtered {
		filtered[i] = make([]float64, len(image[0])-fx/2)
	}
	for y := 0; y < len(filtered); y++ {
		for x := 0; x < len(filtered[0]); x++ {
			sum := 0.0
			for vy := y; vy < fy+y; vy++ {
				for vx := x; vx < fx+x; vx++ {
					sum += image[y][x] * filter[vy-y][vx-x]
				}
			}
			filtered[y][x] = sum
		}
	}
	return filtered
}
