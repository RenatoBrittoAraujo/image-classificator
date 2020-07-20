package helpers

import "os"

// FileExists returns true if file in path exists, false if not and panics if path is folder
func FileExists(path string) bool {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	if info.IsDir() {
		panic("Given path for helpers.fileExists is a folder")
	}
	return true
}
