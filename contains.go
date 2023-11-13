package artSearch

import (
	"errors"
	"io"
	"os"
	"strings"
)

// Contains проверяет содержание подстроки в файле.
func Contains(filePath, substring string) (bool, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return false, errors.New("new error: file not found")
	}
	// Чтение содержимого файла
	content, err := io.ReadAll(file)
	if err != nil {
		return false, err
	}

	// Поиск подстроки в содержимом файла
	contains := strings.Contains(string(content), substring)

	return contains, nil
}
