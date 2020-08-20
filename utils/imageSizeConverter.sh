# Para usar, instale isso: `sudo apt-get install imagemagick`
echo "pasta de input:"
read inputFolder
file
if [[ $inputFolder == "." ]]
then
	inputFolder=""
else
	inputFolder="$inputFolder/"
fi
echo "tamanho:"
read size
files=$(ls $inputFolder)
for item in $files
do
	if [[ $item == *.png ]]
	then
		echo "converting $inputFolder$item... to $sizex$size"
		echo convert $inputFolder$item -resize $size'x'$size'!' $inputFolder$item
		convert $inputFolder$item -resize $size'x'$size'!' $inputFolder$item
	fi
done
