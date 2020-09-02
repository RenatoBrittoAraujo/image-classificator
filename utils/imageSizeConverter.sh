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
echo $files
for item in $files
do
	if [[ $item == *.png ]] || [[ $item == *.jpg ]]
	then
		newName=$item
		if [[ $item == *jpg ]]
		then
			newName=${newName/jpg/png}
			if [[ $newName == *jpg ]]
			then
				echo "WHAT?"
			fi
		fi
		echo "converting $inputFolder$item... to $sizex$size"
		echo convert $inputFolder$item -resize $size'x'$size'!' $inputFolder$newName
		convert $inputFolder$item -resize $size'x'$size'!' $inputFolder$newName
	fi
done
