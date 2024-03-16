string = 'apple,banana;orange|watermelon'
splitted = string.split(',') + string.split(';') + string.split('|')

print(splitted)