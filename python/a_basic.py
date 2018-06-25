print("hello world")

# variables
singleQuotoString = 'single quotations'
doubleQuotoString = "double quotations"
multipleLineString = ''' this is a 
                            multiple 
                            line 
                            string'''

print('\'\' : ' + singleQuotoString + '\n\"\" : '+doubleQuotoString + '\n \'\'\'...\'\'\' : '+ multipleLineString)
print('\n STRINGS ARE IMMUTABLE\n')

age = 20
name = 'Swaroop'

######## the usage of format() method

print('{0} was {1} years old when he wrote this book'.format(name, age))
print('Why is {0} playing with that python?'.format(name))

print('{0:.3f}'.format(1.0/3))
# fill with underscores (_) with the text centered
# (^) to 11 width '___hello___'
print('{0:_^11}'.format('hello'))
# keyword-based 'Swaroop wrote A Byte of Python'
print('{name} wrote {book}'.format(name='Swaroop', book='A Byte of Python'))

__version__ = '0.1'
