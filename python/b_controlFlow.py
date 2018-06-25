number = 23
guess = int(input('enter a guess number: '))
running = True

# if statement
if guess == number:
    print('you have successfully guessed the number')
elif guess < number:
    print('the number is grater than {guess}'.format(guess=guess))
elif guess > number:
    print('the number is lesser than {guess}'.format(guess=guess))

# while statement
while running:
    guess = int(input('enter a guess number: '))
    if guess == number:
        print('you have successfully guessed the number')
        running = False
        print('Done')
    elif guess < number:
        print('the number is grater than {guess}'.format(guess=guess))
    elif guess > number:
        print('the number is less than {guess}'.format(guess = guess))


# for...in statement
for i in range(1,5):
    print(i)
else:
    print('loop is over')

# break & continue statements
while True:
    s = input('enter a string')
    if s == 'exit()':
        break
    elif len(s) < 2:
        print(s)
        continue
