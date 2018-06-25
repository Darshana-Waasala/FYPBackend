
def say_hello():
    print('hello!!!')


def print_max(x,y):
    if x == y:
        print(x,' is equal to ',y)
    elif x < y:
        print(x,' is less than ',y)
        print_max(y, ' is maximum')
    else:
        print(x, ' is grater than ',y)
        print_max(x,' is maximum')

# ####################################
# local variables vs global variables
x = 50


def function_local(x):
    print('the global value of x is : ',x)
    x =2
    print('the local value of x changed to: ',x)


def function2_global():
    global x # accessing the global varible
    print('the global value of x is : ',x)
    x = 2
    print('the global value changed as : ',x)


def local_global_function():
    print('initial value of x : ',x)
    function_local(x)
    print('value of x after local varible change : ',x)
    function2_global()
    print('value of x aftre global variable chage : ',x)

# ####################################
# default arguments
def say(message,times=1):
    print(message*times)

def default_arguments():
    say('hello')
    say('world',5)

# ####################################
# return statement
def getMaxumum(a,b):
    if a<b:
        return b
    elif a>b:
        return a
    else:
        return 'the numbers are equal'

