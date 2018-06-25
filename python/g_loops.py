class Loops:

    def iterateBygivenNumber(self,dictionary:dict,item:int):
        del dictionary[item]
        print(dictionary)


    def dictStructure(self):
        dictionary = {
            1: [1, 2, 3, 4, 5],
            2: [2, 3, 4, 5],
            3: [3, 4, 5],
            3: [3, 4, 5,6]
        }

        for item in dictionary:
            print(item, '  ', dictionary[item])

        # Loops.iterateBygivenNumber(self,dictionary,3)




Loops.dictStructure(0)
