import sys
import getopt
import parser1

def conllu(fp):
    returnList = []
    for line in fp.readlines():
        if(line[0] == "#"):
            continue
        wordList = line.split()
        returnList.append(wordList)

        if not wordList:
            temp_return = returnList
            returnList=[]
            yield temp_return

def trees(fp,me):

    for tree in conllu(fp):
        pos = list()
        word = list()
        label = list()
        Head = list()
        bigList = list()

        word.append('<ROOT>')
        pos.append('<ROOT>')
        label.append('<ROOT>')
        Head.append(0)
        
        if len(tree) > 1:
            for tokens in tree:
                if len(tokens)>0:
                    word.append(tokens[1])
                    pos.append(tokens[3])
                    label.append(tokens[7])
                    if tokens[6]=='_':
                        continue
                    Head.append(int(tokens[6]))
    
            bigList.append(word)
            bigList.append(pos)
            bigList.append(label)
            bigList.append(Head)

        else:
            bigList.append(["<End>"])
            bigList.append(["<End>"])
            bigList.append(["<End>"])
            bigList.append(0)

        yield bigList



def evaluate(train_file, test_file):
    n_examples = None   # Set to None to train on all examples

    with open(train_file,encoding="utf-8") as fp:
        for i, (words, gold_tags, gold_arclabels, gold_tree) in enumerate(trees(fp,1)):
            if(words[0] == "<ROOT>"):
                parser1.update(words, gold_tags, gold_arclabels, gold_tree)
                #print("\rUpdated with sentence #{}".format(i))
                if n_examples and i >= n_examples:
                    print("Finished training")
                    break
            else:
                print("Finished training")
                break
    

