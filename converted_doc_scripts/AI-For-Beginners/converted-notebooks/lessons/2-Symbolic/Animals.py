from jet.logger import logger
from pyknow import *
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Implementing an Animal Expert System

An example from [AI for Beginners Curriculum](http://github.com/microsoft/ai-for-beginners).

In this sample, we will implement a simple knowledge-based system to determine an animal based on some physical characteristics. The system can be represented by the following AND-OR tree (this is a part of the whole tree, we can easily add some more rules):

![](images/AND-OR-Tree.png)

## Our own expert systems shell with backward inference

Let's try to define a simple language for knowledge representation based on production rules. We will use Python classes as keywords to define rules. There would be essentially 3 types of classes:
* `Ask` represents a question that needs to be asked to the user. It contains the set of possible answers.
* `If` represents a rule, and it is just a syntactic sugar to store the content of the rule
* `AND`/`OR` are classes to represent AND/OR branches of the tree. They just store the list of arguments inside. To simplify code, all functionality is defined in the parent class `Content`
"""
logger.info("# Implementing an Animal Expert System")

class Ask():
    def __init__(self,choices=['y','n']):
        self.choices = choices
    def ask(self):
        if max([len(x) for x in self.choices])>1:
            for i,x in enumerate(self.choices):
                logger.debug("{0}. {1}".format(i,x),flush=True)
            x = int(input())
            return self.choices[x]
        else:
            logger.debug("/".join(self.choices),flush=True)
            return input()

class Content():
    def __init__(self,x):
        self.x=x

class If(Content):
    pass

class AND(Content):
    pass

class OR(Content):
    pass

"""
In our system, working memory would contain the list of **facts** as **attribute-value pairs**. The knowledgebase can be defined as one big dictionary that maps actions (new facts that should be inserted into working memory) to conditions, expressed as AND-OR expressions. Also, some facts can be `Ask`-ed.
"""
logger.info("In our system, working memory would contain the list of **facts** as **attribute-value pairs**. The knowledgebase can be defined as one big dictionary that maps actions (new facts that should be inserted into working memory) to conditions, expressed as AND-OR expressions. Also, some facts can be `Ask`-ed.")

rules = {
    'default': Ask(['y','n']),
    'color' : Ask(['red-brown','black and white','other']),
    'pattern' : Ask(['dark stripes','dark spots']),
    'mammal': If(OR(['hair','gives milk'])),
    'carnivor': If(OR([AND(['sharp teeth','claws','forward-looking eyes']),'eats meat'])),
    'ungulate': If(['mammal',OR(['has hooves','chews cud'])]),
    'bird': If(OR(['feathers',AND(['flies','lies eggs'])])),
    'animal:monkey' : If(['mammal','carnivor','color:red-brown','pattern:dark spots']),
    'animal:tiger' : If(['mammal','carnivor','color:red-brown','pattern:dark stripes']),
    'animal:giraffe' : If(['ungulate','long neck','long legs','pattern:dark spots']),
    'animal:zebra' : If(['ungulate','pattern:dark stripes']),
    'animal:ostrich' : If(['bird','long nech','color:black and white','cannot fly']),
    'animal:pinguin' : If(['bird','swims','color:black and white','cannot fly']),
    'animal:albatross' : If(['bird','flies well'])
}

"""
To perform the backward inference, we will define `Knowledgebase` class. It will contain:
* Working `memory` - a dictionary that maps attributes to values
* Knowledgebase `rules` in the format as defined above

Two main methods are:
* `get` to obtain the value of an attribute, performing inference if necessary. For example, `get('color')` would get the value of a color slot (it will ask if necessary, and store the value for later usage in the working memory). If we ask `get('color:blue')`, it will ask for a color, and then return `y`/`n` value depending on the color.
* `eval` performs the actual inference, i.e. traverses AND/OR tree, evaluates sub-goals, etc.
"""
logger.info("To perform the backward inference, we will define `Knowledgebase` class. It will contain:")

class KnowledgeBase():
    def __init__(self,rules):
        self.rules = rules
        self.memory = {}

    def get(self,name):
        if ':' in name:
            k,v = name.split(':')
            vv = self.get(k)
            return 'y' if v==vv else 'n'
        if name in self.memory.keys():
            return self.memory[name]
        for fld in self.rules.keys():
            if fld==name or fld.startswith(name+":"):
                value = 'y' if fld==name else fld.split(':')[1]
                res = self.eval(self.rules[fld],field=name)
                if res!='y' and res!='n' and value=='y':
                    self.memory[name] = res
                    return res
                if res=='y':
                    self.memory[name] = value
                    return value
        res = self.eval(self.rules['default'],field=name)
        self.memory[name]=res
        return res

    def eval(self,expr,field=None):
        if isinstance(expr,Ask):
            logger.debug(field)
            return expr.ask()
        elif isinstance(expr,If):
            return self.eval(expr.x)
        elif isinstance(expr,AND) or isinstance(expr,list):
            expr = expr.x if isinstance(expr,AND) else expr
            for x in expr:
                if self.eval(x)=='n':
                    return 'n'
            return 'y'
        elif isinstance(expr,OR):
            for x in expr.x:
                if self.eval(x)=='y':
                    return 'y'
            return 'n'
        elif isinstance(expr,str):
            return self.get(expr)
        else:
            logger.debug("Unknown expr: {}".format(expr))

"""
Now let's define our animal knowledgebase and perform the consultation. Note that this call will ask you questions. You can answer by typing `y`/`n` for yes-no questions, or by specifying number (0..N) for questions with longer multiple-choice answers.
"""
logger.info("Now let's define our animal knowledgebase and perform the consultation. Note that this call will ask you questions. You can answer by typing `y`/`n` for yes-no questions, or by specifying number (0..N) for questions with longer multiple-choice answers.")

kb = KnowledgeBase(rules)
kb.get('animal')

"""
## Using PyKnow for Forward Inference

In the next example, we will try to implement forward inference using one of the libraries for knowledge representation, [PyKnow](https://github.com/buguroo/pyknow/). **PyKnow** is a library for creating forward inference systems in Python, which is designed to be similar to classical old system [CLIPS](http://www.clipsrules.net/index.html). 

We could have also implemented forward chaining ourselves without many problems, but naive implementations are usually not very efficient. For more effective rule matching a special algorithm [Rete](https://en.wikipedia.org/wiki/Rete_algorithm) is used.
"""
logger.info("## Using PyKnow for Forward Inference")

# !{sys.executable} -m pip install git+https://github.com/buguroo/pyknow/


"""
We will define our system as a class that subclasses `KnowledgeEngine`. Each rule is defined by a separate function with `@Rule` annotation, which specifies when the rule should fire. Inside the rule, we can add new facts using `declare` function, and adding those facts will result in some more rules being called by forward inference engine.
"""
logger.info("We will define our system as a class that subclasses `KnowledgeEngine`. Each rule is defined by a separate function with `@Rule` annotation, which specifies when the rule should fire. Inside the rule, we can add new facts using `declare` function, and adding those facts will result in some more rules being called by forward inference engine.")

class Animals(KnowledgeEngine):
    @Rule(OR(
           AND(Fact('sharp teeth'),Fact('claws'),Fact('forward looking eyes')),
           Fact('eats meat')))
    def cornivor(self):
        self.declare(Fact('carnivor'))

    @Rule(OR(Fact('hair'),Fact('gives milk')))
    def mammal(self):
        self.declare(Fact('mammal'))

    @Rule(Fact('mammal'),
          OR(Fact('has hooves'),Fact('chews cud')))
    def hooves(self):
        self.declare('ungulate')

    @Rule(OR(Fact('feathers'),AND(Fact('flies'),Fact('lays eggs'))))
    def bird(self):
        self.declare('bird')

    @Rule(Fact('mammal'),Fact('carnivor'),
          Fact(color='red-brown'),
          Fact(pattern='dark spots'))
    def monkey(self):
        self.declare(Fact(animal='monkey'))

    @Rule(Fact('mammal'),Fact('carnivor'),
          Fact(color='red-brown'),
          Fact(pattern='dark stripes'))
    def tiger(self):
        self.declare(Fact(animal='tiger'))

    @Rule(Fact('ungulate'),
          Fact('long neck'),
          Fact('long legs'),
          Fact(pattern='dark spots'))
    def giraffe(self):
        self.declare(Fact(animal='giraffe'))

    @Rule(Fact('ungulate'),
          Fact(pattern='dark stripes'))
    def zebra(self):
        self.declare(Fact(animal='zebra'))

    @Rule(Fact('bird'),
          Fact('long neck'),
          Fact('cannot fly'),
          Fact(color='black and white'))
    def straus(self):
        self.declare(Fact(animal='ostrich'))

    @Rule(Fact('bird'),
          Fact('swims'),
          Fact('cannot fly'),
          Fact(color='black and white'))
    def pinguin(self):
        self.declare(Fact(animal='pinguin'))

    @Rule(Fact('bird'),
          Fact('flies well'))
    def albatros(self):
        self.declare(Fact(animal='albatross'))

    @Rule(Fact(animal=MATCH.a))
    def print_result(self,a):
          logger.debug('Animal is {}'.format(a))

    def factz(self,l):
        for x in l:
            self.declare(x)

"""
Once we have defined a knowledgebase, we populate our working memory with some initial facts, and then call `run()` method to perform the inference. You can see as a result that new inferred facts are added to the working memory, including the final fact about the animal (if we set up all the initial facts correctly).
"""
logger.info("Once we have defined a knowledgebase, we populate our working memory with some initial facts, and then call `run()` method to perform the inference. You can see as a result that new inferred facts are added to the working memory, including the final fact about the animal (if we set up all the initial facts correctly).")

ex1 = Animals()
ex1.reset()
ex1.factz([
    Fact(color='red-brown'),
    Fact(pattern='dark stripes'),
    Fact('sharp teeth'),
    Fact('claws'),
    Fact('forward looking eyes'),
    Fact('gives milk')])
ex1.run()
ex1.facts

logger.info("\n\n[DONE]", bright=True)