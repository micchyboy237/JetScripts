from gedcom.element.family import FamilyElement
from gedcom.element.individual import IndividualElement
from gedcom.parser import Parser
from jet.logger import logger
from owlrl import DeductiveClosure, OWLRL_Extension
import os
import rdflib
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
# Family Relationships Ontology

This example is a part of [AI for Beginners Curriculum](http://github.com/microsoft/ai-for-beginners), and it has been inspired by [this blog post](https://habr.com/post/270857/).

I always find it difficult to remember different relationships between people in a family. In this example, we will take an ontology that defines family relationships, and the actual genealogical tree, and show how we can then perform automatic inference to find all relatives.

### Getting the Genealogical Tree

As an example, we will take genealogical tree of [Romanov Tsar Family](https://en.wikipedia.org/wiki/House_of_Romanov). The most common format for describing family relationships is [GEDCOM](https://en.wikipedia.org/wiki/GEDCOM). We will take Romanov family tree in GEDCOM format:
"""
logger.info("# Family Relationships Ontology")

# !head -15 data/tsars.ged

"""
To use GEDCOM file, we can use `python-gedcom` library:
"""
logger.info("To use GEDCOM file, we can use `python-gedcom` library:")

# !{sys.executable} -m pip install python-gedcom

"""
This library takes away some of the technical problems with file parsing, but it still gives us pretty low-level access to all individuals and families in the tree. Here is how we can parse the file, and show the list of all individuals:
"""
logger.info("This library takes away some of the technical problems with file parsing, but it still gives us pretty low-level access to all individuals and families in the tree. Here is how we can parse the file, and show the list of all individuals:")

g = Parser()
g.parse_file('data/tsars.ged')

d = g.get_element_dictionary()
[ (k,v.get_name()) for k,v in d.items() if isinstance(v,IndividualElement)]

"""
Here is how we can get information about families. Note that is gives us a list of **identifiers**, and we need to convert them to names if we want more clarity:
"""
logger.info("Here is how we can get information about families. Note that is gives us a list of **identifiers**, and we need to convert them to names if we want more clarity:")

d = g.get_element_dictionary()
[ (k,[x.get_value() for x in v.get_child_elements()]) for k,v in d.items() if isinstance(v,FamilyElement)]

"""
### Getting Family Ontology

Next, let's have a look at [family ontology](https://raw.githubusercontent.com/blokhin/genealogical-trees/master/data/header.ttl) defined as a set of Semantic Web triplets. This ontology defines such relationships as `isUncleOf`, `isCousinOf`, and many others. All those relationships are defined in terms of basic predicates `isMotherOf`, `isFatherOf`, `isBrotherOf` and `isSisterOf`. We will use automatic reasoning to deduce all other relationships using the ontology.

Here is a sample definition of `isAuntOf` property, which is defined as a composition of `isSisterOf` and `isParentOf` (*Aunt is a sister of one's parent*).

```
fhkb:isAuntOf a owl:ObjectProperty ;
    rdfs:domain fhkb:Woman ;
    rdfs:range fhkb:Person ;
    owl:propertyChainAxiom ( fhkb:isSisterOf fhkb:isParentOf ) .
```
"""
logger.info("### Getting Family Ontology")

# !head -20 data/onto.ttl

"""
### Constructing Ontology for Inference

For simplicity, we will create one ontology file that will include original rules from family ontology, and facts about individuals from our GEDCOM file. We will go through the GEDCOM file and extract information about families and individuals, and convert them to triplets.
"""
logger.info("### Constructing Ontology for Inference")

# !cp data/onto.ttl .

gedcom_dict = g.get_element_dictionary()
individuals, marriages = {}, {}

def term2id(el):
    return "i" + el.get_pointer().replace('@', '').lower()

out = open("onto.ttl","a")

for k, v in gedcom_dict.items():
    if isinstance(v,IndividualElement):
        children, siblings = set(), set()
        idx = term2id(v)

        title = v.get_name()[0] + " " + v.get_name()[1]
        title = title.replace('"', '').replace('[', '').replace(']', '').replace('(', '').replace(')', '').strip()

        own_families = g.get_families(v, 'FAMS')
        for fam in own_families:
            children |= set(term2id(i) for i in g.get_family_members(fam, "CHIL"))

        parent_families = g.get_families(v, 'FAMC')
        if len(parent_families):
            for member in g.get_family_members(parent_families[0], "CHIL"): # NB adoptive families i.e len(parent_families)>1 are not considered (TODO?)
                if member.get_pointer() == v.get_pointer():
                    continue
                siblings.add(term2id(member))

        if idx in individuals:
            children |= individuals[idx].get('children', set())
            siblings |= individuals[idx].get('siblings', set())
        individuals[idx] = {'sex': v.get_gender().lower(), 'children': children, 'siblings': siblings, 'title': title}

    elif isinstance(v,FamilyElement):
        wife, husb, children = None, None, set()
        children = set(term2id(i) for i in g.get_family_members(v, "CHIL"))

        try:
            wife = g.get_family_members(v, "WIFE")[0]
            wife = term2id(wife)
            if wife in individuals: individuals[wife]['children'] |= children
            else: individuals[wife] = {'children': children}
        except IndexError: pass
        try:
            husb = g.get_family_members(v, "HUSB")[0]
            husb = term2id(husb)
            if husb in individuals: individuals[husb]['children'] |= children
            else: individuals[husb] = {'children': children}
        except IndexError: pass

        if wife and husb: marriages[wife + husb] = (term2id(v), wife, husb)

for idx, val in individuals.items():
    added_terms = ''
    if val['sex'] == 'f':
        parent_predicate, sibl_predicate = "isMotherOf", "isSisterOf"
    else:
        parent_predicate, sibl_predicate = "isFatherOf", "isBrotherOf"
    if len(val['children']):
        added_terms += " ;\n    fhkb:" + parent_predicate + " " + ", ".join(["fhkb:" + i for i in val['children']])
    if len(val['siblings']):
        added_terms += " ;\n    fhkb:" + sibl_predicate + " " + ", ".join(["fhkb:" + i for i in val['siblings']])
    out.write("fhkb:%s a owl:NamedIndividual, owl:Thing%s ;\n    rdfs:label \"%s\" .\n" % (idx, added_terms, val['title']))

for k, v in marriages.items():
    out.write("fhkb:%s a owl:NamedIndividual, owl:Thing ;\n    fhkb:hasFemalePartner fhkb:%s ;\n    fhkb:hasMalePartner fhkb:%s .\n" % v)

out.write("[] a owl:AllDifferent ;\n    owl:distinctMembers (")
for idx in individuals.keys():
    out.write("    fhkb:" + idx)
for k, v in marriages.items():
    out.write("    fhkb:" + v[0])
out.write("    ) .")
out.close()

# !tail onto.ttl

"""
### Doing Inference 

Now we want to be able to use this ontology for inference and for querying. We will use [RDFLib](https://github.com/RDFLib), library for reading RDF Graph in different formats, querying it, etc. 

For logical inference, we will use [OWL-RL](https://github.com/RDFLib/OWL-RL) library, which allows us to build **Closure** of the RDF Graph, i.e. add all possible concepts and relations that can be inferred.
"""
logger.info("### Doing Inference")

# !{sys.executable} -m pip install rdflib
# !{sys.executable} -m pip install git+https://github.com/RDFLib/OWL-RL.git

"""
Let's open the ontology file and see how many triplets it contains:
"""
logger.info("Let's open the ontology file and see how many triplets it contains:")


g = rdflib.Graph()
g.parse("onto.ttl", format="turtle")

logger.debug("Triplets found:%d" % len(g))

"""
Now let's build the closure, and see how the number of triplets increase:
"""
logger.info("Now let's build the closure, and see how the number of triplets increase:")

DeductiveClosure(OWLRL_Extension).expand(g)
logger.debug("Triplets after inference:%d" % len(g))

"""
### Querying for Relatives 

Now we can query the graph to see different relations between people. We can use **SPARQL** language together with `query` method. In our case, let's see all **uncles** in our family tree:
"""
logger.info("### Querying for Relatives")

qres = g.query(
    """SELECT DISTINCT ?aname ?bname
       WHERE {
          ?a fhkb:isUncleOf ?b .
          ?a rdfs:label ?aname .
          ?b rdfs:label ?bname .
       }""")

for row in qres:
    logger.debug("%s is uncle of %s" % row)

"""
Feel free to experiment with different other family relations. For example, you can have a look at `isAncestorOf` relation, which recurrently defines all ancestors of a given person.

Finally, let's clean up!
"""
logger.info("Feel free to experiment with different other family relations. For example, you can have a look at `isAncestorOf` relation, which recurrently defines all ancestors of a given person.")

# !rm onto.ttl

logger.info("\n\n[DONE]", bright=True)