from yargy import or_, rule
from yargy.interpretation import fact
from yargy.predicates import gram
from yargy.relations import gnc_relation

gnc = gnc_relation()

Name = fact("Name", ["first", "last"])


FIRSTNAME = gram("Name").interpretation(Name.first.inflected()).match(gnc)

LASTNAME = gram("Surn").interpretation(Name.last.inflected()).match(gnc)

NAME = or_(rule(FIRSTNAME, LASTNAME), rule(LASTNAME, FIRSTNAME)).interpretation(Name)
