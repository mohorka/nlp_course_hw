from yargy import Parser, or_, rule
from yargy.interpretation import fact
from yargy.predicates import normalized

from task1.extractors.date_extractor import DATE
from task1.extractors.location_extractor import BIRTH_PLACE
from task1.extractors.name_extractor import NAME

Person = fact(
    "Person",
    ["name", "birth", "location"],
)

BIRTH = rule(
    rule(normalized("родиться")),
    or_(
        rule(
            BIRTH_PLACE.optional().interpretation(Person.location),
            DATE.optional().interpretation(Person.birth),
        ),
        rule(
            DATE.optional().interpretation(Person.birth),
            BIRTH_PLACE.optional().interpretation(Person.location),
        ),
    ),
)

PERSON = rule(NAME.interpretation(Person.name), BIRTH.optional()).interpretation(Person)

PersonParser = Parser(PERSON)
