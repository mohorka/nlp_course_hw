from yargy import and_, not_, or_, rule
from yargy.interpretation import fact
from yargy.predicates import gram, is_capitalized
from yargy.relations import gnc_relation

gnc = gnc_relation()

Location = fact("Location", ["city"])

PLACE = (
    rule(
        and_(
            gram("Geox"),
            is_capitalized(),
            not_(
                or_(
                    gram("Abbr"),
                    gram("PREP"),
                    gram("CONJ"),
                    gram("PRCL"),
                ),
            ),
        )
    )
    .match(gnc)
    .interpretation(Location.city)
)

BIRTH_PLACE = rule(rule("в"), PLACE).interpretation(Location)
