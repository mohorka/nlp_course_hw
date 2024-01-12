from yargy import and_, or_, rule
from yargy.interpretation import fact
from yargy.predicates import caseless, dictionary, gte, lte, normalized

Date = fact("Date", ["year", "month", "day"])

MONTHS = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}

YEAR_WORDS = or_(rule(caseless("г"), "."), rule(normalized("год")))
DAY = and_(gte(1), lte(31)).interpretation(Date.day.custom(int))
MONTH = and_(gte(1), lte(12)).interpretation(Date.month.custom(int))
YEAR = and_(gte(1), lte(2018)).interpretation(Date.year.custom(int))
MONTH_NAME = dictionary(MONTHS).interpretation(
    Date.month.normalized().custom(MONTHS.__getitem__)
)

DATE = or_(
    rule(YEAR, "-", MONTH, "-", DAY),
    rule(DAY, MONTH_NAME, YEAR, YEAR_WORDS.optional()),
    rule(DAY, ".", MONTH, ".", YEAR, YEAR_WORDS.optional()),
).interpretation(Date)
