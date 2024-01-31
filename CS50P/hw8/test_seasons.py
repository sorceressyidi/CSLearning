from seasons import Season, main
from datetime import date
import inflect
def test_season():
    assert Season.from_date(date.fromisoformat("2020-01-01"), date.today()).get_season() == f"Two million, one hundred forty-two thousand, seven hundred twenty minutes"
    
