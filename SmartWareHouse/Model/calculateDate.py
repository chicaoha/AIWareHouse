import datetime
def CaculateDate( checkinDate, checkoutDate) :
    # Calculate the number of days between checkinDate and checkoutDate
    days = checkoutDate - checkinDate
    return days

chechinDate = datetime.date(2020, 1, 1)
checkoutDate = datetime.date(2020, 1, 10)
print(CaculateDate(chechinDate, checkoutDate))
