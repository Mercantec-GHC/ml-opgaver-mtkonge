import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

data = pd.read_csv('./Data/Police/police.csv')
sanitized_data = data.dropna(subset=["driver_gender", "violation", "stop_outcome"])
dates = sanitized_data["stop_date"].map(lambda date_str : datetime.datetime.strptime(date_str, "%Y-%m-%d"))
days = dates.map( lambda date : date.day)
months = dates.map( lambda date : date.month)
years = dates.map( lambda date : date.year)

sanitized_data["day"] = days
sanitized_data["month"] = months
sanitized_data["year"] = years

print(sanitized_data)


