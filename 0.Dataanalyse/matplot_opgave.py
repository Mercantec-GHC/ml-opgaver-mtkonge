
import pandas as pd
import matplotlib.pyplot as plt

# Indlæs data fra CSV-filen
data = pd.read_csv('./Data/MatPlotLib-Data/sales.csv')
print(data.head())

plt.figure(figsize=(10,6))
plt.plot(data["month_number"], data["total_profit"], marker="o", linestyle="-", color="b", label="Total Profit")
plt.title("Total Profit Per Month")
plt.xlabel("Month Number")
plt.ylabel("Total Profit")
plt.xticks(data["month_number"])
plt.title("Total Profit Per Month")
plt.grid(True)
plt.legend()
plt.show()