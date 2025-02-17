import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def coinflip() -> int: 
    return random.randint(0,1)

def coinflips(amount:int):
    heads = 0
    tails = 0
    for v in range(amount):
        if coinflip() == 0:
            heads += 1
        else:
            tails += 1
    amount = [heads, tails]
    return amount

def show_bar_chart_with_coinflips(coinflip_amount: int):
    sides = ['Heads', 'Tails']
    amount = coinflips(coinflip_amount)

    plt.bar(sides, amount)
    plt.title('coinflips')
    plt.xlabel('Sides')
    plt.ylabel('Amount')
    plt.show()

def show_percentage_graph_per_coinflip(interval: int, total_tries: int):
    total_coinflips = []
    for v in range(total_tries):
        total_coinflips.append(coinflips(interval))
    chance_for_heads_per_interval = []
    for i in range(len(total_coinflips)):
        if i == 0:
            chance_for_heads_per_interval.append((total_coinflips[i][0] / (total_coinflips[i][0] + total_coinflips[i][1]))  * 100)
        else:
            chance_for_heads_per_interval.append(((chance_for_heads_per_interval[i-1])+(total_coinflips[i][0] / (total_coinflips[i][0] + total_coinflips[i][1])  * 100)) / 2)
    x_axis_tries = []
    for i in range(1,total_tries+1):
        x_axis_tries.append(i*interval)



    plt.figure(figsize=(10,6))
    plt.plot(x_axis_tries, chance_for_heads_per_interval, marker="o", linestyle="-", color="b", label="Chance of heads")
    plt.title(f"Chance per {interval} tries")
    plt.xlabel("Chance")
    plt.ylabel("Tries")
    plt.xticks(x_axis_tries)
    plt.title("Chance Per Tries")
    plt.ylim([40, 60])
    plt.grid(True)
    plt.legend()
    plt.show()


show_percentage_graph_per_coinflip(25, 20)

