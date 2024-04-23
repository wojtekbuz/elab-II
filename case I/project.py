import csv
import pandas as pd

with open("supermarket.csv", "r", newline="") as csvfile:
    reader = csv.reader(csvfile)
    rows = []

    for row in reader:
        row = row[:]
        rows.append(row)

df = pd.DataFrame(rows)

department_spend = {}
department_items = {}
department_time = {}
total_items = 0
total_shop_time = 0

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    previous_department = None
    previous_time = 0
    total_spend = 0
    items_bought = 0

    # Iterate over each tuple in the row
    for item in row:
        # Split the tuple
        if item and len(item.split()) == 3:
            department, time, price = item.split()
            time = int(time)
            price = float(price)

        else:
            pass

        if department in department_spend:
            department_spend[department].append(price)
            department_items[department] += 1
        else:
            department_spend[department] = [price]
            department_items[department] = 1

        if department in department_time:
            department_time[department].append(time - previous_time)
        else:
            department_time[department] = [time - previous_time]

        total_spend += price
        items_bought += 1

        previous_department = department
        previous_time = time

    total_items += items_bought
    total_shop_time += previous_time

total_items_bought = sum(department_items.values())

average_spend_per_department = {
    dept: sum(spend) / department_items[dept]
    for dept, spend in department_spend.items()
}
average_spend_per_item_per_department = {
    dept: sum(spend) / total_items_bought for dept, spend in department_spend.items()
}
average_travel_time = {
    dept: sum(time) / department_items[dept] for dept, time in department_time.items()
}
average_items_per_department = {
    dept: department_items[dept] / len(department_items) for dept in department_items
}

new_df = pd.DataFrame(
    {
        "Department": list(department_spend.keys()),
        "Average Spend per Department": list(average_spend_per_department.values()),
        "Average Spend per Item per Department": list(
            average_spend_per_item_per_department.values()
        ),
        "Average Travel Time between Departments": list(average_travel_time.values()),
        "Average Items Bought per Department": list(
            average_items_per_department.values()
        ),
    }
)

print(df.head(5))
print(new_df)
