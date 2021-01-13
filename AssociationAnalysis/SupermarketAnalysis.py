import pandas as pd
from apyori import apriori


# Source: https://drive.google.com/file/d/1y5DYn0dGoSbC22xowBq2d4po6h1JxcTQ/view?usp=sharing bzw. https: //stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/
store_data = pd.read_csv('store_data.csv', header=None)
#print(store_data.head())


records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])

# Removes the 'nan' fields
for i,j in enumerate(records):
    while 'nan' in records[i]: records[i].remove('nan')

#print(records)

# Run through the apriori algorithm
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)
print(len(association_results))
print(association_results[0])

# The output are association rules and the associated support, lift and confidence value
for item in association_results:
    pair = item[0]
    items = [x for x in pair]

    print("Rule: " + str(list(item.ordered_statistics[0].items_base)) + " -> " + str(list(item.ordered_statistics[0].items_add)))
    #second index of the inner list
    print("Support: " + str(item[1]))
    #third index of the list located at 0th
    #of the third index of the inner list
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")


