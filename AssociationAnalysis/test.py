import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules


dataset = [['Milk', 'Eggs', 'Bread'],
['Milk', 'Eggs'],
['Milk', 'Bread'],
['Eggs', 'Apple']]

print(dataset)

te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_array, columns=te.columns_)

print(df)

frequent_itemsets_ap = apriori(df, min_support=0.01, use_colnames=True)
print(frequent_itemsets_ap)

frequent_itemsets_fp=fpgrowth(df, min_support=0.01, use_colnames=True)
print(frequent_itemsets_fp)


rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.8)
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.8)
print(rules_ap)
print(rules_fp)