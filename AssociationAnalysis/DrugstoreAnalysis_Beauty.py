import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

# Sample Dataset with beauty products
dataset = [ ['Lippenstift Cherry'], ['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],['Lippenstift Cherry'],
            ['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],['Lipliner Cherry'],
            ['Lipliner Cherry', 'Lippenstift Cherry'], ['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],['Lipliner Cherry', 'Lippenstift Cherry'],
            ['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],['Lippenstift Rot'],
            ['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],['Lippenstift Rot', 'Lippenstift Cherry'],
            ['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],['Lippenstift Rot', 'Lipliner Rot'],
            ['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],['Lippenstift Rot', 'Lipliner Rot', 'Lipscrub Glow'],
            ['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],['Lipscrub Glow'],
            ['Lippenstift Cherry', 'Lipscrub Glow'], ['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],['Lippenstift Cherry', 'Lipscrub Glow'],
            ]

#print(dataset)

# TransactionEncoder transforms the dataset into an array with boolean values for the sake of memory efficiency
# Source: http://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/
te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
# The dataset is saved in a DataFrame
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)

# ---------------------------------- Apriori ----------------------------------------------
print("----------------- APRIORI Algorithm -------------------")

# Run through the apriori algorithm
frequent_itemsets_ap = apriori(df, min_support=0.0045, use_colnames=True)
#frequent_itemsets_ap['length'] = frequent_itemsets_ap['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets_ap)

# Form the association rules with the metric confidence
rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.8)
print("Association rules: ")
print(rules_ap)


# ---------------------------------- FP Growth --------------------------------------------------------------------------------
print("----------------- FP-Growth Algorithm -----------------")

# Run through the fp growth algorithm
frequent_itemsets_fp=fpgrowth(df, min_support=0.0045, use_colnames=True)
print(frequent_itemsets_fp)

# Form the association rules with the metric confidence
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.8)
print("Association rules: ")
print(rules_fp)



