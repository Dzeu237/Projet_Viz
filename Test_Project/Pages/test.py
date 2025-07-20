import random
import matplotlib.pyplot as plt

# Définir les valeurs possibles
values = [1, 3, 4, 6]

# Effectuer 1000 tirages aléatoires
results = [random.choice(values) for _ in range(1000)]

# Compter les occurrences de chaque valeur
count_1 = results.count(1)
count_3 = results.count(3)
count_4 = results.count(4)
count_6 = results.count(6)

# Afficher les résultats
print(f"Résultats après 1000 tirages aléatoires parmi les valeurs {values}:")
print(f"Valeur 1: {count_1} occurrences ({count_1/10}%)")
print(f"Valeur 3: {count_3} occurrences ({count_3/10}%)")
print(f"Valeur 4: {count_4} occurrences ({count_4/10}%)")
print(f"Valeur 6: {count_6} occurrences ({count_6/10}%)")
print(f"Total: {len(results)} tirages")

# Créer un graphique pour visualiser les résultats
labels = ['1', '3', '4', '6']
counts = [count_1, count_3, count_4, count_6]

plt.figure(figsize=(10, 6))

# Diagramme en barres
plt.subplot(1, 2, 1)
plt.bar(labels, counts, color=['blue', 'green', 'red', 'purple'])
plt.title('Répartition des 1000 tirages aléatoires')
plt.xlabel('Valeurs')
plt.ylabel('Nombre d\'occurrences')

# Diagramme circulaire
plt.subplot(1, 2, 2)
plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['blue', 'green', 'red', 'purple'])
plt.title('Pourcentage des valeurs tirées')

plt.tight_layout()
plt.show()