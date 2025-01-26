#!/bin/bash

# Nom du fichier de sortie
output_file="resultats2.txt"

# Effacer le fichier de sortie s'il existe déjà
> "$output_file"

# Groupes d'arguments à exécuter
group1_args=(
  "100000 1000000"
  "1000000 1000000"
)

group2_args=(
  "10000000 1000000"
)

# Nombre d'itérations par groupe
iterations=10

echo "Début des tests..." | tee -a "$output_file"

# Exécution du premier groupe d'arguments
echo "Lancement des tests pour le premier groupe d'arguments..." | tee -a "$output_file"
for ((i = 1; i <= iterations; i++)); do
  echo "Itération $i pour le premier groupe :" | tee -a "$output_file"
  for args in "${group1_args[@]}"; do
    echo "Exécution de ./BSMADSMID avec les arguments $args..." | tee -a "$output_file"
    ./BSMADSMID $args 2>&1 | tee -a "$output_file"
  done
done

# Exécution du second groupe d'arguments
echo "Lancement des tests pour le second groupe d'arguments..." | tee -a "$output_file"
for ((i = 1; i <= iterations; i++)); do
  echo "Itération $i pour le second groupe :" | tee -a "$output_file"
  for args in "${group2_args[@]}"; do
    echo "Exécution de ./BSMADSMID avec les arguments $args..." | tee -a "$output_file"
    ./BSMADSMID $args 2>&1 | tee -a "$output_file"
  done
done

echo "Tests terminés." | tee -a "$output_file"