import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

def get(person, one_gene, two_genes):
    if person in two_genes:
        return 1 - PROBS["mutation"]
    elif person in one_gene:
        return 0.5
    else:
        return PROBS["mutation"]


def joint_probability(people, one_gene, two_genes, have_trait):
    join_prob = 1
    for person in people:
        gene = 2 if person in two_genes else 1 if person in one_gene else 0
        trait = person in have_trait
        P = 1

        mother_name = people[person]["mother"]
        father_name = people[person]["father"]

        if mother_name is None and father_name is None:
             P *= PROBS["gene"][gene]
        else:
            inherit_mother = get(mother_name, one_gene, two_genes)
            inherit_father = get(father_name, one_gene, two_genes)

            if gene == 0:
                P *= (1 - inherit_mother) * (1 - inherit_father)
            elif gene == 1:
                P *= (inherit_father * (1 - inherit_mother) + inherit_mother * (1 - inherit_father))
            else:
                P *= inherit_father * inherit_mother
        P *= PROBS["trait"][gene][trait]
        join_prob *= P
    return join_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    for person in probabilities:
        gene = 2 if person in two_genes else 1 if person in one_gene else 0
        trait = person in have_trait

        probabilities[person]["gene"][gene] += p
        probabilities[person]["trait"][trait] += p


def normalize(probabilities):
    for person in probabilities:
        sum_gene = 0
        for val in probabilities[person]["gene"].values():
            sum_gene += val
        for i in range(0, 3):
            probabilities[person]["gene"][i] /= sum_gene
        sum_trait = 0
        for val in probabilities[person]["trait"].values():
            sum_trait += val
        for i in range(0, 2):
            probabilities[person]["trait"][i] /= sum_trait


if __name__ == "__main__":
    main()
