import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# 1. FASTA UCITAVANJE 
# ============================================================

def load_fasta(file_path):
    sequences = []
    seq = ""

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                seq = ""
            else:
                seq += line.upper()

    if seq:
        sequences.append(seq)

    return sequences


# ============================================================
# 2. UCITAVANJE PODATAKA IZ data/ FOLDERA
# ============================================================

DATA_DIR = "data"


def load_dataset():
    sequences = []
    labels = []

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            "Nije pronadjen folder data/. Napravi data/ i ubaci .fasta fajlove."
        )

    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".fasta")
    )

    if len(files) == 0:
        raise FileNotFoundError("U folderu data/ nema .fasta fajlova.")

    for filename in files:
        species_name = filename.replace(".fasta", "")
        path = os.path.join(DATA_DIR, filename)

        seqs = load_fasta(path)
        sequences.extend(seqs)
        labels.extend([species_name] * len(seqs))

        print(f"{filename}: {len(seqs)} sekvenci")

    print(f"Loaded {len(sequences)} sequences from {len(set(labels))} species")
    return sequences, labels


# ============================================================
# 3. K-MER FREKVENCIJA
# ============================================================

def kmer_frequency(sequence, k):
    kmers = [
        sequence[i:i+k]
        for i in range(len(sequence) - k + 1)
    ]
    return Counter(kmers)


def build_kmer_matrix(sequences, k):
    all_kmers = set()
    kmer_counts = []

    for seq in sequences:
        counts = kmer_frequency(seq, k)
        kmer_counts.append(counts)
        all_kmers.update(counts.keys())

    all_kmers = sorted(all_kmers)

    X = np.zeros((len(sequences), len(all_kmers)))

    for i, counts in enumerate(kmer_counts):
        for j, kmer in enumerate(all_kmers):
            X[i, j] = counts.get(kmer, 0)

    return X


# ============================================================
# 4. PCA VIZUALIZACIJA
# ============================================================

def run_pca(X, labels, title):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    for lab in sorted(set(labels)):
        idx = [i for i, x in enumerate(labels) if x == lab]
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=lab,
            alpha=0.7
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return X_scaled


# ============================================================
# 5. LOGISTIC REGRESSION + STATISTIKA
# ============================================================

def logistic_analysis(X_scaled, labels, title):
    unique_labels = sorted(set(labels))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_map[x] for x in labels])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=unique_labels,
        output_dict=True,
        zero_division=0
    )

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {report['macro avg']['precision']:.2f}")
    print(f"Recall   : {report['macro avg']['recall']:.2f}")
    print(f"F1-score : {report['macro avg']['f1-score']:.2f}")
    print("\nDetaljni classification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=unique_labels,
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=unique_labels
    ).plot()

    plt.title(title + " - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return {
        "accuracy": accuracy,
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"]
    }


# ============================================================
# ZADATAK 1 - k = 3, 4, 5, 6
# ============================================================

def task_1(sequences, labels):
    print("\n" + "#" * 60)
    print("ZADATAK 1 - PROMJENA k OD 3 DO 6")
    print("#" * 60)

    results = []

    for k in [3, 4, 5, 6]:
        X = build_kmer_matrix(sequences, k)

        X_scaled = run_pca(
            X,
            labels,
            f"PCA of Frog DNA k-mer Features (k={k})"
        )

        result = logistic_analysis(
            X_scaled,
            labels,
            f"Logistic Regression - k={k}"
        )

        results.append((k, result))

    print("\nREZULTATI ZADATKA 1")
    print("k\tAccuracy\tPrecision\tRecall\tF1")

    for k, r in results:
        print(
            f"{k}\t{r['accuracy']:.2f}\t\t"
            f"{r['precision']:.2f}\t\t"
            f"{r['recall']:.2f}\t{r['f1']:.2f}"
        )

    plt.figure(figsize=(8, 5))
    plt.bar(
        [k for k, r in results],
        [r["accuracy"] for k, r in results]
    )
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy prema vrijednosti k")
    plt.ylim(0, 1.05)
    plt.xticks([3, 4, 5, 6])
    plt.tight_layout()
    plt.show()

    return results


# ============================================================
# ZADATAK 2 - uklanjanje Rana temporaria
# ============================================================

def task_2(sequences, labels):
    print("\n" + "#" * 60)
    print("ZADATAK 2 - UKLANJANJE JEDNE VRSTE")
    print("#" * 60)

    remove = "Rana temporaria"

    # Ako se naziv razlikuje zbog naziva FASTA fajla,
    # pronalazi se vrsta koja sadrzi rijec rana.
    if remove not in set(labels):
        candidates = [x for x in set(labels) if "rana" in x.lower()]
        if candidates:
            remove = candidates[0]

    new_sequences = [
        seq for seq, lab in zip(sequences, labels)
        if lab != remove
    ]

    new_labels = [
        lab for lab in labels
        if lab != remove
    ]

    print(f"Uklonjena vrsta: {remove}")
    print(f"Preostale vrste: {sorted(set(new_labels))}")

    X = build_kmer_matrix(new_sequences, 4)

    X_scaled = run_pca(
        X,
        new_labels,
        "PCA - dvije vrste, k=4"
    )

    result = logistic_analysis(
        X_scaled,
        new_labels,
        "Logistic Regression - dvije vrste, k=4"
    )

    print("\nREZULTAT ZADATKA 2")
    print(f"Accuracy : {result['accuracy']:.2f}")
    print(f"Precision: {result['precision']:.2f}")
    print(f"Recall   : {result['recall']:.2f}")
    print(f"F1-score : {result['f1']:.2f}")

    return result


# ============================================================
# ZADATAK 3 - Random Forest
# ============================================================

def task_3(sequences, labels):
    print("\n" + "#" * 60)
    print("ZADATAK 3 - RANDOM FOREST")
    print("#" * 60)

    k = 4
    X = build_kmer_matrix(sequences, k)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    unique_labels = sorted(set(labels))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_map[x] for x in labels])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=unique_labels,
        output_dict=True,
        zero_division=0
    )

    rf_accuracy = accuracy_score(y_test, y_pred)

    print("\nRandom Forest rezultati:")
    print(f"Accuracy : {rf_accuracy:.2f}")
    print(f"Precision: {report['macro avg']['precision']:.2f}")
    print(f"Recall   : {report['macro avg']['recall']:.2f}")
    print(f"F1-score : {report['macro avg']['f1-score']:.2f}")

    print("\nDetaljni classification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=unique_labels,
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=unique_labels
    ).plot()

    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Logistic Regression za direktno poređenje
    lr_result = logistic_analysis(
        X_scaled,
        labels,
        "Logistic Regression - poređenje"
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Logistic Regression", "Random Forest"],
        [lr_result["accuracy"], rf_accuracy]
    )
    plt.ylabel("Accuracy")
    plt.title("Poređenje klasifikatora")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    print("\nPOREĐENJE")
    print(f"Logistic Regression: {lr_result['accuracy']:.2f}")
    print(f"Random Forest      : {rf_accuracy:.2f}")


# ============================================================
# GLAVNI PROGRAM
# ============================================================

def main():
    print("=" * 60)
    print("UuAP - CASE STUDY I")
    print("Analiza DNA sekvenci žaba pomoću k-mera")
    print("Student: Edin Kadric")
    print("=" * 60)

    sequences, labels = load_dataset()

    # Pocetna analiza iz dokumenta: k = 4
    X = build_kmer_matrix(sequences, 4)

    X_scaled = run_pca(
        X,
        labels,
        "PCA of Frog DNA k-mer Features (k=4)"
    )

    logistic_analysis(
        X_scaled,
        labels,
        "Logistic Regression - pocetna analiza k=4"
    )

    # Zadatak 1
    task_1(sequences, labels)

    # Zadatak 2
    task_2(sequences, labels)

    # Zadatak 3
    task_3(sequences, labels)

    print("\nAnaliza je završena.")


if __name__ == "__main__":
    main()
